#!/usr/bin/env python
"""Evaluation script for LWC model on test set and WELFake dataset."""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from lwc.utils.config import load_config
from lwc.utils.device import get_device
from lwc.utils.nltk_setup import setup_nltk_data
from lwc.models.advanced_model import AdvancedFakeNewsModel
from lwc.data.processor import AdvancedDataProcessor
from lwc.data.dataset import AdvancedDataset, custom_collate_fn
from lwc.training.evaluator import AdvancedEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_welfake_data(welfake_path: str) -> tuple[pd.DataFrame, str, str]:
    """
    Load WELFake dataset and detect column names.
    
    Args:
        welfake_path: Path to WELFake CSV file
        
    Returns:
        Tuple of (dataframe, label_column, text_column)
    """
    logger.info(f"Loading WELFake dataset from: {welfake_path}")
    
    try:
        welfake_df = pd.read_csv(welfake_path)
        logger.info(f"WELFake dataset loaded successfully. Shape: {welfake_df.shape}")
        logger.info(f"Columns: {list(welfake_df.columns)}")
    except FileNotFoundError:
        logger.error(f"Error: {welfake_path} not found. Please ensure it exists.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading WELFake dataset: {e}")
        sys.exit(1)
    
    # Detect label column (case-insensitive)
    label_column = None
    for col in welfake_df.columns:
        if col.lower() in ['label', 'labels']:
            label_column = col
            break
    
    if label_column is None:
        logger.error("Error: Could not find 'Label' or 'label' column in WELFake.csv.")
        logger.error(f"Available columns: {list(welfake_df.columns)}")
        sys.exit(1)
    
    # Detect text column (case-insensitive, prefer 'text' over 'title')
    text_column = None
    for preferred in ['text', 'Text', 'title', 'Title', 'content', 'Content']:
        if preferred in welfake_df.columns:
            text_column = preferred
            break
    
    if text_column is None:
        logger.error("Error: Could not find text column ('text', 'Text', 'title', 'Title', 'content', or 'Content') in WELFake.csv.")
        logger.error(f"Available columns: {list(welfake_df.columns)}")
        sys.exit(1)
    
    logger.info(f"Using label column: '{label_column}', text column: '{text_column}'")
    
    return welfake_df, label_column, text_column


def prepare_welfake_labels(welfake_df: pd.DataFrame, label_column: str) -> pd.Series:
    """
    Prepare WELFake labels by inverting if necessary.
    
    WELFake uses: 0=fake, 1=real (original)
    Model expects: 0=real, 1=fake
    
    Args:
        welfake_df: WELFake dataframe
        label_column: Name of the label column
        
    Returns:
        Series with inverted labels
    """
    labels = welfake_df[label_column].copy()
    
    # Invert labels: 0=fake, 1=real -> 0=real, 1=fake
    inverted_labels = labels.apply(lambda x: 1 - x)
    
    logger.info(f"Label distribution - Original: {labels.value_counts().to_dict()}")
    logger.info(f"Label distribution - Inverted: {inverted_labels.value_counts().to_dict()}")
    logger.info("Labels inverted to match model's expectation (0=real, 1=fake)")
    
    return inverted_labels


def evaluate_welfake(config: dict, device: torch.device):
    """Evaluate model on WELFake dataset."""
    logger.info("="*70)
    logger.info("EVALUATING MODEL ON WELFAKE DATASET")
    logger.info("="*70)
    
    # Load WELFake data
    welfake_path = config.get('evaluation', {}).get('welfake_data_path', 'data/welFake.csv')
    welfake_df, label_column, text_column = load_welfake_data(welfake_path)
    
    # Prepare labels
    welfake_labels = prepare_welfake_labels(welfake_df, label_column)
    
    # Initialize data processor
    logger.info("Initializing data processor for WELFake...")
    processor = AdvancedDataProcessor(
        tokenizer_name=config['model']['bert_model_name'],
        max_sentences=config['data_processing']['max_sentences'],
        max_tokens_per_sentence=config['data_processing']['max_tokens_per_sentence'],
        max_tokens_per_article=config['data_processing']['max_tokens_per_article'],
        min_tokens_per_sentence=config['data_processing']['min_tokens_per_sentence']
    )
    
    # Process WELFake data
    logger.info("Processing WELFake data...")
    # Clean data: handle NaN/None values and ensure strings
    logger.info("Cleaning data...")
    welfake_texts = welfake_df[text_column].fillna('').astype(str).tolist()
    welfake_labels_list = welfake_labels.tolist()
    
    # Filter out empty texts and corresponding labels
    logger.info("Filtering invalid texts...")
    valid_indices = [i for i, text in tqdm(enumerate(welfake_texts), total=len(welfake_texts), desc="Validating texts") if text and text.strip()]
    if len(valid_indices) < len(welfake_texts):
        logger.warning(f"Filtering out {len(welfake_texts) - len(valid_indices)} empty/invalid texts")
        welfake_texts = [welfake_texts[i] for i in valid_indices]
        welfake_labels_list = [welfake_labels_list[i] for i in valid_indices]
    
    logger.info(f"Processing {len(welfake_texts)} valid articles...")
    welfake_sentence_ids, welfake_sentence_mask, welfake_article_ids, welfake_article_mask, welfake_sentence_texts, welfake_labels_tensor = \
        processor.prepare_dataset(welfake_texts, welfake_labels_list)
    logger.info("WELFake data processing complete")
    
    # Create dataset
    welfake_dataset = AdvancedDataset(
        welfake_sentence_ids, welfake_sentence_mask,
        welfake_article_ids, welfake_article_mask, welfake_sentence_texts, welfake_labels_tensor
    )
    
    # Create DataLoader
    logger.info("Creating DataLoader for WELFake...")
    welfake_dataloader = DataLoader(
        welfake_dataset,
        sampler=SequentialSampler(welfake_dataset),
        batch_size=config['training']['batch_size'],
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn
    )
    logger.info("WELFake DataLoader created")
    
    # Initialize model
    logger.info("Initializing model for evaluation...")
    logger.info("Loading BERT and plausibility models (this may take a moment)...")
    model_eval = AdvancedFakeNewsModel(
        bert_model_name=config['model']['bert_model_name'],
        freeze_bert=config['model']['freeze_bert'],
        plausibility_model_path=config['model']['plausibility_model_path']
    )
    
    # Load pre-trained weights
    model_weights_path = config['paths']['model_save_path']
    if not Path(model_weights_path).exists():
        logger.error(f"Model weights not found at {model_weights_path}")
        logger.error("Please train the model first using scripts/train.py")
        sys.exit(1)
    
    logger.info(f"Loading model weights from {model_weights_path}...")
    model_eval.load_state_dict(torch.load(model_weights_path, map_location=device))
    model_eval.to(device)
    model_eval.eval()
    logger.info("Model initialized and weights loaded")
    
    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator_welfake = AdvancedEvaluator(model_eval, device=device)
    logger.info("Evaluator initialized")
    
    # Run evaluation
    logger.info("Running evaluation on WELFake dataset...")
    welfake_metrics, welfake_report, _ = evaluator_welfake.evaluate(welfake_dataloader)
    logger.info("Evaluation complete")
    
    # Print metrics
    logger.info("="*70)
    logger.info("WELFAKE EVALUATION METRICS:")
    logger.info(f"  Accuracy: {welfake_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score: {welfake_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC: {welfake_metrics['roc_auc']:.4f}")
    logger.info(f"  Mean Inconsistency (Real): {welfake_metrics['mean_inconsistency_real']:.4f}")
    logger.info(f"  Mean Inconsistency (Fake): {welfake_metrics['mean_inconsistency_fake']:.4f}")
    logger.info(f"  Std Inconsistency (Real): {welfake_metrics['std_inconsistency_real']:.4f}")
    logger.info(f"  Std Inconsistency (Fake): {welfake_metrics['std_inconsistency_fake']:.4f}")
    logger.info(f"  Mean Plausibility (Real): {welfake_metrics['mean_plausibility_real']:.4f}")
    logger.info(f"  Mean Plausibility (Fake): {welfake_metrics['mean_plausibility_fake']:.4f}")
    logger.info(f"  Overall Mean Plausibility: {welfake_metrics['mean_plausibility']:.4f}")
    logger.info(f"  Uncertain Ratio: {welfake_metrics['uncertain_ratio']:.2%}")
    logger.info("="*70)
    
    # Print classification report
    logger.info("\nClassification Report for WELFake Dataset:")
    logger.info(welfake_report)


def main():
    """Main evaluation function."""
    # Setup NLTK data
    setup_nltk_data()
    
    # Load configuration
    config = load_config()
    
    # Get device
    device = get_device()
    
    # Evaluate on WELFake
    evaluate_welfake(config, device)


if __name__ == '__main__':
    main()

