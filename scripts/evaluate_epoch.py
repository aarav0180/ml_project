#!/usr/bin/env python
"""Epoch-based evaluation script for LWC model on WELFake dataset.

This script divides the WELFake dataset into chunks and evaluates the model
on each chunk (epoch), tracking both epoch-level and overall accuracy.
"""

import sys
import logging
from pathlib import Path
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
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


def load_and_prepare_welfake(welfake_path: str, chunk_size: int = 11000) -> list:
    """
    Load WELFake dataset, filter invalid data, and divide into balanced chunks.
    
    Args:
        welfake_path: Path to WELFake CSV file
        chunk_size: Target size for each chunk (will be balanced between true/fake)
        
    Returns:
        List of DataFrames, each containing a balanced chunk
    """
    logger.info(f"Loading WELFake dataset from: {welfake_path}")
    
    try:
        welfake_df = pd.read_csv(welfake_path)
        logger.info(f"WELFake dataset loaded. Shape: {welfake_df.shape}")
        logger.info(f"Columns: {list(welfake_df.columns)}")
    except FileNotFoundError:
        logger.error(f"Error: {welfake_path} not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading WELFake dataset: {e}")
        sys.exit(1)
    
    # Detect columns
    label_column = None
    for col in welfake_df.columns:
        if col.lower() in ['label', 'labels']:
            label_column = col
            break
    
    if label_column is None:
        logger.error("Error: Could not find 'Label' or 'label' column.")
        sys.exit(1)
    
    text_column = None
    for preferred in ['text', 'Text', 'title', 'Title', 'content', 'Content']:
        if preferred in welfake_df.columns:
            text_column = preferred
            break
    
    if text_column is None:
        logger.error("Error: Could not find text column.")
        sys.exit(1)
    
    logger.info(f"Using label column: '{label_column}', text column: '{text_column}'")
    
    # Clean data: handle NaN/None values
    logger.info("Cleaning and filtering data...")
    welfake_df = welfake_df.copy()
    welfake_df[text_column] = welfake_df[text_column].fillna('').astype(str)
    
    # Filter out empty/invalid texts
    valid_mask = welfake_df[text_column].str.strip().astype(bool)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Filtering out {invalid_count} empty/invalid texts")
        welfake_df = welfake_df[valid_mask].reset_index(drop=True)
    
    logger.info(f"Valid data after filtering: {len(welfake_df)} samples")
    
    # Invert labels: WELFake uses 0=fake, 1=real -> Model expects 0=real, 1=fake
    welfake_df['inverted_label'] = welfake_df[label_column].apply(lambda x: 1 - x)
    
    # Show label distribution
    logger.info(f"Label distribution - Original: {welfake_df[label_column].value_counts().to_dict()}")
    logger.info(f"Label distribution - Inverted: {welfake_df['inverted_label'].value_counts().to_dict()}")
    
    # Separate by label for balanced chunks
    real_df = welfake_df[welfake_df['inverted_label'] == 0].copy()
    fake_df = welfake_df[welfake_df['inverted_label'] == 1].copy()
    
    logger.info(f"Real news: {len(real_df)}, Fake news: {len(fake_df)}")
    
    # Shuffle each group
    real_df = real_df.sample(frac=1, random_state=42).reset_index(drop=True)
    fake_df = fake_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create balanced chunks
    samples_per_label = chunk_size // 2
    chunks = []
    total_chunks = min(len(real_df) // samples_per_label, len(fake_df) // samples_per_label)
    
    logger.info(f"Creating {total_chunks} balanced chunks (each with ~{samples_per_label} real + ~{samples_per_label} fake)")
    
    for i in range(total_chunks):
        start_real = i * samples_per_label
        end_real = (i + 1) * samples_per_label
        start_fake = i * samples_per_label
        end_fake = (i + 1) * samples_per_label
        
        chunk_real = real_df.iloc[start_real:end_real].copy()
        chunk_fake = fake_df.iloc[start_fake:end_fake].copy()
        
        chunk = pd.concat([chunk_real, chunk_fake], ignore_index=True)
        chunk = chunk.sample(frac=1, random_state=42 + i).reset_index(drop=True)  # Shuffle chunk
        
        chunks.append({
            'data': chunk,
            'text_column': text_column,
            'label_column': 'inverted_label',
            'epoch': i + 1
        })
        
        logger.info(f"Chunk {i+1}: {len(chunk)} samples ({chunk['inverted_label'].value_counts().to_dict()})")
    
    return chunks


def evaluate_chunk(chunk_data: dict, processor: AdvancedDataProcessor, 
                   model: AdvancedFakeNewsModel, evaluator: AdvancedEvaluator,
                   config: dict, device: torch.device) -> dict:
    """
    Evaluate model on a single chunk.
    
    Args:
        chunk_data: Dictionary containing chunk data and metadata
        processor: Data processor instance
        model: Trained model
        evaluator: Evaluator instance
        config: Configuration dictionary
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    chunk_df = chunk_data['data']
    text_column = chunk_data['text_column']
    label_column = chunk_data['label_column']
    epoch = chunk_data['epoch']
    
    logger.info(f"\n{'='*70}")
    logger.info(f"EPOCH {epoch} - Evaluating {len(chunk_df)} samples")
    logger.info(f"{'='*70}")
    
    # Prepare data
    texts = chunk_df[text_column].tolist()
    labels = chunk_df[label_column].tolist()
    
    logger.info("Processing chunk data...")
    sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, labels_tensor = \
        processor.prepare_dataset(texts, labels)
    
    # Create dataset and dataloader
    chunk_dataset = AdvancedDataset(
        sentence_ids, sentence_mask,
        article_ids, article_mask, sentence_texts, labels_tensor
    )
    
    chunk_dataloader = DataLoader(
        chunk_dataset,
        sampler=SequentialSampler(chunk_dataset),
        batch_size=config['training']['batch_size'],
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn
    )
    
    # Evaluate
    logger.info("Running evaluation...")
    metrics, report, results = evaluator.evaluate(chunk_dataloader)
    
    return {
        'epoch': epoch,
        'samples': len(chunk_df),
        'metrics': metrics,
        'report': report,
        'results': {
            'predictions': results['predictions'].tolist(),
            'labels': results['labels'].tolist(),
            'probabilities': results['probabilities'].tolist()
        }
    }


def save_progress(progress_file: str, all_results: list, overall_metrics: dict):
    """Save evaluation progress to file."""
    progress_data = {
        'epoch_results': all_results,
        'overall_metrics': overall_metrics,
        'summary': {
            'total_epochs': len(all_results),
            'total_samples': sum(r['samples'] for r in all_results)
        }
    }
    
    with open(progress_file, 'wb') as f:
        pickle.dump(progress_data, f)
    
    # Also save as JSON (without numpy arrays)
    json_data = {
        'epoch_results': [
            {
                'epoch': r['epoch'],
                'samples': r['samples'],
                'metrics': {k: float(v) for k, v in r['metrics'].items()}
            }
            for r in all_results
        ],
        'overall_metrics': {k: float(v) for k, v in overall_metrics.items()},
        'summary': progress_data['summary']
    }
    
    json_file = progress_file.replace('.pkl', '.json')
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Progress saved to {progress_file} and {json_file}")


def calculate_overall_metrics(all_results: list) -> dict:
    """Calculate overall metrics across all epochs."""
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    for result in all_results:
        all_predictions.extend(result['results']['predictions'])
        all_labels.extend(result['results']['labels'])
        all_probabilities.extend([p[1] for p in result['results']['probabilities']])  # Fake class probability
    
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    overall_f1 = f1_score(all_labels, all_predictions)
    overall_roc_auc = roc_auc_score(all_labels, all_probabilities)
    
    return {
        'accuracy': overall_accuracy,
        'f1_score': overall_f1,
        'roc_auc': overall_roc_auc,
        'total_samples': len(all_labels)
    }


def main():
    """Main evaluation function."""
    # Setup
    setup_nltk_data()
    config = load_config()
    device = get_device()
    
    # Load and prepare chunks
    welfake_path = config.get('evaluation', {}).get('welfake_data_path', 'data/welFake.csv')
    chunks = load_and_prepare_welfake(welfake_path, chunk_size=11000)
    
    if not chunks:
        logger.error("No valid chunks created. Exiting.")
        sys.exit(1)
    
    # Initialize processor
    logger.info("\nInitializing data processor...")
    processor = AdvancedDataProcessor(
        tokenizer_name=config['model']['bert_model_name'],
        max_sentences=config['data_processing']['max_sentences'],
        max_tokens_per_sentence=config['data_processing']['max_tokens_per_sentence'],
        max_tokens_per_article=config['data_processing']['max_tokens_per_article'],
        min_tokens_per_sentence=config['data_processing']['min_tokens_per_sentence']
    )
    
    # Initialize model
    logger.info("Initializing model...")
    logger.info("Loading BERT and plausibility models (this may take a moment)...")
    model = AdvancedFakeNewsModel(
        bert_model_name=config['model']['bert_model_name'],
        freeze_bert=config['model']['freeze_bert'],
        plausibility_model_path=config['model']['plausibility_model_path']
    )
    
    model_weights_path = config['paths']['model_save_path']
    if not Path(model_weights_path).exists():
        logger.error(f"Model weights not found at {model_weights_path}")
        logger.error("Please train the model first using scripts/train.py")
        sys.exit(1)
    
    logger.info(f"Loading model weights from {model_weights_path}...")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator(model, device=device)
    
    # Evaluate each chunk (epoch)
    all_results = []
    progress_file = 'evaluation_progress.pkl'
    
    logger.info(f"\n{'='*70}")
    logger.info(f"STARTING EPOCH-BASED EVALUATION")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"{'='*70}")
    
    for chunk_data in chunks:
        result = evaluate_chunk(chunk_data, processor, model, evaluator, config, device)
        all_results.append(result)
        
        # Print epoch results
        logger.info(f"\nEpoch {result['epoch']} Results:")
        logger.info(f"  Samples: {result['samples']}")
        logger.info(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        logger.info(f"  F1-Score: {result['metrics']['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {result['metrics']['roc_auc']:.4f}")
        
        # Calculate and save overall progress
        overall_metrics = calculate_overall_metrics(all_results)
        save_progress(progress_file, all_results, overall_metrics)
        
        # Print comparison
        logger.info(f"\nOverall Progress (across {len(all_results)} epochs):")
        logger.info(f"  Total Samples: {overall_metrics['total_samples']}")
        logger.info(f"  Overall Accuracy: {overall_metrics['accuracy']:.4f}")
        logger.info(f"  Overall F1-Score: {overall_metrics['f1_score']:.4f}")
        logger.info(f"  Overall ROC-AUC: {overall_metrics['roc_auc']:.4f}")
        logger.info(f"  Epoch {result['epoch']} Accuracy: {result['metrics']['accuracy']:.4f}")
        logger.info(f"  Difference: {overall_metrics['accuracy'] - result['metrics']['accuracy']:+.4f}")
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total Epochs: {len(all_results)}")
    logger.info(f"Total Samples Evaluated: {overall_metrics['total_samples']}")
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score: {overall_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC: {overall_metrics['roc_auc']:.4f}")
    
    logger.info(f"\nPer-Epoch Accuracies:")
    for result in all_results:
        logger.info(f"  Epoch {result['epoch']}: {result['metrics']['accuracy']:.4f}")
    
    logger.info(f"\nProgress saved to: {progress_file}")


if __name__ == '__main__':
    main()

