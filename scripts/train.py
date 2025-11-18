#!/usr/bin/env python
"""Training script for LWC model."""

import sys
import os
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from lwc.utils.config import load_config
from lwc.utils.device import get_device
from lwc.utils.nltk_setup import setup_nltk_data
from lwc.models.advanced_model import AdvancedFakeNewsModel
from lwc.data.loader import load_data, prepare_data
from lwc.data.processor import AdvancedDataProcessor
from lwc.data.dataset import AdvancedDataset, custom_collate_fn
from lwc.training.trainer import AdvancedTrainer
from lwc.training.evaluator import AdvancedEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    # Setup NLTK data
    setup_nltk_data()
    
    # Load configuration
    config = load_config()
    
    # Get device
    device = get_device()
    
    # Load and prepare data
    logger.info("Loading data...")
    data = load_data(
        true_data_path=config['data']['true_data_path'],
        fake_data_path=config['data']['fake_data_path']
    )
    logger.info(f"Data loaded. Shape: {data.shape}")
    
    # Use 'text' column for advanced model (full article text)
    text_column = config['data']['text_column']
    if text_column not in data.columns:
        logger.warning(f"'{text_column}' column not found. Using 'title' instead.")
        text_column = 'title'
    
    train_text, val_text, test_text, train_labels, val_labels, test_labels = prepare_data(
        data,
        text_column=text_column,
        test_size=config['training']['test_size'],
        val_size=config['training']['val_size'],
        random_state=config['training']['random_state']
    )
    
    logger.info(f"Data split - Train: {len(train_text)}, Val: {len(val_text)}, Test: {len(test_text)}")
    
    # Initialize data processor
    logger.info("Initializing data processor...")
    processor = AdvancedDataProcessor(
        tokenizer_name=config['model']['bert_model_name'],
        max_sentences=config['data_processing']['max_sentences'],
        max_tokens_per_sentence=config['data_processing']['max_tokens_per_sentence'],
        max_tokens_per_article=config['data_processing']['max_tokens_per_article'],
        min_tokens_per_sentence=config['data_processing']['min_tokens_per_sentence']
    )
    
    # Process data
    logger.info("Processing training data...")
    train_sentence_ids, train_sentence_mask, train_article_ids, train_article_mask, train_sentence_texts, train_labels_tensor = \
        processor.prepare_dataset(train_text.tolist(), train_labels.tolist())
    
    logger.info("Processing validation data...")
    val_sentence_ids, val_sentence_mask, val_article_ids, val_article_mask, val_sentence_texts, val_labels_tensor = \
        processor.prepare_dataset(val_text.tolist(), val_labels.tolist())
    
    logger.info("Processing test data...")
    test_sentence_ids, test_sentence_mask, test_article_ids, test_article_mask, test_sentence_texts, test_labels_tensor = \
        processor.prepare_dataset(test_text.tolist(), test_labels.tolist())
    
    logger.info("Data processing complete")
    
    # Create datasets
    train_dataset = AdvancedDataset(
        train_sentence_ids, train_sentence_mask,
        train_article_ids, train_article_mask, train_sentence_texts, train_labels_tensor
    )
    val_dataset = AdvancedDataset(
        val_sentence_ids, val_sentence_mask,
        val_article_ids, val_article_mask, val_sentence_texts, val_labels_tensor
    )
    test_dataset = AdvancedDataset(
        test_sentence_ids, test_sentence_mask,
        test_article_ids, test_article_mask, test_sentence_texts, test_labels_tensor
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config['training']['batch_size'],
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=config['training']['batch_size'],
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=config['training']['batch_size'],
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn
    )
    logger.info("Data loaders created")
    
    # Create model
    logger.info("Creating advanced model...")
    model = AdvancedFakeNewsModel(
        bert_model_name=config['model']['bert_model_name'],
        freeze_bert=config['model']['freeze_bert'],
        plausibility_model_path=config['model']['plausibility_model_path']
    )
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Initialize trainer and evaluator
    trainer = AdvancedTrainer(
        model,
        optimizer,
        device=device,
        lambda1=config['training']['lambda1'],
        lambda2=config['training']['lambda2']
    )
    evaluator = AdvancedEvaluator(model, device=device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")
    
    # Training loop
    logger.info("="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    
    best_val_accuracy = 0.0
    
    # Ensure model save directory exists
    model_save_path = Path(config['paths']['model_save_path'])
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        logger.info(f"{'='*70}")
        
        # Train with progress bar
        logger.info("Training...")
        train_losses = trainer.train_epoch(train_dataloader, show_progress=True)
        train_time = time.time() - epoch_start_time
        
        logger.info(f"Training Losses:")
        logger.info(f"  Classification: {train_losses['classification']:.4f}")
        logger.info(f"  Separation: {train_losses['separation']:.4f}")
        logger.info(f"  Total: {train_losses['total']:.4f}")
        logger.info(f"  Training Time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
        
        # Evaluate
        logger.info("Evaluating...")
        eval_start_time = time.time()
        val_metrics, val_report, _ = evaluator.evaluate(val_dataloader)
        eval_time = time.time() - eval_start_time
        
        logger.info(f"Validation Metrics:")
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {val_metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
        logger.info(f"  Mean Inconsistency (Real): {val_metrics['mean_inconsistency_real']:.4f}")
        logger.info(f"  Mean Inconsistency (Fake): {val_metrics['mean_inconsistency_fake']:.4f}")
        logger.info(f"  Mean Plausibility (Real): {val_metrics['mean_plausibility_real']:.4f}")
        logger.info(f"  Mean Plausibility (Fake): {val_metrics['mean_plausibility_fake']:.4f}")
        logger.info(f"  Uncertain Ratio: {val_metrics['uncertain_ratio']:.2%}")
        logger.info(f"  Evaluation Time: {eval_time:.2f} seconds")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            torch.save(model.state_dict(), config['paths']['model_save_path'])
            logger.info(f"Model saved with validation accuracy: {best_val_accuracy:.4f}")
        
        epoch_total_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} Total Time: {epoch_total_time:.2f} seconds ({epoch_total_time/60:.2f} minutes)")
    
    logger.info("="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    
    # Evaluate on test set
    logger.info("="*70)
    logger.info("Evaluating on test set...")
    logger.info("="*70)
    
    model.load_state_dict(torch.load(config['paths']['model_save_path'], map_location=device))
    
    test_metrics, test_report, _ = evaluator.evaluate(test_dataloader)
    
    logger.info(f"Test Set Metrics:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"  Mean Inconsistency (Real): {test_metrics['mean_inconsistency_real']:.4f}")
    logger.info(f"  Mean Inconsistency (Fake): {test_metrics['mean_inconsistency_fake']:.4f}")
    logger.info(f"  Std Inconsistency (Real): {test_metrics['std_inconsistency_real']:.4f}")
    logger.info(f"  Std Inconsistency (Fake): {test_metrics['std_inconsistency_fake']:.4f}")
    logger.info(f"  Mean Plausibility (Real): {test_metrics['mean_plausibility_real']:.4f}")
    logger.info(f"  Mean Plausibility (Fake): {test_metrics['mean_plausibility_fake']:.4f}")
    logger.info(f"  Overall Mean Plausibility: {test_metrics['mean_plausibility']:.4f}")
    logger.info(f"  Uncertain Ratio: {test_metrics['uncertain_ratio']:.2%}")
    
    logger.info(f"\nClassification Report:")
    logger.info(test_report)
    
    logger.info(f"Model saved to: {config['paths']['model_save_path']}")


if __name__ == '__main__':
    main()

