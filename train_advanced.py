"""
Training script for Advanced Fake News Detection Model with Constraint-based Consistency.
"""
import os
import time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from src.data import load_data, prepare_data
from src.data.advanced_data_loader import AdvancedDataProcessor
from src.models.advanced_model import AdvancedFakeNewsModel
from src.training.advanced_trainer import AdvancedTrainer
from src.training.advanced_evaluator import AdvancedEvaluator


class AdvancedDataset(torch.utils.data.Dataset):
    """Dataset class for advanced model."""
    
    def __init__(self, sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, labels):
        self.sentence_ids = sentence_ids
        self.sentence_mask = sentence_mask
        self.article_ids = article_ids
        self.article_mask = article_mask
        self.sentence_texts = sentence_texts  # Raw sentence texts for plausibility scoring
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.sentence_ids[idx],
            self.sentence_mask[idx],
            self.article_ids[idx],
            self.article_mask[idx],
            self.sentence_texts[idx],  # Include sentence texts
            self.labels[idx]
        )


def main():
    # Configuration
    config = {
        'data': {
            'true_data_path': 'data/True.csv',
            'fake_data_path': 'data/Fake.csv',
            'text_column': 'text'  # Use full text for advanced model
        },
        'model': {
            'bert_model_name': 'bert-base-uncased',
            'freeze_bert': True,
            'plausibility_model_path': 'plausability_model_final'
        },
        'data_processing': {
            'max_sentences': 12,  # Reduced from 20 for speed
            'max_tokens_per_sentence': 48,  # Reduced from 64 for speed
            'max_tokens_per_article': 256,
            'min_tokens_per_sentence': 5
        },
        'training': {
            'batch_size': 8,  # Increased batch size for better GPU utilization
            'learning_rate': 2e-5,
            'epochs': 1,  # Single epoch for faster training
            'test_size': 0.3,
            'val_size': 0.5,
            'random_state': 2018,
            'lambda1': 0.1,  # Real news consistency weight
            'lambda2': 0.1   # Fake news consistency weight
        },
        'paths': {
            'model_save_path': 'models/advanced_model_weights.pt'
        }
    }
    
    # Set device - force CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print(f"Warning: CUDA not available. Using device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and prepare data
    print("\nLoading data...")
    data = load_data(
        true_data_path=config['data']['true_data_path'],
        fake_data_path=config['data']['fake_data_path']
    )
    print(f"Data shape: {data.shape}")
    
    # Use 'text' column for advanced model (full article text)
    text_column = config['data']['text_column']
    if text_column not in data.columns:
        print(f"Warning: '{text_column}' column not found. Using 'title' instead.")
        text_column = 'title'
    
    train_text, val_text, test_text, train_labels, val_labels, test_labels = prepare_data(
        data,
        text_column=text_column,
        test_size=config['training']['test_size'],
        val_size=config['training']['val_size'],
        random_state=config['training']['random_state']
    )
    
    # Initialize data processor
    print("\nInitializing data processor...")
    processor = AdvancedDataProcessor(
        tokenizer_name=config['model']['bert_model_name'],
        max_sentences=config['data_processing']['max_sentences'],
        max_tokens_per_sentence=config['data_processing']['max_tokens_per_sentence'],
        max_tokens_per_article=config['data_processing']['max_tokens_per_article'],
        min_tokens_per_sentence=config['data_processing']['min_tokens_per_sentence']
    )
    
    # Process data
    print("Processing training data...")
    train_sentence_ids, train_sentence_mask, train_article_ids, train_article_mask, train_sentence_texts, train_labels_tensor = \
        processor.prepare_dataset(train_text.tolist(), train_labels.tolist())
    
    print("Processing validation data...")
    val_sentence_ids, val_sentence_mask, val_article_ids, val_article_mask, val_sentence_texts, val_labels_tensor = \
        processor.prepare_dataset(val_text.tolist(), val_labels.tolist())
    
    print("Processing test data...")
    test_sentence_ids, test_sentence_mask, test_article_ids, test_article_mask, test_sentence_texts, test_labels_tensor = \
        processor.prepare_dataset(test_text.tolist(), test_labels.tolist())
    
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
    print("Creating data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config['training']['batch_size'],
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=config['training']['batch_size'],
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=config['training']['batch_size'],
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating advanced model...")
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
    
    # Training loop
    print("\nStarting training...")
    best_val_accuracy = 0.0
    
    for epoch in range(config['training']['epochs']):
        epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"{'='*70}")
        
        # Train with progress bar
        print("\nTraining...")
        train_losses = trainer.train_epoch(train_dataloader, show_progress=True)
        train_time = time.time() - epoch_start_time
        
        print(f"\nTraining Losses:")
        print(f"  Classification: {train_losses['classification']:.4f}")
        print(f"  Separation: {train_losses['separation']:.4f}")
        print(f"  Total: {train_losses['total']:.4f}")
        print(f"  Training Time: {train_time:.2f} seconds")
        
        # Evaluate
        print("\nEvaluating...")
        eval_start_time = time.time()
        val_metrics, val_report, _ = evaluator.evaluate(val_dataloader)
        eval_time = time.time() - eval_start_time
        
        print(f"\nValidation Metrics:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {val_metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  Mean Inconsistency (Real): {val_metrics['mean_inconsistency_real']:.4f}")
        print(f"  Mean Inconsistency (Fake): {val_metrics['mean_inconsistency_fake']:.4f}")
        print(f"  Mean Plausibility (Real): {val_metrics['mean_plausibility_real']:.4f}")
        print(f"  Mean Plausibility (Fake): {val_metrics['mean_plausibility_fake']:.4f}")
        print(f"  Uncertain Ratio: {val_metrics['uncertain_ratio']:.2%}")
        print(f"  Evaluation Time: {eval_time:.2f} seconds")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            torch.save(model.state_dict(), config['paths']['model_save_path'])
            print(f"\n✓ Model saved with validation accuracy: {best_val_accuracy:.4f}")
        
        epoch_total_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Total Time: {epoch_total_time:.2f} seconds ({epoch_total_time/60:.2f} minutes)")
    
    # Load best model and evaluate on test set
    print(f"\n{'='*70}")
    print("Evaluating on test set...")
    print(f"{'='*70}")
    model.load_state_dict(torch.load(config['paths']['model_save_path']))
    
    test_metrics, test_report, test_results = evaluator.evaluate(test_dataloader)
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Mean Inconsistency (Real): {test_metrics['mean_inconsistency_real']:.4f}")
    print(f"  Mean Inconsistency (Fake): {test_metrics['mean_inconsistency_fake']:.4f}")
    print(f"  Std Inconsistency (Real): {test_metrics['std_inconsistency_real']:.4f}")
    print(f"  Std Inconsistency (Fake): {test_metrics['std_inconsistency_fake']:.4f}")
    print(f"  Mean Plausibility (Real): {test_metrics['mean_plausibility_real']:.4f}")
    print(f"  Mean Plausibility (Fake): {test_metrics['mean_plausibility_fake']:.4f}")
    print(f"  Overall Mean Plausibility: {test_metrics['mean_plausibility']:.4f}")
    print(f"  Uncertain Ratio: {test_metrics['uncertain_ratio']:.2%}")
    
    print(f"\nClassification Report:")
    print(test_report)
    
    print(f"\n✓ Training complete! Model saved to: {config['paths']['model_save_path']}")


if __name__ == "__main__":
    main()

