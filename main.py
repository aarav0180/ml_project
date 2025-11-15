"""
Main script for training and evaluating BERT-based fake news detection model.
Supports both basic BERT model and advanced constraint-based model.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from src.data import load_data, prepare_data
from src.models.bert_model import create_model
from src.utils import Tokenizer
from src.training import Trainer, Evaluator


def main(model_type='basic'):
    """
    Main training function.
    
    Args:
        model_type: 'basic' for simple BERT model, 'advanced' for constraint-based model
    """
    # Configuration
    config = {
        'data': {
            'true_data_path': 'data/a1_True.csv',
            'fake_data_path': 'data/a2_Fake.csv',
            'text_column': 'title'
        },
        'model': {
            'bert_model_name': 'bert-base-uncased',
            'freeze_bert': True
        },
        'training': {
            'max_length': 15,
            'batch_size': 32,
            'learning_rate': 1e-5,
            'epochs': 2,
            'test_size': 0.3,
            'val_size': 0.5,
            'random_state': 2018
        },
        'paths': {
            'model_save_path': 'models/c2_new_model_weights.pt'
        }
    }
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and prepare data
    print("\nLoading data...")
    data = load_data(
        true_data_path=config['data']['true_data_path'],
        fake_data_path=config['data']['fake_data_path']
    )
    print(f"Data shape: {data.shape}")
    
    train_text, val_text, test_text, train_labels, val_labels, test_labels = prepare_data(
        data,
        text_column=config['data']['text_column'],
        test_size=config['training']['test_size'],
        val_size=config['training']['val_size'],
        random_state=config['training']['random_state']
    )
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = Tokenizer(model_name=config['model']['bert_model_name'])
    
    # Tokenize data
    print("Tokenizing data...")
    tokens_train = tokenizer.encode(
        train_text,
        max_length=config['training']['max_length']
    )
    tokens_val = tokenizer.encode(
        val_text,
        max_length=config['training']['max_length']
    )
    tokens_test = tokenizer.encode(
        test_text,
        max_length=config['training']['max_length']
    )
    
    # Convert to tensors
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())
    
    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())
    
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())
    
    # Create data loaders
    print("Creating data loaders...")
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=config['training']['batch_size']
    )
    
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_dataloader = DataLoader(
        val_data,
        sampler=SequentialSampler(val_data),
        batch_size=config['training']['batch_size']
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        bert_model_name=config['model']['bert_model_name'],
        device=device,
        freeze_bert=config['model']['freeze_bert']
    )
    
    # Initialize optimizer and loss
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.NLLLoss()
    
    # Initialize trainer and evaluator
    trainer = Trainer(model, optimizer, criterion, device=device)
    evaluator = Evaluator(model, criterion, device=device)
    
    # Training loop
    print("\nStarting training...")
    best_valid_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        train_loss = trainer.train_epoch(train_dataloader)
        val_loss = evaluator.evaluate(val_dataloader)
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), config['paths']['model_save_path'])
            print(f"Model saved with validation loss: {val_loss:.3f}")
        
        print(f"Training Loss: {train_loss:.3f}")
        print(f"Validation Loss: {val_loss:.3f}")
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(config['paths']['model_save_path']))
    
    report, preds = evaluator.evaluate_test(test_seq, test_mask, test_y)
    print("\nClassification Report:")
    print(report)
    
    # Example predictions on unseen text
    print("\nMaking predictions on example texts...")
    unseen_text = [
        "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",
        "WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy",
        "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources",
        "Trump administration issues new rules on U.S. visa waivers"
    ]
    
    tokens_unseen = tokenizer.encode(
        unseen_text,
        max_length=config['training']['max_length']
    )
    
    unseen_seq = torch.tensor(tokens_unseen['input_ids']).to(device)
    unseen_mask = torch.tensor(tokens_unseen['attention_mask']).to(device)
    
    predictions = evaluator.predict(unseen_seq, unseen_mask)
    
    label_map = {0: "True", 1: "Fake"}
    print("\nPredictions:")
    for text, pred in zip(unseen_text, predictions):
        print(f"{label_map[pred]}: {text[:60]}...")


if __name__ == "__main__":
    # Check command line argument for model type
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type not in ['basic', 'advanced']:
            print("Usage: python main.py [basic|advanced]")
            print("  basic: Simple BERT model (default)")
            print("  advanced: Constraint-based model (use train_advanced.py instead)")
            sys.exit(1)
    else:
        model_type = 'basic'
    
    if model_type == 'advanced':
        print("For advanced model, please use: python train_advanced.py")
        sys.exit(0)
    
    main(model_type=model_type)

