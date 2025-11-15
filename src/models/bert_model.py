"""
BERT-based model architecture for fake news detection.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class BERT_Arch(nn.Module):
    """
    BERT-based neural network architecture for binary classification.
    Uses BERT as a feature extractor with a custom classification head.
    """
    
    def __init__(self, bert, dropout_rate=0.1):
        """
        Initialize BERT architecture.
        
        Args:
            bert: Pre-trained BERT model
            dropout_rate: Dropout rate for regularization
        """
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, sent_id, mask):
        """
        Forward pass through the model.
        
        Args:
            sent_id: Tokenized input sequences
            mask: Attention mask
        
        Returns:
            Log probabilities for each class
        """
        # Get BERT's pooled output (CLS token representation)
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        
        # Classification head
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x


def create_model(bert_model_name='bert-base-uncased', device='cpu', freeze_bert=True):
    """
    Create and initialize BERT model for fake news detection.
    
    Args:
        bert_model_name: Name of the BERT model to use
        device: Device to load model on ('cpu' or 'cuda')
        freeze_bert: Whether to freeze BERT parameters
    
    Returns:
        Initialized BERT_Arch model
    """
    # Load pre-trained BERT model
    bert = AutoModel.from_pretrained(bert_model_name)
    
    # Freeze BERT parameters if specified
    if freeze_bert:
        for param in bert.parameters():
            param.requires_grad = False
    
    # Create custom architecture
    model = BERT_Arch(bert)
    model.to(device)
    
    return model

