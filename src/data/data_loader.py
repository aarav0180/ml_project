"""
Data loading and preprocessing module for fake news detection.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(true_data_path='data/True.csv', fake_data_path='data/Fake.csv'):
    """
    Load and combine true and fake news datasets.
    
    Args:
        true_data_path: Path to true news CSV file
        fake_data_path: Path to fake news CSV file
    
    Returns:
        Combined and shuffled DataFrame with labels
    """
    true_data = pd.read_csv(true_data_path)
    fake_data = pd.read_csv(fake_data_path)
    
    # Add target labels
    true_data['Target'] = ['True'] * len(true_data)
    fake_data['Target'] = ['Fake'] * len(fake_data)
    
    # Map labels to numeric values
    label_map = {"Fake": 1, "True": 0}
    
    true_data['label'] = true_data['Target'].map(label_map)
    fake_data['label'] = fake_data['Target'].map(label_map)
    
    # Combine and shuffle
    data = pd.concat([true_data, fake_data]).sample(frac=1).reset_index(drop=True)
    
    return data


def prepare_data(data, text_column='title', test_size=0.3, val_size=0.5, random_state=2018):
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: DataFrame with text and labels
        text_column: Column name containing text data
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_text, val_text, test_text, train_labels, val_labels, test_labels)
    """
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        data[text_column], data['label'],
        random_state=random_state, 
        test_size=test_size, 
        stratify=data['Target']
    )
    
    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels,
        random_state=random_state, 
        test_size=val_size, 
        stratify=temp_labels
    )
    
    return train_text, val_text, test_text, train_labels, val_labels, test_labels

