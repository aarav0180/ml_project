"""Data loading utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def load_data(true_data_path: str = 'data/True.csv', fake_data_path: str = 'data/Fake.csv') -> pd.DataFrame:
    """
    Load and combine true and fake news datasets.
    
    Args:
        true_data_path: Path to true news CSV file
        fake_data_path: Path to fake news CSV file
        
    Returns:
        Combined and shuffled DataFrame
    """
    logger.info(f"Loading data from {true_data_path} and {fake_data_path}")
    true_data = pd.read_csv(true_data_path)
    fake_data = pd.read_csv(fake_data_path)

    true_data['Target'] = ['True'] * len(true_data)
    fake_data['Target'] = ['Fake'] * len(fake_data)

    label_map = {"Fake": 1, "True": 0}
    true_data['label'] = true_data['Target'].map(label_map)
    fake_data['label'] = fake_data['Target'].map(label_map)

    data = pd.concat([true_data, fake_data]).sample(frac=1).reset_index(drop=True)
    logger.info(f"Loaded {len(data)} samples")
    return data


def prepare_data(
    data: pd.DataFrame,
    text_column: str = 'text',
    test_size: float = 0.3,
    val_size: float = 0.5,
    random_state: int = 2018
):
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: DataFrame with data
        text_column: Name of the text column
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation
        random_state: Random seed
        
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

    logger.info(f"Data split - Train: {len(train_text)}, Val: {len(val_text)}, Test: {len(test_text)}")
    return train_text, val_text, test_text, train_labels, val_labels, test_labels

