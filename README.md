# LWC - Latent World Consistency Model for Fake News Detection

A production-ready implementation of BERT + Latent World Consistency (LWC) model for fake news detection.

## Features

- **BERT** for article-level encoding
- **Plausibility Model** (DistilBERT) for sentence-level plausibility scoring
- **Consistency Score C(A)** computed from plausibility violations
- **Feature Fusion** combining BERT embeddings with consistency features
- CUDA acceleration for faster training
- Optimized data loading with pin_memory
- Comprehensive evaluation metrics

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml_test
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (will be done automatically, but you can also do it manually):
```python
import nltk
nltk.download('punkt')
```

## Project Structure

```
ml_test/
├── lwc/                          # Main package
│   ├── models/                   # Model definitions
│   │   ├── plausibility.py      # PlausibilityWorldModel
│   │   └── advanced_model.py    # AdvancedFakeNewsModel
│   ├── data/                     # Data processing
│   │   ├── processor.py         # AdvancedDataProcessor, sentence processing
│   │   ├── dataset.py           # AdvancedDataset, custom_collate_fn
│   │   └── loader.py            # load_data, prepare_data functions
│   ├── training/                 # Training components
│   │   ├── trainer.py           # AdvancedTrainer
│   │   └── evaluator.py         # AdvancedEvaluator
│   └── utils/                    # Utilities
│       ├── device.py            # Device detection and setup
│       ├── nltk_setup.py        # NLTK data download
│       └── config.py            # Configuration loading
├── configs/
│   └── default_config.yaml      # Configuration file
├── scripts/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script (includes WELFake)
├── requirements.txt
└── README.md
```

## Data Preparation

Place your training data files in the `data/` directory:
- `data/True.csv` - True news articles
- `data/Fake.csv` - Fake news articles
- `data/welFake.csv` - WELFake dataset for evaluation (optional)

Ensure the CSV files have a `text` column (or `title` as fallback) containing the article text.

## Configuration

Edit `configs/default_config.yaml` to customize:
- Data paths
- Model parameters
- Training hyperparameters
- Model save paths

## Usage

### Training

Train the model:
```bash
python scripts/train.py
```

The script will:
1. Load and preprocess the data
2. Train the model for the specified number of epochs
3. Save the best model based on validation accuracy
4. Evaluate on the test set

### Evaluation

Evaluate on WELFake dataset:
```bash
python scripts/evaluate.py
```

The script will:
1. Load the trained model weights
2. Load and preprocess WELFake dataset
3. Run evaluation and print metrics

## Model Architecture

The model combines:
1. **Sentence-level plausibility scoring** using a pre-trained DistilBERT model
2. **Consistency score C(A)** computed from plausibility violations
3. **Full article BERT encoding** for semantic understanding
4. **Combined classifier** using both BERT embeddings and consistency features

## Output

The model outputs:
- Classification predictions (real/fake)
- Consistency scores
- Plausibility scores
- Comprehensive evaluation metrics (accuracy, F1-score, ROC-AUC, etc.)

## License

[Add your license here]

## Citation

[Add citation if applicable]

