# Fake News Detection using BERT

A BERT-based binary classification model for detecting fake news articles.

**Two Models Available:**
1. **Basic BERT Model**: Simple and fast, uses headlines only
2. **Advanced Constraint-Based Model**: Uses sentence-level analysis with inconsistency scoring (see [README_ADVANCED.md](README_ADVANCED.md))

## Project Structure

```
ml_test/
├── src/
│   ├── data/
│   │   └── data_loader.py          # Data loading and preprocessing
│   ├── models/
│   │   └── bert_model.py           # BERT model architecture
│   ├── training/
│   │   ├── trainer.py              # Training logic
│   │   └── evaluator.py            # Evaluation logic
│   └── utils/
│       └── tokenizer.py            # Tokenization utilities
├── data/
│   ├── a1_True.csv                 # True news dataset
│   └── a2_Fake.csv                 # Fake news dataset
├── models/                         # Saved model weights
├── main.py                         # Main training script
├── predict.py                      # Interactive prediction script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Place your CSV files (`a1_True.csv` and `a2_Fake.csv`) in the `data/` directory
   - The CSV files should have at least a `title` column containing the news headlines

## Usage

### Training the Models

#### Basic BERT Model (Headlines)

Run the main script to train the basic model:

```bash
python main.py
```

#### Advanced Constraint-Based Model (Full Articles)

Run the advanced training script:

```bash
python train_advanced.py
```

**Note:** The advanced model uses full article text and requires more memory. See [README_ADVANCED.md](README_ADVANCED.md) for details.

The script will:
1. Load and preprocess the data
2. Split data into train/validation/test sets
3. Initialize BERT model
4. Train the model for specified epochs
5. Evaluate on test set
6. Save the best model weights

### Model Configuration

You can modify the configuration in `main.py`:

- **Model**: Change `bert_model_name` to use different BERT variants
- **Training**: Adjust `batch_size`, `learning_rate`, `epochs`, etc.
- **Data**: Modify `max_length` for sequence length

### Using the Models for Predictions

#### Interactive Mode (Recommended)

Run the prediction script in interactive mode:

**Basic Model (headlines):**
```bash
python predict.py --basic
```

**Advanced Model (full articles):**
```bash
python predict.py --advanced
```

This will start an interactive session where you can:
- Enter news headlines one by one
- Get instant predictions (Fake or True) with confidence scores
- Type 'quit' or 'exit' to stop

Example session:
```
Enter news headline: Scientists discover new planet
Result: ✅ TRUE (Confidence: 87.23%)
Headline: Scientists discover new planet
```

#### Command Line Mode

You can also pass text directly as a command line argument:

**Basic Model:**
```bash
python predict.py --basic "Your news headline here"
```

**Advanced Model:**
```bash
python predict.py --advanced "Your full news article text here"
```

#### Programmatic Usage

You can also use the `FakeNewsDetector` class in your own Python scripts:

```python
from predict import FakeNewsDetector

# Initialize detector (loads model once)
detector = FakeNewsDetector()

# Make single prediction
prediction, confidence = detector.predict("Your news headline", return_confidence=True)
print(f"{prediction} ({confidence:.2%})")

# Make batch predictions
texts = ["Headline 1", "Headline 2", "Headline 3"]
results = detector.predict_batch(texts)
for text, (pred, conf) in zip(texts, results):
    print(f"{pred} ({conf:.2%}): {text}")
```

## Model Architecture

- **Base Model**: BERT-base-uncased (frozen parameters)
- **Classification Head**: 
  - Linear(768 → 512) + ReLU + Dropout(0.1)
  - Linear(512 → 2) + LogSoftmax

## Data Format

The CSV files should have the following structure:
- `title`: News headline text
- Other columns are optional but preserved in the dataset

## Output

- Trained model weights saved to `models/c2_new_model_weights.pt`
- Training and validation loss printed during training
- Classification report on test set
- Example predictions on sample texts

## Model Comparison

| Feature | Basic Model | Advanced Model |
|---------|------------|----------------|
| Input | Headlines only | Full article text |
| Processing | Single BERT encoding | Sentence-level + article-level |
| Additional Features | None | Inconsistency scoring (C(A)) |
| Training Loss | Classification only | Classification + Separation + Smoothness |
| Batch Size | 32 | 4 (memory intensive) |
| Speed | Fast | Slower (more computation) |
| Use Case | Quick headline checking | Deep article analysis |

## Notes

- **Basic Model**: Uses only the `title` column for classification
- **Advanced Model**: Uses the `text` column (full article)
- BERT parameters are frozen (not fine-tuned) for faster training
- Models automatically use GPU if available, otherwise fall back to CPU
- Basic model default sequence length: 15 tokens
- Advanced model: 64 tokens per sentence, 256 tokens per article

