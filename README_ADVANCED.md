# Advanced Fake News Detection Model

This document describes the advanced constraint-based fake news detection model that extends the basic BERT implementation.

## Overview

The advanced model uses a novel approach combining:
1. **Sentence-level BERT encoding** with constraint generation
2. **World vector optimization** for inconsistency scoring
3. **Full article BERT encoding**
4. **Combined classifier** using both article features and inconsistency scores

## Architecture

### 1. Sentence Processing
- Articles are split into sentences using NLTK
- Sentences with < 5 tokens are filtered out
- Maximum 20 sentences per article (truncated/padded)
- Each sentence tokenized to max 64 tokens for BERT

### 2. Constraint Generator
- Input: BERT CLS embeddings (768-d) for each sentence
- MLP: 768 → 256 → (d+1) where d=32
- Output: Constraints (a_i, b_i) for each sentence
  - a_i: Normalized d-dimensional vector
  - b_i: Scalar value

### 3. World Vector Optimization
- For each article, optimizes a world vector z ∈ R^d
- Minimizes energy: E(A,z) = Σ(a_i^T z - b_i)^2
- Uses Adam optimizer with lr=0.05 for 30 steps
- Final inconsistency score: C(A) = E(A, z*)

### 4. Article Encoder
- Full article encoded with BERT (max 256 tokens)
- Extracts pooled CLS embedding → x_A ∈ R^768

### 5. Combined Classifier
- Input: concat([x_A (768), C(A) (1)]) = 769 dimensions
- MLP: 769 → 256 → 2
- Output: Binary classification (True/Fake)

## Training Losses

The model uses three loss components:

### 1. Classification Loss
- Standard cross-entropy: L_cls = -y log(ŷ)

### 2. Consistency Separation Loss
- Encourages real articles → low C(A)
- Encourages fake articles → high C(A)
- L_sep = λ₁·E[C(A)]_real - λ₂·E[C(A)]_fake
- Default: λ₁ = 0.1, λ₂ = 0.1

### 3. Constraint Smoothness Loss
- Encourages stable constraint generation
- L_smooth = α·Σ||a_i||²
- Default: α = 0.001

### Total Loss
L = L_cls + L_sep + L_smooth

## Usage

### Training

```bash
python train_advanced.py
```

This will:
1. Load data from `data/a1_True.csv` and `data/a2_Fake.csv`
2. Process articles into sentences
3. Train the advanced model with all three losses
4. Save model to `models/advanced_model_weights.pt`

### Prediction

#### Interactive Mode
```bash
python predict.py --advanced
```

#### Command Line
```bash
python predict.py --advanced "Your news article text here"
```

#### Programmatic Usage
```python
from predict import FakeNewsDetector

detector = FakeNewsDetector(
    model_path='models/advanced_model_weights.pt',
    model_type='advanced'
)

prediction, confidence, inconsistency = detector.predict(
    "Your news article",
    return_confidence=True,
    return_inconsistency=True
)

print(f"{prediction} (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})")
```

## Configuration

Edit `train_advanced.py` to modify:

- **Model parameters:**
  - `latent_dim`: World vector dimension (default: 32)
  - `hidden_dim`: Constraint generator hidden dim (default: 256)
  - `world_opt_lr`: World optimization learning rate (default: 0.05)
  - `world_opt_steps`: Optimization steps (default: 30)

- **Data processing:**
  - `max_sentences`: Max sentences per article (default: 20)
  - `max_tokens_per_sentence`: Max tokens per sentence (default: 64)
  - `max_tokens_per_article`: Max tokens for full article (default: 256)

- **Training:**
  - `batch_size`: Batch size (default: 4, smaller due to memory)
  - `learning_rate`: Learning rate (default: 2e-5)
  - `epochs`: Number of epochs (default: 3)
  - `lambda1`, `lambda2`: Consistency separation weights (default: 0.1)
  - `alpha`: Smoothness weight (default: 0.001)

## Evaluation Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: F1 score for binary classification
- **ROC-AUC**: Area under ROC curve
- **Mean Inconsistency (Real)**: Average C(A) for real news
- **Mean Inconsistency (Fake)**: Average C(A) for fake news
- **Standard Deviations**: Spread of inconsistency scores

## Key Differences from Basic Model

1. **Uses full article text** instead of just headlines
2. **Sentence-level analysis** with constraint generation
3. **Inconsistency scoring** based on world vector optimization
4. **Multiple loss components** for better training
5. **More memory intensive** (smaller batch size required)

## Requirements

Additional dependencies:
- `nltk>=3.8.0` for sentence tokenization
- `spacy>=3.7.0` (optional, for alternative tokenization)

## Notes

- The advanced model requires more computational resources
- Training is slower due to sentence-level processing and world optimization
- Works best with full article text (use 'text' column, not 'title')
- Inconsistency scores provide interpretability: higher scores indicate more inconsistent (likely fake) content

