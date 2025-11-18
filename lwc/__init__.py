"""
LWC - Latent World Consistency Model for Fake News Detection

A production-ready implementation of BERT + Latent World Consistency (LWC) model
for fake news detection combining:
- BERT for article-level encoding
- Plausibility Model (DistilBERT) for sentence-level plausibility scoring
- Consistency Score C(A) computed from plausibility violations
- Feature Fusion combining BERT embeddings with consistency features
"""

__version__ = "1.0.0"

