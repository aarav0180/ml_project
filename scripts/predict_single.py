#!/usr/bin/env python
"""Single article prediction script for LWC model.

This script allows you to evaluate a single news article (with title and/or text)
and get a prediction of whether it's real or fake.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from lwc.utils.config import load_config
from lwc.utils.device import get_device
from lwc.utils.nltk_setup import setup_nltk_data
from lwc.models.advanced_model import AdvancedFakeNewsModel
from lwc.data.processor import AdvancedDataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def combine_title_and_text(title: str = "", text: str = "") -> str:
    """
    Combine title and text into a single article string.
    
    Args:
        title: Article title (optional)
        text: Article text (optional)
        
    Returns:
        Combined article text
    """
    if title and text:
        # Combine title and text with a separator
        return f"{title}. {text}"
    elif title:
        return title
    elif text:
        return text
    else:
        raise ValueError("At least one of 'title' or 'text' must be provided")


def predict_single_article(
    title: str = "",
    text: str = "",
    model_path: str = None,
    config_path: str = 'configs/default_config.yaml'
) -> dict:
    """
    Predict if a single article is real or fake.
    
    Args:
        title: Article title (optional)
        text: Article text (optional, but recommended)
        model_path: Path to model weights (if None, uses config)
        config_path: Path to config file
        
    Returns:
        Dictionary with prediction, confidence, and details
    """
    # Setup
    setup_nltk_data()
    config = load_config(config_path)
    device = get_device()
    
    # Combine title and text
    article_text = combine_title_and_text(title, text)
    
    if not article_text or not article_text.strip():
        raise ValueError("Article text cannot be empty")
    
    logger.info(f"Processing article (length: {len(article_text)} characters)")
    
    # Initialize processor
    processor = AdvancedDataProcessor(
        tokenizer_name=config['model']['bert_model_name'],
        max_sentences=config['data_processing']['max_sentences'],
        max_tokens_per_sentence=config['data_processing']['max_tokens_per_sentence'],
        max_tokens_per_article=config['data_processing']['max_tokens_per_article'],
        min_tokens_per_sentence=config['data_processing']['min_tokens_per_sentence']
    )
    
    # Process the article
    logger.info("Tokenizing article...")
    sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, _ = \
        processor.prepare_dataset([article_text], [0])  # Dummy label, not used
    
    # Tensors from prepare_dataset already have batch dimension
    # No need to unsqueeze - they're already in the correct shape:
    # article_ids: (1, seq_length), article_mask: (1, seq_length)
    # sentence_ids: (1, num_sentences, seq_length), sentence_mask: (1, num_sentences, seq_length)
    
    # Initialize model
    logger.info("Loading model...")
    model = AdvancedFakeNewsModel(
        bert_model_name=config['model']['bert_model_name'],
        freeze_bert=config['model']['freeze_bert'],
        plausibility_model_path=config['model']['plausibility_model_path']
    )
    
    # Load weights
    if model_path is None:
        model_path = config['paths']['model_save_path']
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    logger.info(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Move tensors to device
    article_ids = article_ids.to(device)
    article_mask = article_mask.to(device)
    
    # Predict
    logger.info("Running prediction...")
    with torch.no_grad():
        logits, inconsistency_scores = model(
            sentence_texts=[sentence_texts[0]],
            article_input_ids=article_ids,
            article_attention_mask=article_mask,
            return_inconsistency=True
        )
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)
        probs = probabilities.cpu().numpy()[0]
        
        # Get prediction
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        inconsistency = float(inconsistency_scores.cpu().numpy()[0])
        
        # Get plausibility scores for sentences
        plausibility_scores = model.plausibility_model.score_batch(sentence_texts[0])
        mean_plausibility = float(np.mean(plausibility_scores)) if plausibility_scores else 0.0
    
    # Format results
    result = {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'prediction_label': prediction,
        'confidence': confidence,
        'probabilities': {
            'real': float(probs[0]),
            'fake': float(probs[1])
        },
        'inconsistency_score': inconsistency,
        'mean_plausibility': mean_plausibility,
        'article_length': len(article_text),
        'num_sentences': len([s for s in sentence_texts[0] if s.strip()])
    }
    
    return result


def print_prediction(result: dict, title: str = "", text: str = ""):
    """Print prediction results in a readable format."""
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    if title:
        print(f"\nTitle: {title[:100]}{'...' if len(title) > 100 else ''}")
    if text:
        print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
    
    print(f"\n{'='*70}")
    print(f"PREDICTION: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"{'='*70}")
    
    print(f"\nDetailed Probabilities:")
    print(f"  Real News: {result['probabilities']['real']:.2%}")
    print(f"  Fake News: {result['probabilities']['fake']:.2%}")
    
    print(f"\nModel Analysis:")
    print(f"  Inconsistency Score: {result['inconsistency_score']:.4f}")
    print(f"  Mean Plausibility: {result['mean_plausibility']:.4f}")
    print(f"  Article Length: {result['article_length']} characters")
    print(f"  Number of Sentences: {result['num_sentences']}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if result['prediction'] == 'FAKE':
        if result['confidence'] > 0.8:
            print("  ⚠️  HIGH CONFIDENCE: This article is likely FAKE NEWS")
        elif result['confidence'] > 0.6:
            print("  ⚠️  MODERATE CONFIDENCE: This article may be FAKE NEWS")
        else:
            print("  ⚠️  LOW CONFIDENCE: This article might be FAKE NEWS")
    else:
        if result['confidence'] > 0.8:
            print("  ✓ HIGH CONFIDENCE: This article appears to be REAL NEWS")
        elif result['confidence'] > 0.6:
            print("  ✓ MODERATE CONFIDENCE: This article may be REAL NEWS")
        else:
            print("  ✓ LOW CONFIDENCE: This article might be REAL NEWS")
    
    if result['inconsistency_score'] > 0.3:
        print("  ⚠️  High inconsistency detected - suggests potential fake news")
    if result['mean_plausibility'] < 0.5:
        print("  ⚠️  Low plausibility score - content may be questionable")
    
    print("="*70 + "\n")


def main():
    """Main function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Predict if a single news article is real or fake',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using title and text:
  python scripts/predict_single.py --title "Breaking News" --text "Article content here..."
  
  # Using only text:
  python scripts/predict_single.py --text "Article content here..."
  
  # Using only title:
  python scripts/predict_single.py --title "Breaking News"
  
  # Interactive mode:
  python scripts/predict_single.py --interactive
        """
    )
    
    parser.add_argument('--title', type=str, default='', help='Article title')
    parser.add_argument('--text', type=str, default='', help='Article text')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode (prompt for input)')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights (overrides config)')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*70)
        print("INTERACTIVE MODE - Single Article Prediction")
        print("="*70)
        print("\nEnter article information (press Enter to skip):")
        
        title = input("\nTitle: ").strip()
        text = input("Text: ").strip()
        
        if not title and not text:
            print("\nError: At least one of title or text must be provided.")
            sys.exit(1)
    else:
        title = args.title
        text = args.text
        
        if not title and not text:
            print("\nError: At least one of --title or --text must be provided.")
            print("Use --interactive for interactive mode.")
            parser.print_help()
            sys.exit(1)
    
    try:
        # Predict
        result = predict_single_article(
            title=title,
            text=text,
            model_path=args.model,
            config_path=args.config
        )
        
        # Print results
        print_prediction(result, title, text)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

