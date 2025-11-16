"""
Standalone prediction script for fake news detection.
Interactive script that takes user input and predicts if news is Fake or True.
Supports both basic BERT model and advanced constraint-based model.
Enhanced with multi-sentence consistency analysis.
"""
import os
import sys
import torch
import numpy as np
from itertools import combinations
from src.models.bert_model import create_model
from src.models.advanced_model import AdvancedFakeNewsModel
from src.utils import Tokenizer
from src.data.advanced_data_loader import AdvancedDataProcessor
from src.data.sentence_processor import split_into_sentences
from src.training import Evaluator


class FakeNewsDetector:
    """
    Fake news detector class that loads model once and can make multiple predictions.
    """
    
    def __init__(
        self,
        model_path='models/c2_new_model_weights.pt',
        model_type='basic',
        device=None
    ):
        """
        Initialize the detector by loading the model.
        
        Args:
            model_path: Path to saved model weights
            model_type: 'basic' for simple BERT, 'advanced' for constraint-based
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None
        """
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please train the model first."
            )
        
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        
        print(f"Loading {model_type} model from {model_path}...")
        print(f"Using device: {device}")
        
        # Load appropriate model
        if model_type == 'basic':
            self.model = create_model(device=device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            self.tokenizer = Tokenizer()
            self.evaluator = Evaluator(self.model, None, device=device)
        else:  # advanced
            self.model = AdvancedFakeNewsModel(
                plausibility_model_path='plausability_model_final'
            )
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            self.processor = AdvancedDataProcessor()
            # Get plausibility model for direct access
            self.plausibility_model = self.model.plausibility_model
        
        print("Model loaded successfully!\n")
    
    def analyze_sentence_combinations(self, sentences):
        """
        Analyze plausibility of individual sentences and their combinations.
        
        Args:
            sentences: List of sentence strings
        
        Returns:
            Dictionary with analysis results:
                - individual_scores: List of plausibility scores for each sentence
                - combination_scores: Dict mapping (i, j) pairs to combined plausibility
                - inconsistency_drops: Dict mapping pairs to plausibility drop when combined
                - overall_consistency: Overall consistency score
        """
        if len(sentences) < 2:
            # Single sentence - just return its score
            if sentences:
                score = self.plausibility_model.score_claim(sentences[0])
                return {
                    'individual_scores': [score],
                    'combination_scores': {},
                    'inconsistency_drops': {},
                    'overall_consistency': 1.0 - score,
                    'mean_individual_plausibility': score
                }
            return {
                'individual_scores': [],
                'combination_scores': {},
                'inconsistency_drops': {},
                'overall_consistency': 0.0,
                'mean_individual_plausibility': 0.0
            }
        
        # Score each sentence individually
        individual_scores = self.plausibility_model.score_batch(sentences)
        
        # Score all pairs of sentences when combined
        combination_scores = {}
        inconsistency_drops = {}
        
        # Check all pairs
        for i, j in combinations(range(len(sentences)), 2):
            sent_i = sentences[i]
            sent_j = sentences[j]
            
            # Combine sentences (with a separator)
            combined = f"{sent_i} {sent_j}"
            
            # Score the combination
            combined_score = self.plausibility_model.score_claim(combined)
            combination_scores[(i, j)] = combined_score
            
            # Calculate individual average
            individual_avg = (individual_scores[i] + individual_scores[j]) / 2.0
            
            # If combined score is significantly lower than individual average, it's inconsistent
            drop = individual_avg - combined_score
            inconsistency_drops[(i, j)] = max(0.0, drop)  # Only positive drops
        
        # Check triplets if we have 3+ sentences (for more thorough analysis)
        if len(sentences) >= 3:
            for i, j, k in combinations(range(len(sentences)), 3):
                sent_i = sentences[i]
                sent_j = sentences[j]
                sent_k = sentences[k]
                
                # Combine all three
                combined = f"{sent_i} {sent_j} {sent_k}"
                combined_score = self.plausibility_model.score_claim(combined)
                combination_scores[(i, j, k)] = combined_score
                
                # Calculate individual average
                individual_avg = (individual_scores[i] + individual_scores[j] + individual_scores[k]) / 3.0
                drop = individual_avg - combined_score
                inconsistency_drops[(i, j, k)] = max(0.0, drop)
        
        # Calculate overall consistency score
        # Based on: mean violation of individual sentences + inconsistency from combinations
        violations = [1.0 - score for score in individual_scores]
        mean_violation = np.mean(violations) if violations else 0.0
        
        # Add inconsistency from combinations (weighted by drop magnitude)
        combination_inconsistency = np.mean(list(inconsistency_drops.values())) if inconsistency_drops else 0.0
        
        # Overall consistency = mean violation + combination inconsistency
        overall_consistency = mean_violation + 0.5 * combination_inconsistency  # Weight combination less
        
        return {
            'individual_scores': individual_scores,
            'combination_scores': combination_scores,
            'inconsistency_drops': inconsistency_drops,
            'overall_consistency': overall_consistency,
            'mean_individual_plausibility': np.mean(individual_scores) if individual_scores else 0.0,
            'max_inconsistency_drop': max(inconsistency_drops.values()) if inconsistency_drops else 0.0
        }
    
    def predict(self, text, return_confidence=False, return_inconsistency=False, return_analysis=False):
        """
        Predict whether a news headline/article is fake or true.
        
        Args:
            text: News headline or article text string
            return_confidence: If True, also return confidence score
            return_inconsistency: If True and using advanced model, return inconsistency score
            return_analysis: If True and using advanced model, return detailed sentence analysis
        
        Returns:
            Prediction ('Fake' or 'True') and optionally confidence/inconsistency scores/analysis
        """
        if not text or not text.strip():
            result = "Invalid input" if not return_confidence else ("Invalid input", 0.0)
            if return_inconsistency and self.model_type == 'advanced':
                result = ("Invalid input", 0.0, 0.0)
            if return_analysis and self.model_type == 'advanced':
                result = ("Invalid input", 0.0, 0.0, {})
            return result
        
        if self.model_type == 'basic':
            # Basic BERT model prediction
            tokens = self.tokenizer.encode([text], max_length=15)
            input_ids = torch.tensor(tokens['input_ids']).to(self.device)
            attention_mask = torch.tensor(tokens['attention_mask']).to(self.device)
            
            with torch.no_grad():
                log_probs = self.model(input_ids, attention_mask).cpu().numpy()
                probs = np.exp(log_probs)
            
            prediction_idx = np.argmax(probs, axis=1)[0]
            confidence = probs[0][prediction_idx]
            
            label_map = {0: "True", 1: "Fake"}
            prediction = label_map[prediction_idx]
            
            if return_confidence:
                return prediction, float(confidence)
            return prediction
        
        else:  # advanced model
            # Split text into sentences for analysis
            sentences = split_into_sentences(text)
            sentences = [s.strip() for s in sentences if s.strip()]  # Clean and filter empty
            
            # Analyze sentence combinations if requested
            combination_analysis = None
            if return_analysis:
                combination_analysis = self.analyze_sentence_combinations(sentences)
            
            # Advanced model prediction (using full text)
            sentence_ids, sentence_mask, article_ids, article_mask, sentence_texts, _ = \
                self.processor.process_texts([text])
            
            article_ids = article_ids.to(self.device)
            article_mask = article_mask.to(self.device)
            
            with torch.no_grad():
                logits, inconsistency_score = self.model(
                    sentence_texts=sentence_texts,
                    article_input_ids=article_ids,
                    article_attention_mask=article_mask,
                    return_inconsistency=True
                )
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                inconsistency_score = inconsistency_score.cpu().numpy()[0]
            
            prediction_idx = np.argmax(probs, axis=1)[0]
            confidence = probs[0][prediction_idx]
            
            label_map = {0: "True", 1: "Fake"}
            prediction = label_map[prediction_idx]
            
            # Combine model inconsistency with combination analysis if available
            if combination_analysis:
                # Use the higher of the two inconsistency scores
                combined_inconsistency = max(
                    inconsistency_score,
                    combination_analysis['overall_consistency']
                )
            else:
                combined_inconsistency = inconsistency_score
            
            if return_analysis and combination_analysis:
                return prediction, float(confidence), float(combined_inconsistency), combination_analysis
            elif return_inconsistency:
                return prediction, float(confidence), float(combined_inconsistency)
            elif return_confidence:
                return prediction, float(confidence)
            return prediction
    
    def predict_batch(self, texts):
        """
        Predict multiple news headlines at once.
        
        Args:
            texts: List of news headline strings
        
        Returns:
            List of tuples (prediction, confidence)
        """
        if not texts:
            return []
        
        # Tokenize
        tokens = self.tokenizer.encode(texts, max_length=15)
        input_ids = torch.tensor(tokens['input_ids']).to(self.device)
        attention_mask = torch.tensor(tokens['attention_mask']).to(self.device)
        
        # Get predictions with probabilities
        with torch.no_grad():
            log_probs = self.model(input_ids, attention_mask).cpu().numpy()
            probs = np.exp(log_probs)
        
        # Get predictions
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        label_map = {0: "True", 1: "Fake"}
        results = [
            (label_map[pred], float(conf))
            for pred, conf in zip(predictions, confidences)
        ]
        
        return results


def interactive_mode(model_type='basic'):
    """
    Run interactive mode where user can input news headlines.
    
    Args:
        model_type: 'basic' or 'advanced'
    """
    try:
        # Determine model path
        if model_type == 'basic':
            model_path = 'models/c2_new_model_weights.pt'
        else:
            model_path = 'models/advanced_model_weights.pt'
        
        # Initialize detector
        detector = FakeNewsDetector(model_path=model_path, model_type=model_type)
        
        print("=" * 70)
        print(f"Fake News Detector - Interactive Mode ({model_type.upper()} model)")
        print("=" * 70)
        if model_type == 'basic':
            print("Enter news headlines to check if they are Fake or True.")
        else:
            print("Enter news articles/headlines to check if they are Fake or True.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            # Get user input
            prompt = "Enter news " + ("headline" if model_type == 'basic' else "article") + ": "
            text = input(prompt).strip()
            
            # Check for exit commands
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                print("Please enter a valid news text.\n")
                continue
            
            # Make prediction
            try:
                if model_type == 'advanced':
                    # Check if multiple sentences
                    sentences = split_into_sentences(text)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    
                    if len(sentences) > 1:
                        # Multi-sentence analysis
                        prediction, confidence, inconsistency, analysis = detector.predict(
                            text, return_confidence=True, return_inconsistency=True, return_analysis=True
                        )
                        
                        if prediction == "Fake":
                            result_str = f"❌ FAKE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
                        else:
                            result_str = f"✅ TRUE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
                        
                        print(f"\nResult: {result_str}")
                        print(f"\n📊 Sentence Analysis ({len(sentences)} sentences):")
                        print("-" * 70)
                        
                        # Show individual sentence scores
                        print("\nIndividual Sentence Plausibility:")
                        for i, (sent, score) in enumerate(zip(sentences, analysis['individual_scores']), 1):
                            status = "✅" if score >= 0.5 else "⚠️"
                            print(f"  {i}. [{status} {score:.3f}] {sent[:80]}{'...' if len(sent) > 80 else ''}")
                        
                        # Show combination inconsistencies
                        if analysis['inconsistency_drops']:
                            print(f"\n⚠️  Combination Inconsistencies Found:")
                            for key, drop in sorted(analysis['inconsistency_drops'].items(), key=lambda x: x[1], reverse=True):
                                if drop > 0.1:  # Only show significant drops
                                    if len(key) == 2:
                                        i, j = key
                                        print(f"  Sentences {i+1} & {j+1}: Drop of {drop:.3f} when combined")
                                    elif len(key) == 3:
                                        i, j, k = key
                                        print(f"  Sentences {i+1}, {j+1} & {k+1}: Drop of {drop:.3f} when combined")
                        
                        print(f"\nMean Individual Plausibility: {analysis['mean_individual_plausibility']:.3f}")
                        if analysis['max_inconsistency_drop'] > 0:
                            print(f"Max Inconsistency Drop: {analysis['max_inconsistency_drop']:.3f}")
                        
                        print(f"\nText: {text[:100]}{'...' if len(text) > 100 else ''}\n")
                    else:
                        # Single sentence - standard output
                        prediction, confidence, inconsistency = detector.predict(
                            text, return_confidence=True, return_inconsistency=True
                        )
                        
                        if prediction == "Fake":
                            result_str = f"❌ FAKE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
                        else:
                            result_str = f"✅ TRUE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
                        
                        print(f"\nResult: {result_str}")
                        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}\n")
                else:
                    prediction, confidence = detector.predict(text, return_confidence=True)
                    
                    if prediction == "Fake":
                        result_str = f"❌ FAKE (Confidence: {confidence:.2%})"
                    else:
                        result_str = f"✅ TRUE (Confidence: {confidence:.2%})"
                    
                    print(f"\nResult: {result_str}")
                    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}\n")
                
                print("-" * 70 + "\n")
                
            except Exception as e:
                print(f"Error making prediction: {str(e)}\n")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        if model_type == 'basic':
            print("\nPlease train the model first by running: python main.py")
        else:
            print("\nPlease train the model first by running: python train_advanced.py")
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")


def single_prediction(text, model_type='basic'):
    """
    Make a single prediction from command line argument.
    
    Args:
        text: News headline/article to predict
        model_type: 'basic' or 'advanced'
    """
    try:
        model_path = 'models/c2_new_model_weights.pt' if model_type == 'basic' else 'models/advanced_model_weights.pt'
        detector = FakeNewsDetector(model_path=model_path, model_type=model_type)
        
        if model_type == 'advanced':
            # Check if multiple sentences
            sentences = split_into_sentences(text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) > 1:
                # Multi-sentence analysis
                prediction, confidence, inconsistency, analysis = detector.predict(
                    text, return_confidence=True, return_inconsistency=True, return_analysis=True
                )
                
                if prediction == "Fake":
                    result_str = f"❌ FAKE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
                else:
                    result_str = f"✅ TRUE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
                
                print(f"\nResult: {result_str}")
                print(f"\n📊 Sentence Analysis ({len(sentences)} sentences):")
                print("-" * 70)
                
                # Show individual sentence scores
                print("\nIndividual Sentence Plausibility:")
                for i, (sent, score) in enumerate(zip(sentences, analysis['individual_scores']), 1):
                    status = "✅" if score >= 0.5 else "⚠️"
                    print(f"  {i}. [{status} {score:.3f}] {sent[:80]}{'...' if len(sent) > 80 else ''}")
                
                # Show combination inconsistencies
                if analysis['inconsistency_drops']:
                    print(f"\n⚠️  Combination Inconsistencies Found:")
                    for key, drop in sorted(analysis['inconsistency_drops'].items(), key=lambda x: x[1], reverse=True):
                        if drop > 0.1:  # Only show significant drops
                            if len(key) == 2:
                                i, j = key
                                print(f"  Sentences {i+1} & {j+1}: Drop of {drop:.3f} when combined")
                            elif len(key) == 3:
                                i, j, k = key
                                print(f"  Sentences {i+1}, {j+1} & {k+1}: Drop of {drop:.3f} when combined")
                
                print(f"\nMean Individual Plausibility: {analysis['mean_individual_plausibility']:.3f}")
                if analysis['max_inconsistency_drop'] > 0:
                    print(f"Max Inconsistency Drop: {analysis['max_inconsistency_drop']:.3f}")
            else:
                # Single sentence
                prediction, confidence, inconsistency = detector.predict(
                    text, return_confidence=True, return_inconsistency=True
                )
                
                if prediction == "Fake":
                    result_str = f"❌ FAKE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
                else:
                    result_str = f"✅ TRUE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
                
                print(f"\nResult: {result_str}")
        else:
            prediction, confidence = detector.predict(text, return_confidence=True)
            
            if prediction == "Fake":
                result_str = f"❌ FAKE (Confidence: {confidence:.2%})"
            else:
                result_str = f"✅ TRUE (Confidence: {confidence:.2%})"
            
            print(f"\nResult: {result_str}")
        
        print(f"Text: {text}\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        if model_type == 'basic':
            print("\nPlease train the model first by running: python main.py")
        else:
            print("\nPlease train the model first by running: python train_advanced.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    # Parse command line arguments
    model_type = 'basic'
    text_args_start = 1
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--basic', '--advanced', '-b', '-a']:
            model_type = 'advanced' if 'advanced' in sys.argv[1] else 'basic'
            text_args_start = 2
    
    # Check if text provided as command line argument
    if len(sys.argv) > text_args_start:
        # Join all arguments as the news text
        news_text = " ".join(sys.argv[text_args_start:])
        single_prediction(news_text, model_type=model_type)
    else:
        # Run interactive mode
        print("Usage: python predict.py [--basic|--advanced] [text]")
        print("  --basic: Use basic BERT model (default)")
        print("  --advanced: Use advanced constraint-based model")
        print("  If no text provided, runs in interactive mode\n")
        interactive_mode(model_type=model_type)

