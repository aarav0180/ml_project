"""
Standalone prediction script for fake news detection.
Interactive script that takes user input and predicts if news is Fake or True.
Supports both basic BERT model and advanced constraint-based model.
"""
import os
import sys
import torch
import numpy as np
from src.models.bert_model import create_model
from src.models.advanced_model import AdvancedFakeNewsModel
from src.utils import Tokenizer
from src.data.advanced_data_loader import AdvancedDataProcessor
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
            self.model = AdvancedFakeNewsModel()
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            self.processor = AdvancedDataProcessor()
        
        print("Model loaded successfully!\n")
    
    def predict(self, text, return_confidence=False, return_inconsistency=False):
        """
        Predict whether a news headline/article is fake or true.
        
        Args:
            text: News headline or article text string
            return_confidence: If True, also return confidence score
            return_inconsistency: If True and using advanced model, return inconsistency score
        
        Returns:
            Prediction ('Fake' or 'True') and optionally confidence/inconsistency scores
        """
        if not text or not text.strip():
            result = "Invalid input" if not return_confidence else ("Invalid input", 0.0)
            if return_inconsistency and self.model_type == 'advanced':
                result = ("Invalid input", 0.0, 0.0)
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
            # Advanced model prediction
            sentence_ids, sentence_mask, article_ids, article_mask, _ = \
                self.processor.process_texts([text])
            
            sentence_ids = sentence_ids.to(self.device)
            sentence_mask = sentence_mask.to(self.device)
            article_ids = article_ids.to(self.device)
            article_mask = article_mask.to(self.device)
            
            with torch.no_grad():
                logits, inconsistency_score = self.model(
                    sentence_ids, sentence_mask,
                    article_ids, article_mask,
                    return_inconsistency=True
                )
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                inconsistency_score = inconsistency_score.cpu().numpy()[0]
            
            prediction_idx = np.argmax(probs, axis=1)[0]
            confidence = probs[0][prediction_idx]
            
            label_map = {0: "True", 1: "Fake"}
            prediction = label_map[prediction_idx]
            
            if return_inconsistency:
                return prediction, float(confidence), float(inconsistency_score)
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
                    prediction, confidence, inconsistency = detector.predict(
                        text, return_confidence=True, return_inconsistency=True
                    )
                    
                    if prediction == "Fake":
                        result_str = f"❌ FAKE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
                    else:
                        result_str = f"✅ TRUE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
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
            prediction, confidence, inconsistency = detector.predict(
                text, return_confidence=True, return_inconsistency=True
            )
            
            if prediction == "Fake":
                result_str = f"❌ FAKE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
            else:
                result_str = f"✅ TRUE (Confidence: {confidence:.2%}, Inconsistency: {inconsistency:.4f})"
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

