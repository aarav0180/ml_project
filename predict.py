"""
Standalone prediction script for fake news detection.
Interactive script that takes user input and predicts if news is Fake or True.
"""
import os
import torch
import numpy as np
from src.models.bert_model import create_model
from src.utils import Tokenizer
from src.training import Evaluator


class FakeNewsDetector:
    """
    Fake news detector class that loads model once and can make multiple predictions.
    """
    
    def __init__(self, model_path='models/c2_new_model_weights.pt', device=None):
        """
        Initialize the detector by loading the model.
        
        Args:
            model_path: Path to saved model weights
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None
        """
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please train the model first using main.py"
            )
        
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model_path = model_path
        
        print(f"Loading model from {model_path}...")
        print(f"Using device: {device}")
        
        # Load model once
        self.model = create_model(device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()  # Set to evaluation mode
        
        # Initialize tokenizer and evaluator
        self.tokenizer = Tokenizer()
        self.evaluator = Evaluator(self.model, None, device=device)
        
        print("Model loaded successfully!\n")
    
    def predict(self, text, return_confidence=False):
        """
        Predict whether a news headline is fake or true.
        
        Args:
            text: News headline string
            return_confidence: If True, also return confidence score
        
        Returns:
            Prediction ('Fake' or 'True') and optionally confidence score
        """
        if not text or not text.strip():
            return "Invalid input" if not return_confidence else ("Invalid input", 0.0)
        
        # Tokenize
        tokens = self.tokenizer.encode([text], max_length=15)
        input_ids = torch.tensor(tokens['input_ids']).to(self.device)
        attention_mask = torch.tensor(tokens['attention_mask']).to(self.device)
        
        # Get prediction with probabilities
        with torch.no_grad():
            log_probs = self.model(input_ids, attention_mask).cpu().numpy()
            probs = np.exp(log_probs)  # Convert log probabilities to probabilities
        
        # Get prediction
        prediction_idx = np.argmax(probs, axis=1)[0]
        confidence = probs[0][prediction_idx]
        
        label_map = {0: "True", 1: "Fake"}
        prediction = label_map[prediction_idx]
        
        if return_confidence:
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


def interactive_mode():
    """
    Run interactive mode where user can input news headlines.
    """
    try:
        # Initialize detector
        detector = FakeNewsDetector()
        
        print("=" * 70)
        print("Fake News Detector - Interactive Mode")
        print("=" * 70)
        print("Enter news headlines to check if they are Fake or True.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            # Get user input
            text = input("Enter news headline: ").strip()
            
            # Check for exit commands
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                print("Please enter a valid news headline.\n")
                continue
            
            # Make prediction
            try:
                prediction, confidence = detector.predict(text, return_confidence=True)
                
                # Format output with color coding
                if prediction == "Fake":
                    result_str = f"❌ FAKE (Confidence: {confidence:.2%})"
                else:
                    result_str = f"✅ TRUE (Confidence: {confidence:.2%})"
                
                print(f"\nResult: {result_str}")
                print(f"Headline: {text}\n")
                print("-" * 70 + "\n")
                
            except Exception as e:
                print(f"Error making prediction: {str(e)}\n")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease train the model first by running: python main.py")
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")


def single_prediction(text):
    """
    Make a single prediction from command line argument.
    
    Args:
        text: News headline to predict
    """
    try:
        detector = FakeNewsDetector()
        prediction, confidence = detector.predict(text, return_confidence=True)
        
        if prediction == "Fake":
            result_str = f"❌ FAKE (Confidence: {confidence:.2%})"
        else:
            result_str = f"✅ TRUE (Confidence: {confidence:.2%})"
        
        print(f"\nResult: {result_str}")
        print(f"Headline: {text}\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease train the model first by running: python main.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    # Check if text provided as command line argument
    if len(sys.argv) > 1:
        # Join all arguments as the news text
        news_text = " ".join(sys.argv[1:])
        single_prediction(news_text)
    else:
        # Run interactive mode
        interactive_mode()

