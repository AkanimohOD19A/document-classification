import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    GPT2Tokenizer, GPT2ForSequenceClassification
)
import json
import numpy as np
from typing import Dict, Tuple, List
import time
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedDocumentClassifier:
    """Improved document classifier with proper trained models"""

    def __init__(self, model_path: str, model_type: str = "distilbert"):
        """
        Initialize classifier with trained model

        Args:
            model_path: Path to the trained model directory
            model_type: Type of model ("distilbert" or "gpt2")
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load label mapping
        self.label_mapping = self._load_label_mapping()
        self.id_to_label = self.label_mapping['id_to_label']
        self.label_to_id = self.label_mapping['label_to_id']

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded {model_type} model from {model_path}")

    def _load_label_mapping(self) -> Dict:
        """Load label mapping from file"""
        mapping_file = os.path.join(self.model_path, 'label_mapping.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
        else:
            # Default mapping if file doesn't exist
            return {
                'label_to_id': {
                    'business': 0,
                    'technology': 1,
                    'sports': 2,
                    'entertainment': 3,
                    'politics': 4
                },
                'id_to_label': {
                    '0': 'business',
                    '1': 'technology',
                    '2': 'sports',
                    '3': 'entertainment',
                    '4': 'politics'
                }
            }

    def _load_model_and_tokenizer(self):
        """Load trained model and tokenizer"""
        if self.model_type == "distilbert":
            tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        elif self.model_type == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            model = GPT2ForSequenceClassification.from_pretrained(self.model_path)
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            # Generic transformer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        return model, tokenizer

    def predict(self, text: str) -> Tuple[str, float, float]:
        """
        Predict the class of a text document

        Args:
            text: Input text to classify

        Returns:
            Tuple of (predicted_label, confidence, inference_time)
        """
        start_time = time.time()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Get prediction and confidence
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()

        inference_time = time.time() - start_time
        predicted_label = self.id_to_label[str(predicted_id)]

        return predicted_label, confidence, inference_time


class ImprovedABTester:
    """Improved A/B testing with proper statistical analysis"""

    def __init__(self, classifier_a: ImprovedDocumentClassifier,
                 classifier_b: ImprovedDocumentClassifier):
        self.classifier_a = classifier_a
        self.classifier_b = classifier_b
        self.results = []
        self._current_model = None
        self._traffic_split = 0.5  # 50-50 split

    def select_model(self) -> str:
        """Select model based on traffic split"""
        if np.random.random() < self._traffic_split:
            self._current_model = "model_a"
            return "model_a"
        else:
            self._current_model = "model_b"
            return "model_b"

    def predict(self, text: str) -> Dict:
        """Make prediction using selected model"""
        model_version = self.select_model()

        if model_version == "model_a":
            label, confidence, inference_time = self.classifier_a.predict(text)
        else:
            label, confidence, inference_time = self.classifier_b.predict(text)

        return {
            "predicted_label": label,
            "confidence": confidence,
            "model_version": model_version,
            "inference_time": inference_time
        }

    def record_result(self, text: str, predicted_label: str, actual_label: str,
                      confidence: float, model_version: str, inference_time: float):
        """Record prediction result"""
        is_correct = predicted_label.lower() == actual_label.lower()

        result = {
            "text": text,
            "predicted_label": predicted_label,
            "actual_label": actual_label,
            "correct": is_correct,
            "confidence": confidence,
            "model_version": model_version,
            "inference_time": inference_time,
            "timestamp": datetime.now().isoformat()
        }

        self.results.append(result)
        return result

    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics for both models"""
        if not self.results:
            return {}

        model_a_results = [r for r in self.results if r["model_version"] == "model_a"]
        model_b_results = [r for r in self.results if r["model_version"] == "model_b"]

        def calculate_metrics(results):
            if not results:
                return {"accuracy": 0, "avg_confidence": 0, "avg_latency": 0, "count": 0}

            accuracy = sum(1 for r in results if r["correct"]) / len(results)
            avg_confidence = np.mean([r["confidence"] for r in results])
            avg_latency = np.mean([r["inference_time"] for r in results])

            return {
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "avg_latency": avg_latency,
                "count": len(results)
            }

        return {
            "model_a": calculate_metrics(model_a_results),
            "model_b": calculate_metrics(model_b_results),
            "overall": calculate_metrics(self.results)
        }


def create_sample_test_data() -> List[Dict]:
    """Create sample test data with correct labels"""
    return [
        {
            "text": "Apple Inc. reported strong quarterly earnings driven by iPhone sales and services revenue growth.",
            "actual_label": "business"
        },
        {
            "text": "The new AI breakthrough in natural language processing could revolutionize how we interact with computers.",
            "actual_label": "technology"
        },
        {
            "text": "The Lakers defeated the Warriors 112-108 in an exciting overtime finish at Staples Center.",
            "actual_label": "sports"
        },
        {
            "text": "The upcoming superhero movie broke box office records in its opening weekend worldwide.",
            "actual_label": "entertainment"
        },
        {
            "text": "Congress passed the new infrastructure bill with bipartisan support after months of negotiations.",
            "actual_label": "politics"
        },
        {
            "text": "Tesla's stock price surged after announcing record delivery numbers for the quarter.",
            "actual_label": "business"
        },
        {
            "text": "Researchers developed a new quantum computing algorithm that could solve complex optimization problems.",
            "actual_label": "technology"
        },
        {
            "text": "Manchester United signed a new midfielder in a record-breaking transfer deal worth $100 million.",
            "actual_label": "sports"
        },
        {
            "text": "The film festival showcased independent movies from emerging directors around the world.",
            "actual_label": "entertainment"
        },
        {
            "text": "The Senate committee held hearings on the proposed healthcare reform legislation.",
            "actual_label": "politics"
        }
    ]


def run_improved_test():
    """Run test with improved classifiers"""

    # Check if trained models exist
    distilbert_path = "./models/distilbert_classifier"
    gpt2_path = "./models/gpt2_classifier"

    if not os.path.exists(distilbert_path) or not os.path.exists(gpt2_path):
        print("❌ Trained models not found!")
        print("Please run the training script first:")
        print("python train_classifiers.py")
        return

    try:
        # Load trained models
        print("📦 Loading trained models...")
        classifier_a = ImprovedDocumentClassifier(distilbert_path, "distilbert")
        classifier_b = ImprovedDocumentClassifier(gpt2_path, "gpt2")

        # Create A/B tester
        ab_tester = ImprovedABTester(classifier_a, classifier_b)

        # Test data
        test_data = create_sample_test_data()

        print(f"🧪 Testing with {len(test_data)} samples...")
        print("-" * 80)

        # Run predictions
        results = []
        for i, sample in enumerate(test_data, 1):
            text = sample["text"]
            actual_label = sample["actual_label"]

            # Get prediction
            prediction = ab_tester.predict(text)

            # Record result
            result = ab_tester.record_result(
                text=text,
                predicted_label=prediction["predicted_label"],
                actual_label=actual_label,
                confidence=prediction["confidence"],
                model_version=prediction["model_version"],
                inference_time=prediction["inference_time"]
            )

            results.append(result)

            # Print result
            status = "✅" if result["correct"] else "❌"
            print(f"{status} Sample {i}: {result['predicted_label']} "
                  f"(confidence: {result['confidence']:.3f}, "
                  f"model: {result['model_version']}, "
                  f"time: {result['inference_time']:.3f}s)")

            if not result["correct"]:
                print(f"   Expected: {actual_label}")
                print(f"   Text: {text[:100]}...")

        print("-" * 80)

        # Print performance metrics
        metrics = ab_tester.get_performance_metrics()

        print("📊 Performance Summary:")
        print(f"Overall Accuracy: {metrics['overall']['accuracy']:.1%}")
        print(f"Average Confidence: {metrics['overall']['avg_confidence']:.3f}")
        print(f"Average Latency: {metrics['overall']['avg_latency']:.3f}s")

        print("\n🅰️ Model A (DistilBERT):")
        print(f"Accuracy: {metrics['model_a']['accuracy']:.1%}")
        print(f"Samples: {metrics['model_a']['count']}")
        print(f"Avg Confidence: {metrics['model_a']['avg_confidence']:.3f}")

        print("\n🅱️ Model B (GPT-2):")
        print(f"Accuracy: {metrics['model_b']['accuracy']:.1%}")
        print(f"Samples: {metrics['model_b']['count']}")
        print(f"Avg Confidence: {metrics['model_b']['avg_confidence']:.3f}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"./results/improved_test_results_{timestamp}.json"

        os.makedirs("./results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump({
                "results": results,
                "metrics": metrics,
                "timestamp": timestamp
            }, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_improved_test()