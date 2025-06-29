import unittest
import yaml
import json
from models.model_manager import ModelManager
from models.classifier import DocumentClassifier
from monitoring.ab_testing import ABTester
from monitoring.metrics import PerformanceMonitor


class TestDocumentClassifier(unittest.TestCase):
    """Test cases for document classification system"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        with open('config/config.yaml', 'r') as f:
            cls.config = yaml.safe_load(f)

        cls.model_manager = ModelManager()
        cls.ab_tester = ABTester(cls.config)
        cls.monitor = PerformanceMonitor(cls.config)

    def test_model_loading(self):
        """Test model loading functionality"""
        tokenizer, model, config = self.model_manager.load_model("model", "a")
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(model)
        self.assertIsNotNone(config)

    def test_classification(self):
        """Test document classification"""
        tokenizer, model, config = self.model_manager.load_model("model", "a")
        classifier = DocumentClassifier(tokenizer, model, config)

        test_text = "Apple Inc. reported strong quarterly earnings."
        prediction, confidence, latency = classifier.predict_single(test_text)

        self.assertIsInstance(prediction, str)
        self.assertTrue(0 <= confidence <= 1)
        self.assertGreater(latency, 0)

    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        tokenizer, model, config = self.model_manager.load_model("model", "a")
        classifier = DocumentClassifier(tokenizer, model, config)

        test_texts = [
            "Tech company announces new AI breakthrough",
            "Basketball team wins championship game"
        ]

        predictions, confidences, latency = classifier.predict(test_texts)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(confidences), 2)
        self.assertGreater(latency, 0)

    def test_ab_testing(self):
        """Test A/B testing functionality"""
        # Record some test predictions
        self.ab_tester.record_prediction("model_a", "business", "business", 0.1)
        self.ab_tester.record_prediction("model_b", "technology", "business", 0.15)

        # Calculate metrics
        metrics_a = self.ab_tester.calculate_metrics("model_a")
        metrics_b = self.ab_tester.calculate_metrics("model_b")

        self.assertIn('accuracy', metrics_a)
        self.assertIn('avg_latency', metrics_a)
        self.assertIn('sample_size', metrics_a)

    def test_performance_monitoring(self):
        """Test performance monitoring"""
        # Log some test predictions
        self.monitor.log_prediction("business", "business", 0.9, 0.1, "model_a")
        self.monitor.log_prediction("technology", "business", 0.8, 0.15, "model_a")

        # Get current metrics
        metrics = self.monitor.get_current_metrics()

        self.assertIn('accuracy', metrics)
        self.assertIn('avg_latency', metrics)
        self.assertIn('prediction_count', metrics)


if __name__ == '__main__':
    unittest.main()