import pandas as pd
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json
import os
from datetime import datetime


class BatchProcessor:
    """Process documents in batches for classification"""

    def __init__(self, classifier_a, classifier_b, ab_tester, monitor, config):
        self.classifier_a = classifier_a
        self.classifier_b = classifier_b
        self.ab_tester = ab_tester
        self.monitor = monitor
        self.config = config
        self.batch_size = config['data']['batch_size']

        # Initialize start time for monitoring
        self.monitor.start_time = datetime.now().timestamp()

    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of documents"""

        results = []

        for doc in tqdm(documents, desc="Processing documents"):
            # Assign model version for A/B testing
            model_version = self.ab_tester.assign_model()

            # Select classifier
            classifier = self.classifier_a if model_version == 'model_a' else self.classifier_b

            # Make prediction
            prediction, confidence, latency = classifier.predict_single(doc['text'])

            # Record for A/B testing (if ground truth available)
            if 'label' in doc:
                self.ab_tester.record_prediction(
                    model_version, prediction, doc['label'], latency
                )

                # Record for monitoring
                self.monitor.log_prediction(
                    prediction, doc['label'], confidence, latency, model_version
                )

            # Store result
            result = {
                'document_id': doc.get('id', len(results)),
                'text': doc['text'][:100] + '...' if len(doc['text']) > 100 else doc['text'],
                'predicted_label': prediction,
                'confidence': confidence,
                'model_version': model_version,
                'inference_time': latency,
                'timestamp': datetime.now().isoformat()
            }

            if 'label' in doc:
                result['actual_label'] = doc['label']
                result['correct'] = prediction == doc['label']

            results.append(result)

        return results

    def process_file(self, file_path: str) -> str:
        """Process documents from a file"""

        # Load documents
        with open(file_path, 'r') as f:
            documents = json.load(f)

        print(f"Processing {len(documents)} documents from {file_path}")

        # Process in batches
        all_results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)

        # Save results
        output_filename = f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(self.config['storage']['data_storage'], 'processed', output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Results saved to {output_path}")
        return output_path

    def generate_batch_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a report for batch processing results"""

        # Calculate overall metrics
        total_docs = len(results)
        correct_predictions = sum(1 for r in results if r.get('correct', False))
        accuracy = correct_predictions / total_docs if total_docs > 0 else 0

        avg_confidence = sum(r['confidence'] for r in results) / total_docs if total_docs > 0 else 0
        avg_latency = sum(r['inference_time'] for r in results) / total_docs if total_docs > 0 else 0

        # Model version distribution
        model_a_count = sum(1 for r in results if r['model_version'] == 'model_a')
        model_b_count = sum(1 for r in results if r['model_version'] == 'model_b')

        # Category distribution
        category_counts = {}
        for result in results:
            category = result['predicted_label']
            category_counts[category] = category_counts.get(category, 0) + 1

        report = {
            'summary': {
                'total_documents': total_docs,
                'overall_accuracy': accuracy,
                'average_confidence': avg_confidence,
                'average_latency': avg_latency,
                'model_a_usage': model_a_count,
                'model_b_usage': model_b_count
            },
            'category_distribution': category_counts,
            'processing_timestamp': datetime.now().isoformat()
        }

        return report