import random
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats

@dataclass
class ABTestResult:
    """A/B test result data structure"""
    model_a_accuracy: float
    model_b_accuracy: float
    model_a_latency: float
    model_b_latency: float
    sample_size_a: int
    sample_size_b: int
    p_value: float
    is_significant: bool
    winner: str


class ABTester:
    """A/B testing framework for model comparison"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.traffic_split = config['ab_testing']['traffic_split']
        self.minimum_samples = config['ab_testing']['minimum_samples']
        self.significance_level = config['ab_testing']['significance_level']

        # Initialize experiment data
        self.experiment_data = {
            'model_a': {'predictions': [], 'latencies': [], 'correct': []},
            'model_b': {'predictions': [], 'latencies': [], 'correct': []}
        }

        self.metrics_storage = config['storage']['metrics_storage']
        os.makedirs(self.metrics_storage, exist_ok=True)

    def assign_model(self) -> str:
        """Assign a user to model A or B based on traffic split"""
        return 'model_a' if random.random() < self.traffic_split else 'model_b'

    def record_prediction(self, model_version: str, prediction: str,
                          actual: str, latency: float) -> None:
        """Record a prediction result for A/B testing"""

        is_correct = prediction == actual

        self.experiment_data[model_version]['predictions'].append(prediction)
        self.experiment_data[model_version]['correct'].append(is_correct)
        self.experiment_data[model_version]['latencies'].append(latency)

    def calculate_metrics(self, model_version: str) -> Dict[str, float]:
        """Calculate metrics for a model version"""

        data = self.experiment_data[model_version]

        if not data['correct']:
            return {'accuracy': 0.0, 'avg_latency': 0.0, 'sample_size': 0}

        accuracy = np.mean(data['correct'])
        avg_latency = np.mean(data['latencies'])
        sample_size = len(data['correct'])

        return {
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'sample_size': sample_size
        }

    def run_statistical_test(self) -> ABTestResult:
        """Run statistical significance test between models"""

        metrics_a = self.calculate_metrics('model_a')
        metrics_b = self.calculate_metrics('model_b')

        # Check if we have enough samples
        if (metrics_a['sample_size'] < self.minimum_samples or
            metrics_b['sample_size'] < self.minimum_samples):
            return ABTestResult(
                model_a_accuracy=metrics_a['accuracy'],
                model_b_accuracy=metrics_b['accuracy'],
                model_a_latency=metrics_a['avg_latency'],
                model_b_latency=metrics_b['avg_latency'],
                sample_size_a=metrics_a['sample_size'],
                sample_size_b=metrics_b['sample_size'],
                p_value=1.0,
                is_significant=False,
                winner="insufficient_data"
            )

        # Perform two-proportion z-test for accuracy
        successes_a = sum(self.experiment_data['model_a']['correct'])
        successes_b = sum(self.experiment_data['model_b']['correct'])

        # Calculate z-statistic and p-value
        p1 = successes_a / metrics_a['sample_size']
        p2 = successes_b / metrics_b['sample_size']

        p_pooled = (successes_a + successes_b) / (metrics_a['sample_size'] + metrics_b['sample_size'])

        if p_pooled == 0 or p_pooled == 1:
            z_stat = 0
            p_value = 1.0
        else:
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/metrics_a['sample_size'] + 1/metrics_b['sample_size']))
            z_stat = (p1 - p2) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        is_significant = p_value < self.significance_level

        # Determine winner
        if is_significant:
            if metrics_a['accuracy'] > metrics_b['accuracy']:
                winner = "model_a"
            else:
                winner = "model_b"
        else:
            winner = "no_significant_difference"

        return ABTestResult(
            model_a_accuracy=metrics_a['accuracy'],
            model_b_accuracy=metrics_b['accuracy'],
            model_a_latency=metrics_a['avg_latency'],
            model_b_latency=metrics_b['avg_latency'],
            sample_size_a=metrics_a['sample_size'],
            sample_size_b=metrics_b['sample_size'],
            p_value=p_value,
            is_significant=is_significant,
            winner=winner
        )

    def save_experiment_results(self, results: ABTestResult) -> None:
        """Save A/B test results to file"""

        result_data = {
            'timestamp': datetime.now().isoformat(),
            'results': {
                'model_a_accuracy': results.model_a_accuracy,
                'model_b_accuracy': results.model_b_accuracy,
                'model_a_latency': results.model_a_latency,
                'model_b_latency': results.model_b_latency,
                'sample_size_a': results.sample_size_a,
                'sample_size_b': results.sample_size_b,
                'p_value': results.p_value,
                'is_significant': results.is_significant,
                'winner': results.winner
            }
        }

        filename = f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.metrics_storage, filename)

        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"A/B test results saved to {filepath}")