import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceMonitor:
    """Monitor model performance metrics"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_interval = config['monitoring']['metrics_interval']
        self.performance_threshold = config['monitoring']['performance_threshold']
        self.latency_threshold = config['monitoring']['latency_threshold']

        # Initialize metrics storage
        self.metrics_storage = config['storage']['metrics_storage']
        os.makedirs(self.metrics_storage, exist_ok=True)

        # Initialize metric collectors
        self.accuracy_buffer = deque(maxlen=1000)
        self.latency_buffer = deque(maxlen=1000)
        self.prediction_counts = defaultdict(int)
        self.error_counts = defaultdict(int)

        self.metrics_log = []
        self.prediction_counter = 0

    def log_prediction(self, prediction: str, actual: str, confidence: float,
                       latency: float, model_version: str) -> None:
        """Log a single prediction for monitoring"""

        is_correct = prediction == actual

        # Update buffers
        self.accuracy_buffer.append(is_correct)
        self.latency_buffer.append(latency)

        # Update counters
        self.prediction_counts[model_version] += 1
        if not is_correct:
            self.error_counts[f"{model_version}_error"] += 1

        self.prediction_counter += 1

        # Log metrics at specified intervals
        if self.prediction_counter % self.metrics_interval == 0:
            self._log_metrics(model_version)

    def _log_metrics(self, model_version: str) -> None:
        """Log aggregated metrics"""

        if not self.accuracy_buffer or not self.latency_buffer:
            return

        # Calculate metrics
        current_accuracy = np.mean(list(self.accuracy_buffer))
        current_latency = np.mean(list(self.latency_buffer))

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'accuracy': current_accuracy,
            'avg_latency': current_latency,
            'prediction_count': self.prediction_counter,
            'error_rate': 1 - current_accuracy,
            'throughput': self.prediction_counter / (time.time() - self.start_time) if hasattr(self,
                                                                                               'start_time') else 0
        }

        self.metrics_log.append(metrics)

        # Check for alerts
        self._check_alerts(metrics)

        print(f"Metrics logged: Accuracy={current_accuracy:.3f}, Latency={current_latency:.3f}s")

    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check for performance alerts"""

        alerts = []

        if metrics['accuracy'] < self.performance_threshold:
            alerts.append(f"Low accuracy alert: {metrics['accuracy']:.3f} < {self.performance_threshold}")

        if metrics['avg_latency'] > self.latency_threshold:
            alerts.append(f"High latency alert: {metrics['avg_latency']:.3f}s > {self.latency_threshold}s")

        if alerts:
            print("🚨 ALERTS:")
            for alert in alerts:
                print(f"  - {alert}")

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""

        if not self.accuracy_buffer or not self.latency_buffer:
            return {}

        return {
            'accuracy': np.mean(list(self.accuracy_buffer)),
            'avg_latency': np.mean(list(self.latency_buffer)),
            'prediction_count': self.prediction_counter,
            'error_rate': 1 - np.mean(list(self.accuracy_buffer))
        }

    def save_metrics(self) -> str:
        """Save metrics to file"""

        filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.metrics_storage, filename)

        with open(filepath, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)

        return filepath

    def generate_report(self) -> str:
        """Generate performance report"""

        if not self.metrics_log:
            return "No metrics available for report generation"

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extract data for plotting
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in self.metrics_log]
        accuracies = [m['accuracy'] for m in self.metrics_log]
        latencies = [m['avg_latency'] for m in self.metrics_log]
        error_rates = [m['error_rate'] for m in self.metrics_log]

        # Accuracy over time
        axes[0, 0].plot(timestamps, accuracies, marker='o')
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Latency over time
        axes[0, 1].plot(timestamps, latencies, marker='o', color='orange')
        axes[0, 1].set_title('Latency Over Time')
        axes[0, 1].set_ylabel('Latency (s)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Error rate distribution
        axes[1, 0].hist(error_rates, bins=20, alpha=0.7, color='red')
        axes[1, 0].set_title('Error Rate Distribution')
        axes[1, 0].set_xlabel('Error Rate')
        axes[1, 0].set_ylabel('Frequency')

        # Accuracy vs Latency scatter
        axes[1, 1].scatter(latencies, accuracies, alpha=0.6)
        axes[1, 1].set_title('Accuracy vs Latency')
        axes[1, 1].set_xlabel('Latency (s)')
        axes[1, 1].set_ylabel('Accuracy')

        plt.tight_layout()

        # Save plot
        plot_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_filepath = os.path.join(self.metrics_storage, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_filepath