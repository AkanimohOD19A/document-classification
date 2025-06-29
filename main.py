import yaml
import json
import argparse
from models.model_manager import ModelManager
from models.classifier import DocumentClassifier
from monitoring.ab_testing import ABTester
from monitoring.metrics import PerformanceMonitor
from pipeline.batch_processor import BatchProcessor


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Document Classification System')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--data', default='data/sample_documents.json', help='Input data file')
    parser.add_argument('--mode', choices=['batch', 'interactive', 'test'], default='batch', help='Run mode')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("🚀 Starting Document Classification System")

    # Initialize components
    model_manager = ModelManager(args.config)
    ab_tester = ABTester(config)
    monitor = PerformanceMonitor(config)

    # Load models
    print("📦 Loading models...")
    tokenizer_a, model_a, config_a = model_manager.load_model("model", "a")
    tokenizer_b, model_b, config_b = model_manager.load_model("model", "b")

    # Initialize classifiers
    classifier_a = DocumentClassifier(tokenizer_a, model_a, config_a)
    classifier_b = DocumentClassifier(tokenizer_b, model_b, config_b)

    print("✅ Models loaded successfully")

    if args.mode == 'batch':
        run_batch_processing(args.data, classifier_a, classifier_b, ab_tester, monitor, config)
    elif args.mode == 'interactive':
        run_interactive_mode(classifier_a, classifier_b, ab_tester, monitor)
    elif args.mode == 'test':
        run_ab_test(args.data, classifier_a, classifier_b, ab_tester, monitor, config)


def run_batch_processing(data_file, classifier_a, classifier_b, ab_tester, monitor, config):
    """Run batch processing mode"""

    print(f"📊 Processing documents from {data_file}")

    # Initialize batch processor
    processor = BatchProcessor(classifier_a, classifier_b, ab_tester, monitor, config)

    # Process documents
    results_file = processor.process_file(data_file)

    # Load results for report generation
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Generate batch report
    batch_report = processor.generate_batch_report(results)
    print("\n📈 Batch Processing Report:")
    print(f"  Total Documents: {batch_report['summary']['total_documents']}")
    print(f"  Overall Accuracy: {batch_report['summary']['overall_accuracy']:.3f}")
    print(f"  Average Confidence: {batch_report['summary']['average_confidence']:.3f}")
    print(f"  Average Latency: {batch_report['summary']['average_latency']:.3f}s")
    print(f"  Model A Usage: {batch_report['summary']['model_a_usage']}")
    print(f"  Model B Usage: {batch_report['summary']['model_b_usage']}")

    # Run A/B test analysis
    ab_results = ab_tester.run_statistical_test()
    print(f"\n🧪 A/B Test Results:")
    print(f"  Model A Accuracy: {ab_results.model_a_accuracy:.3f}")
    print(f"  Model B Accuracy: {ab_results.model_b_accuracy:.3f}")
    print(f"  P-value: {ab_results.p_value:.6f}")
    print(f"  Significant: {ab_results.is_significant}")
    print(f"  Winner: {ab_results.winner}")

    # Save A/B test results
    ab_tester.save_experiment_results(ab_results)

    # Generate performance report
    report_path = monitor.generate_report()
    print(f"📊 Performance report saved to: {report_path}")

    # Save metrics
    metrics_path = monitor.save_metrics()
    print(f"📈 Metrics saved to: {metrics_path}")


def run_interactive_mode(classifier_a, classifier_b, ab_tester, monitor):
    """Run interactive mode for real-time classification"""

    print("💬 Interactive Mode - Type 'quit' to exit")

    while True:
        text = input("\nEnter text to classify: ")
        if text.lower() == 'quit':
            break

        # Assign model for A/B testing
        model_version = ab_tester.assign_model()
        classifier = classifier_a if model_version == 'model_a' else classifier_b

        # Make prediction
        prediction, confidence, latency = classifier.predict_single(text)

        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Model Version: {model_version}")
        print(f"Inference Time: {latency:.3f}s")

        # Get ground truth for monitoring (optional)
        actual = input("Enter actual label (optional, press enter to skip): ")
        if actual:
            ab_tester.record_prediction(model_version, prediction, actual, latency)
            monitor.log_prediction(prediction, actual, confidence, latency, model_version)


def run_ab_test(data_file, classifier_a, classifier_b, ab_tester, monitor, config):
    """Run comprehensive A/B test"""

    print("🧪 Running A/B Test Mode")

    # Load test data
    with open(data_file, 'r') as f:
        documents = json.load(f)

    print(f"Testing with {len(documents)} documents")

    # Process documents for A/B testing
    for doc in documents:
        model_version = ab_tester.assign_model()
        classifier = classifier_a if model_version == 'model_a' else classifier_b

        prediction, confidence, latency = classifier.predict_single(doc['text'])

        if 'label' in doc:
            ab_tester.record_prediction(model_version, prediction, doc['label'], latency)
            monitor.log_prediction(prediction, doc['label'], confidence, latency, model_version)

    # Analyze results
    ab_results = ab_tester.run_statistical_test()

    print("\n🧪 A/B Test Results:")
    print(f"  Model A: {ab_results.sample_size_a} samples, {ab_results.model_a_accuracy:.3f} accuracy")
    print(f"  Model B: {ab_results.sample_size_b} samples, {ab_results.model_b_accuracy:.3f} accuracy")
    print(f"  Statistical Significance: {ab_results.is_significant} (p={ab_results.p_value:.6f})")
    print(f"  Winner: {ab_results.winner}")

    # Save results
    ab_tester.save_experiment_results(ab_results)


if __name__ == "__main__":
    main()