# Document Classification with Model Versioning - Complete Tutorial

## Project Overview
This tutorial will guide you through building a production-ready document classification system with proper model versioning, A/B testing, and performance monitoring.

## Prerequisites
- Python 3.8+
- Basic understanding of machine learning
- Familiarity with transformers and Hugging Face

## Project Structure
```
document_classifier/
├── config/
│   ├── __init__.py
│   └── config.yaml
├── data/
│   ├── sample_documents.json
│   └── processed/
├── models/
│   ├── __init__.py
│   ├── model_manager.py
│   └── classifier.py
│   └── distilbert_classifier/
   |   |-- checkpoint-10
   |   |   |-- config.json
   |   |   |-- model.safetensors
   |   |   |-- optimizer.pt
   |   |   |-- rng_state.pth
   |   |   |-- scheduler.pt
   |   |   |-- trainer_state.json
   |   |   `-- training_args.bin
   |   |-- checkpoint-15
   ..
│   └── gpt2_classifier/
   |   |-- checkpoint-10
   |   |   |-- config.json
   |   |   |-- model.safetensors
   |   |   |-- optimizer.pt
   |   |   |-- rng_state.pth
   |   |   |-- scheduler.pt
   |   |   |-- trainer_state.json
   |   |   `-- training_args.bin
   ..
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py
│   └── ab_testing.py
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── batch_processor.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── tests/
│   └── test_classifier.py
├── requirements.txt
├── main.py
└── README.md
```

## Usage Examples and Best Practices

### Running the System

1. **Setup Environment**:
```bash
python deploy.py setup
```

2. **Run Health Check**:
```bash
python deploy.py health-check
```

3. **Batch Processing**:
```bash
python main.py --mode batch --data data/sample_documents.json
```

4. **Interactive Mode**:
```bash
python main.py --mode interactive
```

5. **A/B Testing**:
```bash
python main.py --mode test --data data/sample_documents.json
```

6. **Run Tests**:
```bash
python -m pytest tests/ -v
```

### Best Practices

#### 1. Model Versioning
- Always register new models with proper versioning
- Include performance metrics with each model version
- Use semantic versioning (v1.0.0, v1.1.0, etc.)
- Keep model metadata for reproducibility

#### 2. A/B Testing
- Ensure sufficient sample size for statistical significance
- Monitor both accuracy and latency metrics
- Use proper statistical tests for comparison
- Document experiment parameters and results

#### 3. Performance Monitoring
- Set appropriate thresholds for alerts
- Monitor multiple metrics (accuracy, latency, throughput)
- Generate regular performance reports
- Track model drift over time

#### 4. Batch Processing
- Use appropriate batch sizes for your hardware
- Implement proper error handling
- Monitor resource usage during batch jobs
- Save intermediate results for recovery

#### 5. Configuration Management
- Use configuration files for all parameters
- Version control your configurations
- Use different configs for different environments
- Document all configuration options

### Expected Output

When you run the system, you should see output like:
```
🚀 Starting Document Classification System
📦 Loading models...
✅ Models loaded successfully
📊 Processing documents from data/sample_documents.json
Processing documents: 100%|████████████| 10/10 [00:02<00:00,  4.21it/s]
Results saved to data/processed/classification_results_20241201_143022.json

📈 Batch Processing Report:
  Total Documents: 10
  Overall Accuracy: 0.800
  Average Confidence: 0.856
  Average Latency: 0.123s
  Model A Usage: 5
  Model B Usage: 5

🧪 A/B Test Results:
  Model A Accuracy: 0.800
  Model B Accuracy: 0.800
  P-value: 1.000000
  Significant: False
  Winner: no_significant_difference

📊 Performance report saved to: monitoring/metrics/performance_report_20241201_143025.png
📈 Metrics saved to: monitoring/metrics/metrics_20241201_143025.json
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Use CPU instead of GPU
   - Use smaller model variants

2. **Model Loading Errors**:
   - Check internet connection for downloading models
   - Verify model names in configuration
   - Ensure sufficient disk space

3. **Statistical Test Failures**:
   - Increase sample size
   - Check data quality
   - Verify label distribution

### Performance Optimization

1. **Batch Size Tuning**:
   - Start with small batches (8-16)
   - Increase gradually based on memory
   - Monitor GPU utilization

2. **Model Selection**:
   - Use DistilBERT for faster inference
   - Consider quantized models for production
   - Cache tokenizer outputs when possible

3. **Monitoring Overhead**:
   - Adjust metrics logging frequency
   - Use sampling for large datasets
   - Implement async logging

## Next Steps

1. **Advanced Features**:
   - Add model ensemble capabilities
   - Implement online learning
   - Add drift detection algorithms

2. **Production Enhancements**:
   - Add REST API endpoints
   - Implement proper logging
   - Add database integration

3. **MLOps Integration**:
   - Connect to MLflow or similar platforms
   - Add CI/CD pipelines
   - Implement automated retraining

This tutorial provides a complete foundation for building production-ready ML systems with proper versioning, testing, and monitoring capabilities.