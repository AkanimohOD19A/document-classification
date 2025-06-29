#!/usr/bin/env python3
"""
Script to validate that retrained models can be loaded successfully
"""

import os
import yaml
import sys
from models.model_manager import ModelManager
from models.classifier import DocumentClassifier

def validate_model_directory(model_path):
    """Validate a model directory structure"""
    print(f"\n📁 Validating path: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Path does not exist: {model_path}")
        return False

    # Check if we're pointing to a checkpoint directory directly
    if os.path.basename(model_path).startswith('checkpoint-'):
        print(f"📍 Validating checkpoint directory: {os.path.basename(model_path)}")
        return validate_checkpoint_directory(model_path)

    # Otherwise, look for checkpoints in the directory
    checkpoints = []
    for item in os.listdir(model_path):
        if item.startswith('checkpoint-') and os.path.isdir(os.path.join(model_path, item)):
            checkpoints.append(item)

    if not checkpoints:
        print(f"❌ No checkpoints found in {model_path}")
        return False

    print(f"✅ Found {len(checkpoints)} checkpoints: {sorted(checkpoints)}")

    # Validate the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    checkpoint_path = os.path.join(model_path, latest_checkpoint)
    return validate_checkpoint_directory(checkpoint_path)

def validate_checkpoint_directory(checkpoint_path):
    """Validate a specific checkpoint directory"""
    print(f"🔍 Validating checkpoint: {checkpoint_path}")

    required_files = ['config.json']
    model_files = ['model.safetensors', 'pytorch_model.bin']

    # Check config.json
    if os.path.exists(os.path.join(checkpoint_path, 'config.json')):
        print(f"✅ Found config.json")
    else:
        print(f"❌ Missing config.json")
        return False

    # Check model file
    model_file_found = False
    for model_file in model_files:
        if os.path.exists(os.path.join(checkpoint_path, model_file)):
            print(f"✅ Found {model_file}")
            model_file_found = True
            break

    if not model_file_found:
        print(f"❌ No model file found")
        return False

    return True


def test_model_loading():
    """Test loading models with current configuration"""
    print("\n🔧 Testing model loading...")

    try:
        # Initialize model manager
        model_manager = ModelManager('config/config.yaml')

        # Test loading model A
        print("\n📦 Loading Model A...")
        tokenizer_a, model_a, config_a = model_manager.load_model("model", "a")
        classifier_a = DocumentClassifier(tokenizer_a, model_a, config_a)
        print("✅ Model A loaded successfully")

        # Test loading model B
        print("\n📦 Loading Model B...")
        tokenizer_b, model_b, config_b = model_manager.load_model("model", "b")
        classifier_b = DocumentClassifier(tokenizer_b, model_b, config_b)
        print("✅ Model B loaded successfully")

        # Test predictions
        test_text = "This is a test document about artificial intelligence and machine learning."

        print(f"\n🧪 Testing predictions with text: '{test_text[:50]}...'")

        # Test Model A prediction
        pred_a, conf_a, time_a = classifier_a.predict_single(test_text)
        print(f"Model A - Prediction: {pred_a}, Confidence: {conf_a:.3f}, Time: {time_a:.3f}s")

        # Test Model B prediction
        pred_b, conf_b, time_b = classifier_b.predict_single(test_text)
        print(f"Model B - Prediction: {pred_b}, Confidence: {conf_b:.3f}, Time: {time_b:.3f}s")

        print("\n✅ All models loaded and tested successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during model loading/testing: {str(e)}")
        return False


def main():
    """Main validation function"""
    print("🚀 Model Validation Script")
    print("=" * 50)

    # Load configuration
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # Get model paths from config
    model_paths = {}
    for version, model_config in config['models'].items():
        model_paths[version] = model_config['name']

    print("📋 Configured model paths:")
    for version, path in model_paths.items():
        print(f"  {version}: {path}")

    # Validate each model directory
    all_valid = True
    for version, path in model_paths.items():
        if path.startswith('./') or os.path.exists(path):
            # It's a local path, validate it
            if not validate_model_directory(path):
                all_valid = False
        else:
            print(f"\n📡 {version}: Using HuggingFace model - {path}")

    if not all_valid:
        print("\n❌ Some model directories failed validation")
        sys.exit(1)

    # Test actual model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed")
        sys.exit(1)

    print("\n🎉 All validations passed! Your retrained models are ready to use.")
    print("\nNext steps:")
    print("1. Run: python main.py --mode batch --data data/sample_documents.json")
    print("2. Or run: python main.py --mode interactive")


if __name__ == "__main__":
    main()