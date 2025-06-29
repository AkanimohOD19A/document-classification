#!/usr/bin/env python3
"""
Deployment script for document classification system
"""

import os
import yaml
import subprocess
import argparse
from models.model_manager import ModelManager


def deploy_model(model_id: str, stage: str = "production"):
    """Deploy a model to specified stage"""

    print(f"🚀 Deploying model {model_id} to {stage}")

    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model manager
    model_manager = ModelManager()

    # Promote model
    model_manager.promote_model(model_id, stage)

    print(f"✅ Model {model_id} deployed to {stage}")


def setup_environment():
    """Set up deployment environment"""

    print("🔧 Setting up environment...")

    # Create necessary directories
    directories = [
        "models/registry",
        "monitoring/metrics",
        "data/processed",
        "logs"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created directory: {directory}")

    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

    print("✅ Environment setup complete")


def run_health_check():
    """Run system health check"""

    print("🏥 Running health check...")

    # Test model loading
    try:
        model_manager = ModelManager()
        tokenizer, model, config = model_manager.load_model("model", "a")
        print("✅ Model A loading: OK")

        tokenizer, model, config = model_manager.load_model("model", "b")
        print("✅ Model B loading: OK")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

    # Test configuration
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loading: OK")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

    print("✅ Health check passed")
    return True


def main():
    parser = argparse.ArgumentParser(description='Deployment script')
    parser.add_argument('action', choices=['deploy', 'setup', 'health-check'])
    parser.add_argument('--model-id', help='Model ID for deployment')
    parser.add_argument('--stage', default='production', help='Deployment stage')

    args = parser.parse_args()

    if args.action == 'setup':
        setup_environment()
    elif args.action == 'deploy':
        if not args.model_id:
            print("❌ Model ID required for deployment")
            exit(1)
        deploy_model(args.model_id, args.stage)
    elif args.action == 'health-check':
        if not run_health_check():
            exit(1)


if __name__ == "__main__":
    main()