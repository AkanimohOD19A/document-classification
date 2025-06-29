import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi, Repository
import torch

class ModelManager:
    """Manage model versions and deployment"""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.registry_path = self.config['storage']['model_registry']
        os.makedirs(self.registry_path, exist_ok=True)

        self.hf_api = HfApi() # initialize HuggingFace

    def register_model(self, model_name: str,
                       version: str,
                       model_path: str,
                       metrics: Dict[str, float]) -> str:
        """Register a new model version with the registry"""

        model_id = f"{model_name}_{version}"
        model_info = {
            "model_name": model_name,
            "version": version,
            "model_id": model_id,
            "model_path": model_path,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }

        # Save model metadata
        metadata_path = os.path.join(self.registry_path, f"{model_id}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"Model {model_id} registered successfully")
        return model_id

    def _validate_local_model_path(self, model_path: str) -> bool:
        """Validate that the local model path exists and contains required files"""
        if not os.path.exists(model_path):
            return False

        # Check for required model files
        required_files = ['config.json']
        model_file_exists = (
            os.path.exists(os.path.join(model_path, 'model.safetensors')) or
            os.path.exists(os.path.join(model_path, 'pytorch_model.bin'))
        )

        config_exists = os.path.exists(os.path.join(model_path, 'config.json'))

        return model_file_exists and config_exists

    def load_model(self, model_name: str, version: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, Dict]:
        """Load a model version - supports both local and HuggingFace models"""
        model_config = self.config['models'].get(f"version_{version}")
        if not model_config:
            raise ValueError(f"Model {model_name} version {version} has no model config")

        model_path = model_config['name']

        print(f"Loading model from: {model_path}")

        try:
            # Check if it's a local path
            if model_path.startswith('./') or model_path.startswith('/') or os.path.exists(model_path):
                if not self._validate_local_model_path(model_path):
                    raise ValueError(f"Local model path {model_path} does not exist or is missing required files")

                print(f"Loading local retrained model from: {model_path}")

                # Load tokenizer - try local first, fallback to base model if needed
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                except Exception as e:
                    print(f"Warning: Could not load tokenizer from {model_path}, using base model tokenizer")
                    # Fallback to base model tokenizer
                    base_model_name = self._get_base_model_name(model_config.get('model_type', 'distilbert'))
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

                # Load model
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=model_config['num_labels'],
                    local_files_only=True
                )

            else:
                # Load from HuggingFace Hub
                print(f"Loading model from HuggingFace Hub: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=model_config['num_labels']
                )

            print(f"✅ Successfully loaded model: {model_path}")
            return tokenizer, model, model_config

        except Exception as e:
            print(f"❌ Error loading model {model_path}: {str(e)}")
            raise

    def _get_base_model_name(self, model_type: str) -> str:
        """Get the base model name for tokenizer fallback"""
        base_models = {
            'distilbert': 'distilbert-base-uncased',
            'gpt2': 'gpt2',
            'bert': 'bert-base-uncased',
            'roberta': 'roberta-base'
        }
        return base_models.get(model_type, 'distilbert-base-uncased')

    def get_best_checkpoint(self, model_dir: str) -> str:
        """Find the best checkpoint in a model directory based on highest checkpoint number"""
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist")

        checkpoints = []
        for item in os.listdir(model_dir):
            if item.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir, item)):
                try:
                    checkpoint_num = int(item.split('-')[1])
                    checkpoints.append((checkpoint_num, item))
                except ValueError:
                    continue

        if not checkpoints:
            raise ValueError(f"No valid checkpoints found in {model_dir}")

        # Return the path to the highest numbered checkpoint
        best_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
        return os.path.join(model_dir, best_checkpoint)

    def list_models(self) -> Dict[str, Any]:
        """List all registered model versions"""

        models = {}
        for filename in os.listdir(self.registry_path):
            if filename.endswith('_metadata.json'):
                with open(os.path.join(self.registry_path, filename), 'r') as f:
                    model_info = json.load(f)
                    models[model_info['model_id']] = model_info

        return models

    def promote_model(self, model_id: str, stage: str = "production") -> None:
        """Promote a model version to specific stage"""

        metadata_path = os.path.join(self.registry_path, f"{model_id}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                model_info = json.load(f)

            model_info['stage'] = stage
            model_info['promoted_at'] = datetime.now().isoformat()

            with open(metadata_path, "w") as f:
                json.dump(model_info, f, indent=2)

            print(f"Model {model_id} promoted to {stage} successfully")
        else:
            raise ValueError(f"Model {model_id} does not exist")