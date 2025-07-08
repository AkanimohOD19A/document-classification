import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from typing import Dict, List, Tuple
import logging

# SetUp Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentDataset(Dataset):
    """Custom dataset for document classification."""

    def __init__(self, texts: List[str],
                 labels: List[int],
                 tokenizer,
                 max_length: int=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DocumentClassificationTrainer:
    """Trainer for document classification models"""

    def __init__(self):
        # label mapping:
        self.label_to_id = {
            'business': 0,
            'technology': 1,
            'sports': 2,
            'entertainment': 3,
            'politics': 4
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.num_labels = len(self.label_to_id)

    def create_training_data(self) -> Tuple[List[str], List[str]]:
        """Create comprehensive training data"""

        # Samples
        with open("junk/training_samples.json", "r", encoding="utf-8") as f:
            samples = json.load(f)

        business_texts = samples[0]['business']
        technology_texts = samples[1]['technology']
        sports_texts = samples[2]['sports']
        entertainment_texts = samples[3]['entertainment']
        politics_texts = samples[4]['politics']

        # Combine all data
        texts = (business_texts +
                 technology_texts +
                 sports_texts +
                 entertainment_texts +
                 politics_texts)

        labels = (['business'] * len(business_texts) +
                  ['technology'] * len(technology_texts) +
                  ['sports'] * len(sports_texts) +
                  ['entertainment'] * len(entertainment_texts) +
                  ['politics'] * len(politics_texts))

        logger.info(f"Created training data: {len(texts)} samples")
        logger.info(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

        return texts, labels

    def prepare_model_and_tokenizer(self, model_name: str):
        """Prepare model and tokenizer based on model name"""

        if 'distilbert' in model_name.lower():
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_labels
            )
        elif 'gpt2' in model_name.lower() or 'dialogpt' in model_name.lower():
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            # GPT2 doesn't have a pad token, so we add one
            tokenizer.pad_token = tokenizer.eos_token

            # Load model with additional safety measures for GPT2
            model = GPT2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_labels,
                torch_dtype=torch.float32,  # Force float32 to avoid precision issues
                low_cpu_mem_usage=True      # Use less memory during loading
            )
            # Set pad_token_id for the model
            model.config.pad_token_id = tokenizer.pad_token_id

            # Add special handling for DialoGPT
            if 'dialogpt' in model_name.lower():
                # DialoGPT sometimes has issues with sequence classification
                # Let's make sure the model is properly configured
                model.config.use_cache = False  # Disable caching to avoid memory issues

        else:
            # Generic transformer model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_labels
            )

        return model, tokenizer

    def compute_metrics(self, eval_pred: EvalPrediction):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}

    def train_model(self, model_name: str,
                    output_dir: str,
                    epochs: int = 3):
        """Train a classification model"""

        logger.info(f"Training model {model_name}")

        # Get training data
        texts, labels = self.create_training_data()

        # Convert labels to ids
        label_ids = [self.label_to_id[label] for label in labels]

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
        )

        logger.info(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

        # Prep. model and tokenizer
        model, tokenizer = self.prepare_model_and_tokenizer(model_name)

        # Create datasets
        train_dataset = DocumentDataset(train_texts, train_labels, tokenizer)
        val_dataset = DocumentDataset(val_texts, val_labels, tokenizer)

        # Training arguments - more conservative settings for GPT models
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4 if 'gpt' in model_name.lower() or 'dialogpt' in model_name.lower() else 8,  # Smaller batch for GPT
            per_device_eval_batch_size=4 if 'gpt' in model_name.lower() or 'dialogpt' in model_name.lower() else 8,   # Smaller batch for GPT
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            dataloader_pin_memory=False,    # Disable pin_memory to avoid warnings
            fp16=False,                     # Disable mixed precision for stability
            gradient_checkpointing=True if 'gpt' in model_name.lower() or 'dialogpt' in model_name.lower() else False,  # Memory optimization for GPT
        )

        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Save label mapping
        with open(f'{output_dir}/label_mapping.json', 'w') as f:
            json.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label
            }, f, indent=2)

        logger.info(f"Model saved to {output_dir}")

        return trainer

def main():
    """Main training function"""

    trainer = DocumentClassificationTrainer()

    # Models to train
    models_to_train = [
        {
            'name': 'distilbert-base-uncased',
            'output_dir': './models/distilbert_classifier'
        },
        {
            # 'name': 'microsoft/DialoGPT-medium',
            'name': 'gpt2',  # Use base GPT2 instead of DialoGPT for stability
            'output_dir': './models/gpt2_classifier'
        }
    ]

    for model_config in models_to_train:
        try:
            logger.info(f"Starting training for {model_config['name']}")
            trainer.train_model(
                model_name=model_config['name'],
                output_dir=model_config['output_dir'],
                epochs=3
            )
            logger.info(f"Completed training for {model_config['name']}")
        except Exception as e:
            logger.error(f"Error training {model_config['name']}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue with next model even if one fails
            continue


if __name__ == '__main__':
    main()