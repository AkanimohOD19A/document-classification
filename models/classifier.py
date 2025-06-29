import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import time
import numpy as np

class DocumentClassifier:
    """Document classification model wrapper"""

    def __init__(self, tokenizer, model, config: Dict):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Fix padding token issue
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Add a special padding token if no eos_token exists
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))

        # Categorize mapping
        self.categories = [
            "business",
            "technology",
            "sports",
            "entertainment",
            "politics"
        ]

    def preprocess_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess text for model classification."""

        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.get('max_length', 512),
            return_tensors="pt"
        )

        return {k: v.to(self.device) for k, v in encoding.items()}

    def predict(self, texts: List[str]) -> Tuple[List[str], List[float], float]:
        """Make predictions on a batch of texts"""

        start_time = time.time()

        # Preprocess
        inputs = self.preprocess_text(texts)

        # Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        # Convert to labels and confidence scores
        predicted_labels = [self.categories[pred.item()]
                            for pred in predictions]
        confidence_scores = [prob.max().item() for prob in probabilities]  # Fixed: .item() instead of .items()

        inference_time = time.time() - start_time

        return predicted_labels, confidence_scores, inference_time


    def predict_single(self, text: str) -> Tuple[str, float, float]:
        """Make prediction on a single text"""

        labels, scores, inference_time = self.predict([text])
        return labels[0], scores[0], inference_time

