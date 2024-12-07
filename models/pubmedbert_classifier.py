import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
import logging
import platform
from typing import Dict, Union
from utils.text_extractor import ProtocolTextExtractor


class ProtocolDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class PubMedBERTClassifier:
    def __init__(self, model_path: str = "./protocol_classifier", max_length: int = 8000):
        """Initialize the classifier."""
        # Device setup for M1 Mac
        if platform.processor() == 'arm':
            device_name = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device_name)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {device_name}")
        
        # Initialize text extractor
        self.extractor = ProtocolTextExtractor(max_length=max_length)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def classify_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Union[str, float]]:
        """Classify a single PDF."""
        # Extract text
        result = self.extractor.extract_from_pdf(pdf_path)
        text = result["full_text"]
        
        if not text:
            return {
                "file_name": str(pdf_path),
                "classification": "unknown",
                "confidence": 0.0,
                "error": "Text extraction failed"
            }

        try:
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                confidence = probabilities[0][prediction].item()

            label = "cancer" if prediction.item() == 1 else "non-cancer"
            
            return {
                "file_name": str(pdf_path),
                "classification": label,
                "confidence": round(confidence * 100, 2)
            }

        except Exception as e:
            self.logger.error(f"Error during classification: {e}")
            return {
                "file_name": str(pdf_path),
                "classification": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
