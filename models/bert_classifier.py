import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import logging
import platform
from typing import Dict, Union
from tqdm import tqdm
from utils.text_extractor import get_extractor

class BERTClassifier:
    MODEL_PATHS = {
        'biobert': "dmis-lab/biobert-v1.1",
        'clinicalbert': "emilyalsentzer/Bio_ClinicalBERT",
        'pubmedbert': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    }

    def __init__(self, model_type: str, model_path: str = None, max_length: int = 8000, extractor_type: str = "simple"):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of BERT model ('biobert', 'clinicalbert', or 'pubmedbert')
            model_path: Path to fine-tuned model. If None, uses the base model.
            max_length: Maximum length for text extraction
            extractor_type: Type of text extractor to use ('simple' or 'section')
        """
        if model_type not in self.MODEL_PATHS:
            raise ValueError(f"model_type must be one of {list(self.MODEL_PATHS.keys())}")

        # Device setup for M1 Mac
        if platform.processor() == 'arm':
            device_name = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device_name)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {device_name}")
        
        # Initialize text extractor
        self.extractor = get_extractor(extractor_type)
        
        try:
            if model_path and Path(model_path).exists():
                # Load fine-tuned model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                # Load base model
                base_model = self.MODEL_PATHS[model_type]
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    base_model,
                    num_labels=2
                )
            
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
            return self.classify_text(text)
        except Exception as e:
            self.logger.error(f"Error during classification: {e}")
            return {
                "file_name": str(pdf_path),
                "classification": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def classify_text(self, text: str) -> Dict[str, Union[str, float]]:
        """Classify text directly without PDF extraction."""
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

            label = "cancer" if prediction.item() == 1 else "non_cancer"
            
            return {
                "classification": label,
                "confidence": round(confidence * 100, 2)
            }

        except Exception as e:
            self.logger.error(f"Error during text classification: {e}")
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def classify_pdfs_batch(self, pdf_paths, batch_size=8):
        """Classify multiple PDFs in batches for better GPU utilization."""
        results = []
        texts = []
        valid_paths = []

        # First extract all texts
        for pdf_path in tqdm(pdf_paths, desc="Extracting text from PDFs", unit="file"):
            result = self.extractor.extract_from_pdf(pdf_path)
            text = result["full_text"]
            if text:
                texts.append(text)
                valid_paths.append(pdf_path)
            else:
                results.append({
                    "file_name": str(pdf_path),
                    "classification": "unknown",
                    "confidence": 0.0,
                    "error": "Text extraction failed"
                })

        if not texts:
            return results

        # Process in batches with progress bar
        num_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches", total=num_batches, unit="batch"):
            batch_texts = texts[i:i + batch_size]
            batch_paths = valid_paths[i:i + batch_size]
            
            try:
                # Tokenize the batch
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    confidences = probabilities[range(len(predictions)), predictions]

                # Process results
                for path, pred, conf in zip(batch_paths, predictions, confidences):
                    label = "cancer" if pred.item() == 1 else "non_cancer"
                    results.append({
                        "file_name": str(path),
                        "classification": label,
                        "confidence": round(conf.item() * 100, 2)
                    })

            except Exception as e:
                self.logger.error(f"Error during batch classification: {e}")
                for path in batch_paths:
                    results.append({
                        "file_name": str(path),
                        "classification": "unknown",
                        "confidence": 0.0,
                        "error": str(e)
                    })

        return results
