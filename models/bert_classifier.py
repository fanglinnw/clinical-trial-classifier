import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import logging
from typing import Dict, Union
from utils.text_extractor import get_extractor

class BERTClassifier:
    MODEL_PATHS = {
        'biobert': "dmis-lab/biobert-v1.1",
        'clinicalbert': "emilyalsentzer/Bio_ClinicalBERT",
        'pubmedbert': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    }

    def __init__(self, model_type: str, model_path: str = None, max_length: int = 512, extractor_type: str = "simple"):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of BERT model ('biobert', 'clinicalbert', or 'pubmedbert')
            model_path: Path to fine-tuned model. If None, uses the base model
            max_length: Maximum sequence length for the model
            extractor_type: Type of text extractor to use
        """
        if model_type not in self.MODEL_PATHS:
            raise ValueError(f"model_type must be one of {list(self.MODEL_PATHS.keys())}")

        self.logger = logging.getLogger(__name__)
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.extractor = get_extractor(extractor_type)
        
        # Load model and tokenizer
        model_source = model_path if model_path and Path(model_path).exists() else self.MODEL_PATHS[model_type]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                num_labels=2
            ).to(self.device).eval()
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _predict(self, text: str) -> Dict[str, Union[str, float]]:
        """Make prediction on a single text."""
        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probs, dim=1)
                confidence = probs[0][prediction].item()

            return {
                "classification": "cancer" if prediction.item() == 1 else "non_cancer",
                "confidence": round(confidence * 100, 2)
            }
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def classify_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Union[str, float]]:
        """Classify a single PDF."""
        result = self.extractor.extract_from_pdf(pdf_path)
        if not result["full_text"]:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "error": "Text extraction failed"
            }
        
        return self._predict(result["full_text"])

    def classify_pdfs(self, pdf_paths: list) -> list:
        """Classify multiple PDFs."""
        return [self.classify_pdf(path) for path in pdf_paths]