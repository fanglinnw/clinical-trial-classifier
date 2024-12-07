from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from transformers import pipeline
import numpy as np
import platform
import torch
import joblib
import logging
from pathlib import Path
from typing import Dict, Union, List
from utils.text_extractor import ProtocolTextExtractor


class BaselineClassifiers:
    def __init__(self, model_dir: str = "./baseline_models", max_length: int = 8000):
        """Initialize baseline classifiers."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Setup for M1 Mac
        if platform.processor() == 'arm':
            device_name = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device_name
        
        # Initialize text extractor
        self.extractor = ProtocolTextExtractor(max_length=max_length)
        
        # Initialize zero-shot classifier
        self.logger.info("Loading zero-shot classifier...")
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="FacebookAI/roberta-large-mnli",
            device=self.device
        )
        
        # Traditional ML models will be initialized during training
        self.tfidf = None
        self.log_reg = None
        self.svm = None

    def train_traditional_models(self, train_dir: str):
        """Train TF-IDF + LogisticRegression and SVM models."""
        self.logger.info("Training traditional ML models...")
        
        texts = []
        labels = []
        
        # Process cancer protocols
        cancer_dir = Path(train_dir) / 'cancer' / 'train'
        for pdf_path in cancer_dir.glob('*.pdf'):
            result = self.extractor.extract_from_pdf(pdf_path)
            if result["full_text"]:
                texts.append(result["full_text"])
                labels.append(1)
        
        # Process non-cancer protocols
        non_cancer_dir = Path(train_dir) / 'non_cancer' / 'train'
        for pdf_path in non_cancer_dir.glob('*.pdf'):
            result = self.extractor.extract_from_pdf(pdf_path)
            if result["full_text"]:
                texts.append(result["full_text"])
                labels.append(0)
        
        # Train models
        self.tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X = self.tfidf.fit_transform(texts)
        
        self.log_reg = LogisticRegression(max_iter=1000)
        self.log_reg.fit(X, labels)
        
        self.svm = LinearSVC(max_iter=1000)
        self.svm.fit(X, labels)
        
        # Save models
        joblib.dump(self.tfidf, self.model_dir / 'tfidf.joblib')
        joblib.dump(self.log_reg, self.model_dir / 'logistic_regression.joblib')
        joblib.dump(self.svm, self.model_dir / 'svm.joblib')

    def load_traditional_models(self):
        """Load trained traditional ML models."""
        self.tfidf = joblib.load(self.model_dir / 'tfidf.joblib')
        self.log_reg = joblib.load(self.model_dir / 'logistic_regression.joblib')
        self.svm = joblib.load(self.model_dir / 'svm.joblib')

    def classify_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Dict[str, Union[str, float]]]:
        """Classify a PDF using all methods."""
        # Extract text
        result = self.extractor.extract_from_pdf(pdf_path)
        text = result["full_text"]
        
        if not text:
            return {
                "file_name": str(pdf_path),
                "error": "Text extraction failed"
            }
        
        # Get predictions from all models
        traditional_results = self._classify_traditional(text)
        zero_shot_results = self._classify_zero_shot(text)
        
        return {
            "file_name": str(pdf_path),
            "traditional_ml": traditional_results,
            "zero_shot": zero_shot_results
        }

    def _classify_traditional(self, text: str) -> Dict[str, Union[str, float]]:
        """Get predictions from traditional ML models."""
        X = self.tfidf.transform([text])
        
        log_reg_prob = self.log_reg.predict_proba(X)[0]
        log_reg_pred = "cancer" if log_reg_prob[1] > 0.5 else "non-cancer"
        
        svm_decision = self.svm.decision_function(X)[0]
        svm_prob = 1 / (1 + np.exp(-svm_decision))
        svm_pred = "cancer" if svm_prob > 0.5 else "non-cancer"
        
        return {
            "log_reg_prediction": log_reg_pred,
            "log_reg_confidence": round(max(log_reg_prob) * 100, 2),
            "svm_prediction": svm_pred,
            "svm_confidence": round(max(svm_prob, 1-svm_prob) * 100, 2)
        }

    def _classify_zero_shot(self, text: str) -> Dict[str, Union[str, float]]:
        """Get prediction from zero-shot classifier."""
        # candidate_labels = ["cancer clinical trial", "non-cancer clinical trial"]
        # candidate_labels = ["cancer research protocol", "non-cancer clinical trial protocol"]
        result = self.zero_shot(text, candidate_labels=["cancer research protocol",
                                          "non-cancer clinical trial protocol"],
                        hypothesis_template="This document describes a {}.")
        
        prediction = "cancer" if result['labels'][0] == "cancer research protocol" else "non-cancer"
        confidence = round(result['scores'][0] * 100, 2)
        
        return {
            "prediction": prediction,
            "confidence": confidence
        }
