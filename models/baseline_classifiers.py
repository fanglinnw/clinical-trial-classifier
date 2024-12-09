from typing import Dict, Union, List
import platform
import torch
import joblib
import logging
from pathlib import Path
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import pipeline, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from utils.text_extractor import ProtocolTextExtractor


class BaselineClassifiers:
    def __init__(self, model_dir: str = "./trained_models/baseline", max_length: int = 8000):
        """Initialize baseline classifiers."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
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
        
        # Initialize zero-shot classifier and tokenizer
        self.logger.info("Loading zero-shot classifier...")
        model_name = "FacebookAI/roberta-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model=model_name,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=32
        )
        
        # Initialize traditional ML models
        self.tfidf = None
        self.log_reg = None
        self.svm = None
        
        # Try to load trained models if they exist
        try:
            self.load_traditional_models()
            self.logger.info("Successfully loaded traditional ML models")
        except Exception as e:
            self.logger.warning(f"Could not load traditional ML models: {e}")

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

    def train_with_data(self, texts: List[str], labels: List[int]):
        """Train models with pre-loaded data."""
        self.logger.info("Training traditional ML models...")
        
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
        tfidf_path = self.model_dir / 'tfidf.joblib'
        log_reg_path = self.model_dir / 'logistic_regression.joblib'
        svm_path = self.model_dir / 'svm.joblib'
        
        if not all(path.exists() for path in [tfidf_path, log_reg_path, svm_path]):
            raise FileNotFoundError("One or more model files not found. Please train the models first.")
            
        self.tfidf = joblib.load(tfidf_path)
        self.log_reg = joblib.load(log_reg_path)
        self.svm = joblib.load(svm_path)

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

    def classify_text(self, text: str) -> Dict[str, Dict[str, Union[str, float]]]:
        """Classify text directly using all methods."""
        if not text:
            return {
                "error": "Empty text"
            }
        
        # Get predictions from all models
        traditional_results = self._classify_traditional(text)
        zero_shot_results = self._classify_zero_shot(text)
        
        return {
            "traditional_ml": traditional_results,
            "zero_shot": zero_shot_results
        }

    def classify_text_batch(self, texts: List[str]) -> List[Dict[str, Dict[str, Union[str, float]]]]:
        """Classify a batch of texts using all methods."""
        if not texts:
            return []
        
        results = []
        
        # Get traditional ML predictions
        X = self.tfidf.transform(texts)
        log_reg_probs = self.log_reg.predict_proba(X)
        svm_decisions = self.svm.decision_function(X)
        
        # Create dataset for zero-shot classification
        from datasets import Dataset, disable_progress_bar
        
        # Disable progress bar for dataset operations
        disable_progress_bar()
        
        # Create dataset with the texts
        dataset = Dataset.from_dict({"text": texts})
        
        # Define the classification function
        def classify_batch(examples):
            outputs = self.zero_shot(
                examples["text"],
                candidate_labels=["cancer research protocol", "non_cancer clinical trial protocol"],
                hypothesis_template="This document is a {}.",
                batch_size=32
            )
            return {"zero_shot_output": outputs}
        
        # Apply the classification function to the dataset
        dataset = dataset.map(
            classify_batch,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
            desc=None  # Disable the description to hide progress bar
        )
        
        # Process results for each text
        zero_shot_results = dataset["zero_shot_output"]
        
        for i, text in enumerate(texts):
            # Process logistic regression results
            log_reg_prob = log_reg_probs[i]
            log_reg_pred = "cancer" if log_reg_prob[1] > 0.5 else "non_cancer"
            log_reg_conf = max(log_reg_prob) * 100
            
            # Process SVM results
            svm_decision = svm_decisions[i]
            svm_pred = "cancer" if svm_decision > 0 else "non_cancer"
            svm_conf = (1 / (1 + np.exp(-svm_decision))) * 100
            
            # Process zero-shot results
            try:
                zero_shot_result = zero_shot_results[i]
                prediction = "cancer" if zero_shot_result['labels'][0] == "cancer research protocol" else "non_cancer"
                zero_shot_conf = max(zero_shot_result['scores']) * 100
            except (IndexError, KeyError):
                prediction = "unknown"
                zero_shot_conf = 0.0
            
            results.append({
                "traditional_ml": {
                    "log_reg_prediction": log_reg_pred,
                    "log_reg_confidence": round(log_reg_conf, 2),
                    "svm_prediction": svm_pred,
                    "svm_confidence": round(svm_conf, 2)
                },
                "zero_shot": {
                    "prediction": prediction,
                    "confidence": round(zero_shot_conf, 2)
                }
            })
        
        return results

    def _classify_traditional(self, text: str) -> Dict[str, Union[str, float]]:
        """Get predictions from traditional ML models."""
        X = self.tfidf.transform([text])
        
        # Logistic Regression prediction
        log_reg_prob = self.log_reg.predict_proba(X)[0]
        log_reg_pred = "cancer" if log_reg_prob[1] > 0.5 else "non_cancer"
        log_reg_conf = max(log_reg_prob) * 100
        
        # SVM prediction using decision_function
        svm_decision = self.svm.decision_function(X)[0]
        svm_pred = "cancer" if svm_decision > 0 else "non_cancer"
        # Convert decision function to probability-like score
        svm_conf = (1 / (1 + np.exp(-svm_decision))) * 100
        
        return {
            "log_reg_prediction": log_reg_pred,
            "log_reg_confidence": round(log_reg_conf, 2),
            "svm_prediction": svm_pred,
            "svm_confidence": round(svm_conf, 2)
        }

    def _classify_zero_shot(self, text: str) -> Dict[str, Union[str, float]]:
        """Get prediction from zero-shot classifier."""
        try:
            result = self.zero_shot(text, candidate_labels=["cancer research protocol",
                                          "non_cancer clinical trial protocol"],
                        hypothesis_template="This document is a {}.")
            prediction = "cancer" if result['labels'][0] == "cancer research protocol" else "non_cancer"
            zero_shot_conf = max(result['scores']) * 100
        except Exception as e:
            self.logger.error(f"Zero-shot classification failed: {e}")
            prediction = "unknown"
            zero_shot_conf = 0.0
        
        return {
            "prediction": prediction,
            "confidence": round(zero_shot_conf, 2)
        }
