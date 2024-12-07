import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from pathlib import Path
import fitz
import joblib
import numpy as np
import logging
from tqdm import tqdm
import json
import platform
import argparse


class BaselineClassifiers:
    def __init__(self, model_dir: str = "./baseline_models"):
        """
        Initialize baseline classifiers
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Setup for M1 Mac
        if platform.processor() == 'arm':
            device_name = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device_name
        
        # Initialize zero-shot classifier
        self.logger.info("Loading zero-shot classifier...")
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        
        # Traditional ML models will be initialized during training
        self.tfidf = None
        self.log_reg = None
        self.svm = None

    def extract_text_from_pdf(self, pdf_path: str, max_length: int = 4000) -> str:
        """Extract text from a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            
            if len(text) > max_length:
                text = ' '.join(text[:max_length].split()[:-1])
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return ""
        finally:
            if 'doc' in locals():
                doc.close()

    def train_traditional_models(self, train_dir: str):
        """Train TF-IDF + LogisticRegression and SVM models"""
        self.logger.info("Training traditional ML models...")
        
        # Prepare training data
        texts = []
        labels = []
        
        # Process cancer protocols
        cancer_dir = Path(train_dir) / 'cancer' / 'train'
        for pdf_path in tqdm(list(cancer_dir.glob('*.pdf')), desc="Processing cancer protocols"):
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                texts.append(text)
                labels.append(1)
        
        # Process non-cancer protocols
        non_cancer_dir = Path(train_dir) / 'non_cancer' / 'train'
        for pdf_path in tqdm(list(non_cancer_dir.glob('*.pdf')), desc="Processing non-cancer protocols"):
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                texts.append(text)
                labels.append(0)
        
        # Train TF-IDF
        self.logger.info("Training TF-IDF vectorizer...")
        self.tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X = self.tfidf.fit_transform(texts)
        
        # Train Logistic Regression
        self.logger.info("Training Logistic Regression...")
        self.log_reg = LogisticRegression(max_iter=1000)
        self.log_reg.fit(X, labels)
        
        # Train SVM
        self.logger.info("Training SVM...")
        self.svm = LinearSVC(max_iter=1000)
        self.svm.fit(X, labels)
        
        # Save models
        joblib.dump(self.tfidf, self.model_dir / 'tfidf.joblib')
        joblib.dump(self.log_reg, self.model_dir / 'logistic_regression.joblib')
        joblib.dump(self.svm, self.model_dir / 'svm.joblib')
        
        self.logger.info("Traditional models trained and saved")

    def load_traditional_models(self):
        """Load trained traditional ML models"""
        self.tfidf = joblib.load(self.model_dir / 'tfidf.joblib')
        self.log_reg = joblib.load(self.model_dir / 'logistic_regression.joblib')
        self.svm = joblib.load(self.model_dir / 'svm.joblib')

    def classify_text_traditional(self, text: str) -> dict:
        """Classify text using traditional ML models"""
        if not text:
            return {
                "log_reg_prediction": "unknown",
                "log_reg_confidence": 0.0,
                "svm_prediction": "unknown",
                "svm_confidence": 0.0
            }
        
        # Transform text
        X = self.tfidf.transform([text])
        
        # Logistic Regression prediction
        log_reg_prob = self.log_reg.predict_proba(X)[0]
        log_reg_pred = "cancer" if log_reg_prob[1] > 0.5 else "non-cancer"
        
        # SVM prediction
        svm_decision = self.svm.decision_function(X)[0]
        svm_prob = 1 / (1 + np.exp(-svm_decision))  # Convert to probability
        svm_pred = "cancer" if svm_prob > 0.5 else "non-cancer"
        
        return {
            "log_reg_prediction": log_reg_pred,
            "log_reg_confidence": round(max(log_reg_prob) * 100, 2),
            "svm_prediction": svm_pred,
            "svm_confidence": round(max(svm_prob, 1-svm_prob) * 100, 2)
        }

    def classify_text_zero_shot(self, text: str) -> dict:
        """Classify text using zero-shot classification"""
        if not text:
            return {
                "prediction": "unknown",
                "confidence": 0.0
            }
        
        # Define candidate labels
        candidate_labels = ["cancer clinical trial", "non-cancer clinical trial"]
        
        # Get prediction
        result = self.zero_shot(text, candidate_labels, multi_label=False)
        
        # Process results
        prediction = "cancer" if result['labels'][0] == "cancer clinical trial" else "non-cancer"
        confidence = round(result['scores'][0] * 100, 2)
        
        return {
            "prediction": prediction,
            "confidence": confidence
        }

    def classify_pdf(self, pdf_path: str) -> dict:
        """Classify a single PDF using all methods"""
        text = self.extract_text_from_pdf(pdf_path)
        
        # Get predictions from all models
        traditional_results = self.classify_text_traditional(text)
        zero_shot_results = self.classify_text_zero_shot(text)
        
        return {
            "file_name": pdf_path,
            "traditional_ml": traditional_results,
            "zero_shot": zero_shot_results
        }

    def batch_classify_pdfs(self, pdf_dir: str) -> list:
        """Classify all PDFs in a directory"""
        pdf_dir = Path(pdf_dir)
        results = []
        
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        for pdf_path in tqdm(pdf_files, desc="Classifying protocols"):
            result = self.classify_pdf(str(pdf_path))
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description='Baseline Clinical Trial Protocol Classifiers')
    parser.add_argument('--train', action='store_true', help='Train traditional ML models')
    parser.add_argument('--train-dir', default='./protocol_documents', 
                      help='Directory containing training data')
    parser.add_argument('--input', help='Path to PDF file or directory for classification')
    parser.add_argument('--output', help='Path to save results JSON file')
    args = parser.parse_args()

    classifiers = BaselineClassifiers()

    if args.train:
        classifiers.train_traditional_models(args.train_dir)
    else:
        classifiers.load_traditional_models()

    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            result = classifiers.classify_pdf(str(input_path))
            results = [result]
            print(f"\nResults for {input_path}:")
            print("\nTraditional ML Results:")
            print(f"LogReg: {result['traditional_ml']['log_reg_prediction']} "
                  f"({result['traditional_ml']['log_reg_confidence']}%)")
            print(f"SVM: {result['traditional_ml']['svm_prediction']} "
                  f"({result['traditional_ml']['svm_confidence']}%)")
            print("\nZero-shot Results:")
            print(f"Prediction: {result['zero_shot']['prediction']} "
                  f"({result['zero_shot']['confidence']}%)")
        else:
            results = classifiers.batch_classify_pdfs(str(input_path))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
