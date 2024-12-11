import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Union, List, Tuple

from utils.text_extractor import get_extractor
from models.bert_classifier import BERTClassifier
from models.baseline_classifiers import BaselineClassifiers

class ProtocolEvaluator:
    def __init__(self, models_dir: str, model_type: str = None, max_length: int = 8000, extractor_type: str = "simple"):
        """
        Initialize evaluator with option for single model evaluation.
        
        Args:
            models_dir: Directory containing trained models
            model_type: Optional - specific model to evaluate ('biobert', 'clinicalbert', 'pubmedbert', or 'baseline')
            max_length: Maximum sequence length
            extractor_type: Type of text extractor
        """
        self.logger = self._setup_logging()
        self.models_dir = models_dir
        self.model_type = model_type
        self.classifiers = self._load_classifiers(max_length, extractor_type)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_classifiers(self, max_length: int, extractor_type: str) -> Dict:
        """Load all classifiers or just the specified one."""
        classifiers = {}
        
        # Define available BERT models
        bert_models = ['biobert', 'clinicalbert', 'pubmedbert']
        
        if self.model_type:
            # Load only the specified model
            if self.model_type in bert_models:
                try:
                    classifiers[self.model_type] = BERTClassifier(
                        model_type=self.model_type,
                        model_path=f"{self.models_dir}",  # For single model, models_dir is the specific model path
                        max_length=max_length,
                        extractor_type=extractor_type
                    )
                except Exception as e:
                    self.logger.error(f"Failed to load {self.model_type}: {e}")
            elif self.model_type == 'baseline':
                try:
                    classifiers['baseline'] = BaselineClassifiers(
                        model_dir=f"{self.models_dir}",
                        max_length=max_length,
                        extractor_type=extractor_type
                    )
                except Exception as e:
                    self.logger.error(f"Failed to load baseline models: {e}")
        else:
            # Load all available models
            for model_type in bert_models:
                try:
                    classifiers[model_type] = BERTClassifier(
                        model_type=model_type,
                        model_path=f"{self.models_dir}/{model_type}",
                        max_length=max_length,
                        extractor_type=extractor_type
                    )
                except Exception as e:
                    self.logger.error(f"Failed to load {model_type}: {e}")

            # Load baseline models
            try:
                classifiers['baseline'] = BaselineClassifiers(
                    model_dir=f"{self.models_dir}/baseline",
                    max_length=max_length,
                    extractor_type=extractor_type
                )
            except Exception as e:
                self.logger.error(f"Failed to load baseline models: {e}")

        return classifiers

    def evaluate_directory(self, directory: Path, is_cancer: bool) -> List[Dict]:
        """Evaluate all models on a directory of PDFs."""
        results = []
        pdf_files = list(directory.glob('*.pdf'))
        
        for pdf_path in tqdm(pdf_files, desc=f"Evaluating {'cancer' if is_cancer else 'non-cancer'} protocols"):
            file_result = {
                'file_path': str(pdf_path),
                'true_label': 1 if is_cancer else 0
            }
            
            for name, classifier in self.classifiers.items():
                try:
                    if isinstance(classifier, BERTClassifier):
                        result = classifier.classify_pdf(pdf_path)
                        file_result.update({
                            f'{name}_prediction': 1 if result['classification'].lower() == 'cancer' else 0,
                            f'{name}_confidence': result['confidence']
                        })
                    else:  # Baseline classifier
                        result = classifier.classify_pdf(pdf_path)
                        file_result.update({
                            f'logreg_prediction': 1 if result['traditional_ml']['log_reg_prediction'].lower() == 'cancer' else 0,
                            f'logreg_confidence': result['traditional_ml']['log_reg_confidence'],
                            f'svm_prediction': 1 if result['traditional_ml']['svm_prediction'].lower() == 'cancer' else 0,
                            f'svm_confidence': result['traditional_ml']['svm_confidence'],
                            f'zeroshot_prediction': 1 if result['zero_shot']['prediction'].lower() == 'cancer' else 0,
                            f'zeroshot_confidence': result['zero_shot']['confidence']
                        })
                except Exception as e:
                    self.logger.error(f"Error processing {pdf_path} with {name}: {e}")
                    file_result[f'{name}_error'] = str(e)
            
            results.append(file_result)
        
        return results

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics for all models from evaluation results."""
        metrics = {}
        true_labels = [r['true_label'] for r in results]
        
        # Calculate metrics for BERT models
        for model in self.classifiers.keys():
            if model != 'baseline':
                pred_key = f'{model}_prediction'
                if pred_key in results[0]:
                    predictions = [r[pred_key] for r in results]
                    metrics[model] = self._get_metrics(true_labels, predictions)
        
        # Calculate metrics for baseline models if present
        if 'baseline' in self.classifiers:
            for model in ['logreg', 'svm', 'zeroshot']:
                pred_key = f'{model}_prediction'
                if pred_key in results[0]:
                    predictions = [r[pred_key] for r in results]
                    metrics[f'baseline_{model}'] = self._get_metrics(true_labels, predictions)
        
        return metrics

    def _get_metrics(self, true_labels: List[int], predictions: List[int]) -> Dict:
        """Calculate standard classification metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        accuracy = accuracy_score(true_labels, predictions)
        
        cm = confusion_matrix(true_labels, predictions)
        self.logger.info("\nConfusion Matrix:")
        self.logger.info(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        self.logger.info(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return {
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2)
        }