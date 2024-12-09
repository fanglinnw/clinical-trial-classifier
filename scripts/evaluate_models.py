import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Union, List, Tuple
from collections import Counter
import sys
import os
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from models.bert_classifier import BERTClassifier
from models.baseline_classifiers import BaselineClassifiers

class ProtocolClassifierEnsemble:
    def __init__(self,
                 trained_models_dir: str = "./trained_models",
                 max_length: int = 8000):
        """
        Initialize ensemble of classifiers.
        """
        self.logger = logging.getLogger(__name__)
        self.classifiers = {}
        
        # Initialize BERT models
        model_types = ['biobert', 'clinicalbert', 'pubmedbert']
        for model_type in model_types:
            model_path = os.path.join(trained_models_dir, model_type)
            try:
                self.classifiers[model_type] = BERTClassifier(
                    model_type=model_type,
                    model_path=model_path,
                    max_length=max_length
                )
                self.logger.info(f"Loaded {model_type} classifier")
            except Exception as e:
                self.logger.error(f"Failed to load {model_type} classifier: {e}")
        
        # Initialize baseline models
        try:
            baseline_path = os.path.join(trained_models_dir, "baseline")
            self.classifiers['baseline'] = BaselineClassifiers(model_dir=baseline_path)
            self.logger.info("Loaded baseline classifiers")
        except Exception as e:
            self.logger.error(f"Failed to load baseline classifiers: {e}")

    def classify_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Classify a PDF using all available classifiers.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with predictions from each classifier
        """
        results = {}
        for name, classifier in self.classifiers.items():
            try:
                results[name] = classifier.classify_pdf(pdf_path)
            except Exception as e:
                self.logger.error(f"Error with {name} classifier: {e}")
                results[name] = {
                    "file_name": str(pdf_path),
                    "classification": "unknown",
                    "confidence": 0.0,
                    "error": str(e)
                }
        return results

    def evaluate_directory(self, directory: Union[str, Path], expected_class: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all classifiers on a directory of PDFs.
        
        Args:
            directory: Path to directory containing PDFs
            expected_class: Expected class label ('cancer' or 'non_cancer')
            
        Returns:
            Dictionary with evaluation metrics for each classifier
        """
        directory = Path(directory)
        pdf_files = list(directory.glob('*.pdf'))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {directory}")
            return {}
        
        # Initialize results storage
        predictions = {name: [] for name in self.classifiers.keys()}
        expected = []
        
        # Process BERT models in batches
        for name, classifier in self.classifiers.items():
            self.logger.info(f"\nEvaluating {name} classifier on {directory}")
            if isinstance(classifier, BERTClassifier):
                batch_results = classifier.classify_pdfs_batch(pdf_files)
                for result in batch_results:
                    predictions[name].append(1 if result['classification'] == 'cancer' else 0)
            else:
                # Process other classifiers normally with progress bar
                for pdf_path in tqdm(pdf_files, desc=f"Processing files", unit="file"):
                    result = classifier.classify_pdf(pdf_path)
                    predictions[name].append(1 if result['classification'] == 'cancer' else 0)
        
        # Set expected values
        expected = [1 if expected_class == 'cancer' else 0] * len(pdf_files)
        
        # Calculate metrics for each classifier
        metrics = {}
        for name, preds in predictions.items():
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    expected, preds, average='binary', zero_division=0
                )
                accuracy = accuracy_score(expected, preds)
                
                metrics[name] = {
                    'accuracy': round(accuracy * 100, 2),
                    'precision': round(precision * 100, 2),
                    'recall': round(recall * 100, 2),
                    'f1': round(f1 * 100, 2)
                }
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {name}: {e}")
                metrics[name] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'error': str(e)
                }
        
        return metrics

def format_summary(summary: Dict) -> str:
    """Format evaluation summary for display."""
    headers = ['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1']
    rows = []
    
    for classifier, metrics in summary.items():
        if 'error' not in metrics:
            rows.append([
                classifier,
                f"{metrics['accuracy']}%",
                f"{metrics['precision']}%",
                f"{metrics['recall']}%",
                f"{metrics['f1']}%"
            ])
    
    return tabulate(rows, headers=headers, tablefmt='grid')

def main():
    parser = argparse.ArgumentParser(description='Evaluate protocol classifiers')
    parser.add_argument('--cancer-dir', required=True,
                        help='Directory containing cancer protocol PDFs')
    parser.add_argument('--non-cancer-dir', required=True,
                        help='Directory containing non-cancer protocol PDFs')
    parser.add_argument('--models-dir', default='./trained_models',
                        help='Directory containing trained models')
    parser.add_argument('--max-length', type=int, default=8000,
                        help='Maximum text length to process')
    parser.add_argument('--model', choices=['biobert', 'clinicalbert', 'pubmedbert', 'baseline'],
                        help='Specify a single model to evaluate. If not provided, all models will be evaluated.')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize ensemble
    ensemble = ProtocolClassifierEnsemble(
        trained_models_dir=args.models_dir,
        max_length=args.max_length
    )

    # If a specific model is selected, keep only that model
    if args.model:
        if args.model in ensemble.classifiers:
            selected_classifier = ensemble.classifiers[args.model]
            ensemble.classifiers.clear()
            ensemble.classifiers[args.model] = selected_classifier
            logger.info(f"Evaluating only the {args.model} model")
        else:
            logger.error(f"Model {args.model} not found or failed to load")
            return

    # Evaluate on cancer and non-cancer directories
    logger.info("\nEvaluating cancer protocols...")
    cancer_metrics = ensemble.evaluate_directory(args.cancer_dir, 'cancer')
    
    logger.info("\nEvaluating non-cancer protocols...")
    non_cancer_metrics = ensemble.evaluate_directory(args.non_cancer_dir, 'non_cancer')

    # Calculate and display average metrics
    summary = {}
    for name in ensemble.classifiers.keys():
        if name in cancer_metrics and name in non_cancer_metrics:
            summary[name] = {
                metric: round((cancer_metrics[name][metric] + non_cancer_metrics[name][metric]) / 2, 2)
                for metric in ['accuracy', 'precision', 'recall', 'f1']
            }

    # Display results
    logger.info("\nEvaluation Results (Averaged across cancer and non-cancer):")
    print(format_summary(summary))

    # Save results
    results = {
        'cancer': cancer_metrics,
        'non_cancer': non_cancer_metrics,  # Keep underscore here for backward compatibility with existing results files
        'average': summary
    }
    
    results_file = os.path.join(args.models_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()