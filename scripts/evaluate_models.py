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
import pandas as pd

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
        self.models_dir = trained_models_dir
        
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

    def evaluate_directory(self, directory: Union[str, Path], expected_class: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[int]]]:
        """
        Evaluate all classifiers on a directory of PDFs.
        
        Args:
            directory: Path to directory containing PDFs
            expected_class: Expected class label ('cancer' or 'non_cancer')
            
        Returns:
            Dictionary with evaluation metrics for each classifier and a dictionary of predictions
        """
        directory = Path(directory)
        pdf_files = list(directory.glob('*.pdf'))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {directory}")
            return {}, {}
        
        # Initialize results storage
        predictions = {name: [] for name in self.classifiers.keys()}
        baseline_predictions = {
            'baseline_log_reg': [],
            'baseline_svm': [],
            'baseline_zero_shot': []
        }
        detailed_results = []  # Store detailed results for CSV
        
        # Set expected values (1 for cancer, 0 for non_cancer)
        expected = [1 if expected_class == 'cancer' else 0] * len(pdf_files)
        
        # Create progress bars
        self.logger.info(f"\nEvaluating {expected_class} protocols...")
        main_pbar = tqdm(total=len(pdf_files), desc=f"Overall progress", position=0)
        model_pbar = tqdm(total=len(self.classifiers), desc=f"Current file progress", position=1, leave=False)
        
        try:
            # Process each file
            for i, pdf_path in enumerate(pdf_files):
                file_result = {
                    'file_path': str(pdf_path),
                    'expected_class': expected_class,
                    'expected_label': expected[i]
                }
                
                # Reset model progress bar for each file
                model_pbar.reset()
                model_pbar.set_description(f"Processing {pdf_path.name}")
                
                # Get predictions from each classifier
                for name, classifier in self.classifiers.items():
                    try:
                        if isinstance(classifier, BERTClassifier):
                            result = classifier.classify_pdf(pdf_path)
                            pred_class = result.get('classification', '').lower()
                            confidence = result.get('confidence', 0.0)
                            pred = 1 if pred_class == 'cancer' else 0
                            predictions[name].append(pred)
                            
                            # Store detailed results
                            file_result[f'{name}_prediction'] = pred_class
                            file_result[f'{name}_confidence'] = confidence
                            file_result[f'{name}_numeric'] = pred
                        else:  # Baseline classifier
                            result = classifier.classify_pdf(pdf_path)
                            # Check both logistic regression and SVM predictions
                            log_reg_pred = result.get('traditional_ml', {}).get('log_reg_prediction', '').lower()
                            svm_pred = result.get('traditional_ml', {}).get('svm_prediction', '').lower()
                            log_reg_conf = result.get('traditional_ml', {}).get('log_reg_confidence', 0.0)
                            svm_conf = result.get('traditional_ml', {}).get('svm_confidence', 0.0)
                            
                            # Store predictions separately for each baseline approach
                            baseline_predictions['baseline_log_reg'].append(1 if log_reg_pred == 'cancer' else 0)
                            baseline_predictions['baseline_svm'].append(1 if svm_pred == 'cancer' else 0)
                            
                            # Get zero-shot prediction
                            zero_shot_result = result.get('zero_shot', {})
                            zero_shot_pred = zero_shot_result.get('prediction', '').lower()
                            zero_shot_conf = zero_shot_result.get('confidence', 0.0)
                            baseline_predictions['baseline_zero_shot'].append(1 if zero_shot_pred == 'cancer' else 0)
                            
                            # Use logistic regression as primary prediction
                            pred = 1 if log_reg_pred == 'cancer' else 0
                            predictions[name].append(pred)
                            
                            # Store detailed results
                            file_result[f'{name}_log_reg_prediction'] = log_reg_pred
                            file_result[f'{name}_log_reg_confidence'] = log_reg_conf
                            file_result[f'{name}_svm_prediction'] = svm_pred
                            file_result[f'{name}_svm_confidence'] = svm_conf
                            file_result[f'{name}_zero_shot_prediction'] = zero_shot_pred
                            file_result[f'{name}_zero_shot_confidence'] = zero_shot_conf
                            file_result[f'{name}_numeric'] = pred
                    except Exception as e:
                        self.logger.error(f"Error processing {pdf_path} with {name}: {e}")
                        # Append a default prediction in case of error
                        predictions[name].append(0)  # Default to non-cancer on error
                        file_result[f'{name}_error'] = str(e)
                    
                    model_pbar.update(1)
                
                detailed_results.append(file_result)
                main_pbar.update(1)
            
        finally:
            # Close progress bars
            model_pbar.close()
            main_pbar.close()
        
        # Save detailed results to CSV
        csv_filename = f"evaluation_results_{expected_class}.csv"
        csv_path = os.path.join(self.models_dir, csv_filename)
        
        df = pd.DataFrame(detailed_results)
        df.to_csv(csv_path, index=False)
        self.logger.info(f"\nDetailed results saved to: {csv_path}")
        
        # Calculate metrics for each classifier
        metrics = {}
        for name, preds in predictions.items():
            try:
                if not preds:
                    raise ValueError(f"No predictions found for {name}")
                
                # Use binary average for metrics since we have binary classification
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
                
                # Print confusion matrix for debugging
                cm = confusion_matrix(expected, preds)
                self.logger.info(f"\nConfusion Matrix for {name}:")
                self.logger.info("True Negative  False Positive")
                self.logger.info("False Negative True Positive")
                self.logger.info(f"{cm[0][0]:^13} {cm[0][1]:^13}")
                self.logger.info(f"{cm[1][0]:^13} {cm[1][1]:^13}")
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {name}: {e}")
                metrics[name] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'error': str(e)
                }
        
        # Calculate and display metrics for baseline approaches
        if 'baseline' in predictions:
            self.logger.info("\nDetailed Baseline Model Results:")
            for approach, preds in baseline_predictions.items():
                if preds:  # Only calculate if we have predictions
                    try:
                        # Force labels to include both classes even if not present
                        cm = confusion_matrix(expected, preds, labels=[0, 1])
                        self.logger.info(f"\nConfusion Matrix for {approach}:")
                        self.logger.info("True Negative  False Positive")
                        self.logger.info("False Negative True Positive")
                        self.logger.info(f"{cm[0][0]:^13} {cm[0][1]:^13}")
                        self.logger.info(f"{cm[1][0]:^13} {cm[1][1]:^13}")
                        
                        # Calculate metrics with zero_division=0
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            expected, preds, average='binary', zero_division=0, labels=[0, 1]
                        )
                        accuracy = accuracy_score(expected, preds)
                        
                        self.logger.info(f"\nMetrics for {approach}:")
                        self.logger.info(f"Accuracy: {accuracy * 100:.2f}%")
                        self.logger.info(f"Precision: {precision * 100:.2f}%")
                        self.logger.info(f"Recall: {recall * 100:.2f}%")
                        self.logger.info(f"F1: {f1 * 100:.2f}%")
                    except Exception as e:
                        self.logger.error(f"Error calculating metrics for {approach}: {e}")
        
        return metrics, predictions

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
    cancer_metrics, cancer_predictions = ensemble.evaluate_directory(args.cancer_dir, 'cancer')
    
    logger.info("\nEvaluating non-cancer protocols...")
    non_cancer_metrics, non_cancer_predictions = ensemble.evaluate_directory(args.non_cancer_dir, 'non_cancer')

    # Calculate combined metrics across both datasets
    summary = {}
    for name in ensemble.classifiers.keys():
        if name in cancer_metrics and name in non_cancer_metrics:
            # Combine predictions and true labels from both evaluations
            all_preds = cancer_predictions[name] + non_cancer_predictions[name]
            all_labels = [1] * len(cancer_predictions[name]) + [0] * len(non_cancer_predictions[name])
            
            # Calculate metrics on combined data
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0
            )
            accuracy = accuracy_score(all_labels, all_preds)
            
            summary[name] = {
                'accuracy': round(accuracy * 100, 2),
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1': round(f1 * 100, 2)
            }

    # Display results
    logger.info("\nEvaluation Results (Combined metrics across all data):")
    print(format_summary(summary))

    # Save results
    results = {
        'cancer': cancer_metrics,
        'non_cancer': non_cancer_metrics,
        'combined': summary
    }
    
    results_file = os.path.join(args.models_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()