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
from datetime import datetime

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.text_extractor import get_extractor
from models.bert_classifier import BERTClassifier
from models.baseline_classifiers import BaselineClassifiers

class ProtocolClassifierEnsemble:
    def __init__(self,
                 trained_models_dir: str = "./trained_models",
                 max_length: int = 8000,
                 extractor_type: str = "simple"):
        """
        Initialize ensemble of classifiers.
        
        Args:
            trained_models_dir: Directory containing trained models
            max_length: Maximum length of input text
            extractor_type: Type of text extractor to use ('simple' or 'sections')
        """
        self.logger = logging.getLogger(__name__)
        self.classifiers = {}
        self.models_dir = trained_models_dir
        self.detailed_results = []  # Store detailed results for CSV
        self.baseline_predictions = {}  # Store baseline predictions
        
        # Initialize BERT models
        model_types = ['biobert', 'clinicalbert', 'pubmedbert']
        for model_type in model_types:
            model_path = os.path.join(trained_models_dir, model_type)
            try:
                self.classifiers[model_type] = BERTClassifier(
                    model_type=model_type,
                    model_path=model_path,
                    max_length=max_length,
                    extractor_type=extractor_type
                )
                self.logger.info(f"Loaded {model_type} classifier")
            except Exception as e:
                self.logger.error(f"Failed to load {model_type} classifier: {e}")
        
        # Initialize baseline models
        try:
            baseline_path = os.path.join(trained_models_dir, "baseline")
            self.classifiers['baseline'] = BaselineClassifiers(
                model_dir=baseline_path,
                max_length=max_length,
                extractor_type=extractor_type
            )
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
        self.baseline_predictions = {
            'baseline_log_reg': [],
            'baseline_svm': [],
            'baseline_zero_shot': []
        }
        
        # Clear previous detailed results
        self.detailed_results = []
        
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
                            self.baseline_predictions['baseline_log_reg'].append(1 if log_reg_pred == 'cancer' else 0)
                            self.baseline_predictions['baseline_svm'].append(1 if svm_pred == 'cancer' else 0)
                            
                            # Get zero-shot prediction
                            zero_shot_result = result.get('zero_shot', {})
                            zero_shot_pred = zero_shot_result.get('prediction', '').lower()
                            zero_shot_conf = zero_shot_result.get('confidence', 0.0)
                            self.baseline_predictions['baseline_zero_shot'].append(1 if zero_shot_pred == 'cancer' else 0)
                            
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
                
                self.detailed_results.append(file_result)
                main_pbar.update(1)
            
        finally:
            # Close progress bars
            model_pbar.close()
            main_pbar.close()
        
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
                self.logger.info("Predicted:")
                self.logger.info("                Non-Cancer    Cancer")
                self.logger.info("Actual:")
                self.logger.info(f"Non-Cancer     {cm[0][0]:^10}    {cm[0][1]:^6}")
                self.logger.info(f"Cancer         {cm[1][0]:^10}    {cm[1][1]:^6}")
                self.logger.info("")
                
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
            for approach, preds in self.baseline_predictions.items():
                if preds:  # Only calculate if we have predictions
                    try:
                        # Calculate metrics with zero_division=0
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            expected, preds, average='binary', zero_division=0, labels=[0, 1]
                        )
                        accuracy = accuracy_score(expected, preds)
                        
                        # Calculate confusion matrix with correct label ordering
                        cm = confusion_matrix(expected, preds, labels=[0, 1])
                        
                        # Print confusion matrix with clear labels
                        self.logger.info(f"\nConfusion Matrix for {approach}:")
                        self.logger.info("Predicted:")
                        self.logger.info("                Non-Cancer    Cancer")
                        self.logger.info("Actual:")
                        self.logger.info(f"Non-Cancer     {cm[0][0]:^10}    {cm[0][1]:^6}")
                        self.logger.info(f"Cancer         {cm[1][0]:^10}    {cm[1][1]:^6}")
                        self.logger.info("")
                        
                        self.logger.info(f"Metrics for {approach}:")
                        self.logger.info(f"Accuracy: {accuracy * 100:.2f}%")
                        self.logger.info(f"Precision: {precision * 100:.2f}%")
                        self.logger.info(f"Recall: {recall * 100:.2f}%")
                        self.logger.info(f"F1: {f1 * 100:.2f}%")
                    except Exception as e:
                        self.logger.error(f"Error calculating metrics for {approach}: {e}")
        
        return metrics, predictions

def format_summary(summary: Dict) -> str:
    """Format evaluation summary for display."""
    # Define model display names
    model_display_names = {
        'biobert': 'BioBERT',
        'clinicalbert': 'ClinicalBERT',
        'pubmedbert': 'PubMedBERT',
        'baseline': 'Baseline (LogReg)',
        'baseline_log_reg': 'Baseline LogReg',
        'baseline_svm': 'Baseline SVM',
        'baseline_zero_shot': 'Baseline Zero-Shot'
    }
    
    # Prepare rows with sorted model names
    rows = []
    for model_name in sorted(summary.keys()):
        metrics = summary[model_name]
        if 'error' not in metrics:
            display_name = model_display_names.get(model_name, model_name)
            rows.append([
                display_name,
                f"{metrics['accuracy']:>6.2f}%",
                f"{metrics['precision']:>6.2f}%",
                f"{metrics['recall']:>6.2f}%",
                f"{metrics['f1']:>6.2f}%"
            ])
    
    # Create table with headers
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1']
    table = tabulate(rows, headers=headers, tablefmt='grid', numalign='right')
    
    return table

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
    parser.add_argument('--extractor-type', type=str, choices=['simple', 'sections'],
                        default='simple', help='Type of text extractor to use (default: simple)')
    parser.add_argument('--model', choices=['biobert', 'clinicalbert', 'pubmedbert', 'baseline'],
                        help='Specify a single model to evaluate. If not provided, all models will be evaluated.')
    parser.add_argument('--output-dir', default=None,
                        help='Directory to save detailed results. If not provided, uses models-dir')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.models_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize ensemble
    ensemble = ProtocolClassifierEnsemble(
        trained_models_dir=args.models_dir,
        max_length=args.max_length,
        extractor_type=args.extractor_type
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

    # Store all detailed results
    all_results = []
    all_predictions = {name: [] for name in ensemble.classifiers.keys()}
    all_baseline_predictions = {
        'baseline_log_reg': [],
        'baseline_svm': [],
        'baseline_zero_shot': []
    }
    all_labels = []

    # First evaluate cancer protocols (positive class)
    logger.info("\nEvaluating cancer protocols...")
    cancer_metrics, cancer_predictions = ensemble.evaluate_directory(args.cancer_dir, 'cancer')
    
    # Store cancer protocol results
    for file_result in ensemble.detailed_results:
        file_result['dataset'] = 'cancer'
        all_results.append(file_result)
    
    # Store cancer predictions and labels
    cancer_count = len(ensemble.detailed_results)
    all_labels.extend([1] * cancer_count)
    for name, preds in cancer_predictions.items():
        all_predictions[name].extend(preds)
    for name, preds in ensemble.baseline_predictions.items():
        all_baseline_predictions[name].extend(preds)
    
    # Then evaluate non-cancer protocols (negative class)
    logger.info("\nEvaluating non-cancer protocols...")
    non_cancer_metrics, non_cancer_predictions = ensemble.evaluate_directory(args.non_cancer_dir, 'non_cancer')
    
    # Store non-cancer protocol results
    for file_result in ensemble.detailed_results:
        file_result['dataset'] = 'non_cancer'
        all_results.append(file_result)
    
    # Store non-cancer predictions and labels
    non_cancer_count = len(ensemble.detailed_results)
    all_labels.extend([0] * non_cancer_count)
    for name, preds in non_cancer_predictions.items():
        all_predictions[name].extend(preds)
    for name, preds in ensemble.baseline_predictions.items():
        all_baseline_predictions[name].extend(preds)

    # Calculate and display combined metrics for all models
    logger.info("\nDetailed Results by Model:")
    for name in ensemble.classifiers.keys():
        if name in all_predictions:
            preds = all_predictions[name]
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, preds, average='binary', zero_division=0
            )
            accuracy = accuracy_score(all_labels, preds)
            
            # Calculate confusion matrix
            cm = confusion_matrix(all_labels, preds)
            
            logger.info(f"\nResults for {name}:")
            logger.info("Confusion Matrix:")
            logger.info("Predicted:")
            logger.info("                Non-Cancer    Cancer")
            logger.info("Actual:")
            logger.info(f"Non-Cancer     {cm[0][0]:^10}    {cm[0][1]:^6}")
            logger.info(f"Cancer         {cm[1][0]:^10}    {cm[1][1]:^6}")
            logger.info("")
            logger.info(f"Accuracy:  {accuracy * 100:.2f}%")
            logger.info(f"Precision: {precision * 100:.2f}%")
            logger.info(f"Recall:    {recall * 100:.2f}%")
            logger.info(f"F1:        {f1 * 100:.2f}%")
    
    # Calculate and display combined metrics for baseline approaches
    logger.info("\nDetailed Results by Baseline Approach:")
    for approach in all_baseline_predictions.keys():
        preds = all_baseline_predictions[approach]
        if preds:
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, preds, average='binary', zero_division=0
            )
            accuracy = accuracy_score(all_labels, preds)
            
            # Calculate confusion matrix
            cm = confusion_matrix(all_labels, preds)
            
            logger.info(f"\nResults for {approach}:")
            logger.info("Confusion Matrix:")
            logger.info("Predicted:")
            logger.info("                Non-Cancer    Cancer")
            logger.info("Actual:")
            logger.info(f"Non-Cancer     {cm[0][0]:^10}    {cm[0][1]:^6}")
            logger.info(f"Cancer         {cm[1][0]:^10}    {cm[1][1]:^6}")
            logger.info("")
            logger.info(f"Accuracy:  {accuracy * 100:.2f}%")
            logger.info(f"Precision: {precision * 100:.2f}%")
            logger.info(f"Recall:    {recall * 100:.2f}%")
            logger.info(f"F1:        {f1 * 100:.2f}%")

    # Save detailed results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_csv = os.path.join(output_dir, f"detailed_predictions_{timestamp}.csv")
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    columns = ['dataset', 'file_path', 'expected_class']
    for model in ensemble.classifiers.keys():
        if model == 'baseline':
            columns.extend([
                f'{model}_log_reg_prediction',
                f'{model}_log_reg_confidence',
                f'{model}_svm_prediction',
                f'{model}_svm_confidence',
                f'{model}_zero_shot_prediction',
                f'{model}_zero_shot_confidence'
            ])
        else:
            columns.extend([
                f'{model}_prediction',
                f'{model}_confidence'
            ])
    
    # Add any remaining columns not explicitly ordered
    remaining_cols = [col for col in df.columns if col not in columns]
    columns.extend(remaining_cols)
    
    # Reorder and save
    df = df[columns]
    df.to_csv(detailed_csv, index=False)
    logger.info(f"\nDetailed predictions saved to: {detailed_csv}")

    # Save summary results
    summary = {}
    for name in ensemble.classifiers.keys():
        preds = all_predictions[name]
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average='binary', zero_division=0
        )
        accuracy = accuracy_score(all_labels, preds)
        
        summary[name] = {
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2)
        }
    
    # Add baseline model results to summary
    for approach in all_baseline_predictions.keys():
        preds = all_baseline_predictions[approach]
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average='binary', zero_division=0
        )
        accuracy = accuracy_score(all_labels, preds)
        
        summary[approach] = {
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2)
        }

    # Display final comparison table
    logger.info("\nFinal Model Comparison:")
    print(format_summary(summary))

    # Save results
    results = {
        'cancer': cancer_metrics,
        'non_cancer': non_cancer_metrics,
        'combined': summary,
        'detailed_predictions_file': detailed_csv
    }
    
    results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"\nResults summary saved to: {results_file}")

if __name__ == "__main__":
    main()