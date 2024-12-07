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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from models import PubMedBERTClassifier, BaselineClassifiers


class ProtocolClassifierEnsemble:
    def __init__(self,
                 pubmedbert_path: str = "./protocol_classifier",
                 baseline_path: str = "./baseline_models",
                 max_length: int = 8000):
        """
        Initialize ensemble of classifiers.
        """
        self.logger = logging.getLogger(__name__)

        try:
            self.pubmedbert = PubMedBERTClassifier(
                model_path=pubmedbert_path,
                max_length=max_length
            )
            self.logger.info("Loaded PubMedBERT classifier")
        except Exception as e:
            self.pubmedbert = None
            self.logger.error(f"Failed to load PubMedBERT classifier: {e}")

        try:
            self.baseline = BaselineClassifiers(
                model_dir=baseline_path,
                max_length=max_length
            )
            self.baseline.load_traditional_models()
            self.logger.info("Loaded baseline classifiers")
        except Exception as e:
            self.baseline = None
            self.logger.error(f"Failed to load baseline classifiers: {e}")

    def classify_pdf(self, pdf_path: Union[str, Path], true_label: str = None) -> Dict:
        """
        Classify a single PDF using all available models.
        """
        results = {
            "file_name": str(pdf_path),
            "true_label": true_label
        }

        if self.pubmedbert:
            try:
                pubmedbert_result = self.pubmedbert.classify_pdf(pdf_path)
                results["pubmedbert"] = {
                    "classification": pubmedbert_result["classification"],
                    "confidence": pubmedbert_result["confidence"]
                }
            except Exception as e:
                self.logger.error(f"PubMedBERT classification failed: {e}")
                results["pubmedbert"] = {"error": str(e)}

        if self.baseline:
            try:
                baseline_result = self.baseline.classify_pdf(pdf_path)
                results["traditional_ml"] = baseline_result["traditional_ml"]
                results["zero_shot"] = baseline_result["zero_shot"]
            except Exception as e:
                self.logger.error(f"Baseline classification failed: {e}")
                results["baseline"] = {"error": str(e)}

        return results

    def process_protocol_directories(self, cancer_dir: Union[str, Path], non_cancer_dir: Union[str, Path]) -> List[
        Dict]:
        """
        Process all PDFs in cancer and non-cancer directories.
        """
        results = []

        # Process cancer protocols
        cancer_dir = Path(cancer_dir)
        cancer_pdfs = list(cancer_dir.glob("**/*.pdf"))
        self.logger.info(f"Found {len(cancer_pdfs)} cancer protocol PDFs in {cancer_dir}")
        for pdf_path in cancer_pdfs:
            result = self.classify_pdf(pdf_path, true_label="cancer")
            results.append(result)

        # Process non-cancer protocols
        non_cancer_dir = Path(non_cancer_dir)
        non_cancer_pdfs = list(non_cancer_dir.glob("**/*.pdf"))
        self.logger.info(f"Found {len(non_cancer_pdfs)} non-cancer protocol PDFs in {non_cancer_dir}")
        for pdf_path in non_cancer_pdfs:
            result = self.classify_pdf(pdf_path, true_label="non-cancer")
            results.append(result)

        return results

    def generate_performance_summary(self, results: List[Dict]) -> Dict:
        """
        Generate comprehensive performance metrics for all models.
        """
        summary = {
            "total_documents": len(results),
            "model_metrics": {},
            "confidence_stats": {},
            "confusion_matrices": {},
            "error_rates": {}
        }

        # Initialize data structures for predictions and true labels
        model_data = {
            "pubmedbert": {"preds": [], "conf": [], "true": [], "errors": 0},
            "log_reg": {"preds": [], "conf": [], "true": [], "errors": 0},
            "svm": {"preds": [], "conf": [], "true": [], "errors": 0},
            "zero_shot": {"preds": [], "conf": [], "true": [], "errors": 0}
        }

        # Collect predictions and confidences
        for result in results:
            true_label = result["true_label"]

            # PubMedBERT
            if "pubmedbert" in result and "error" not in result["pubmedbert"]:
                pub = result["pubmedbert"]
                model_data["pubmedbert"]["preds"].append(pub["classification"])
                model_data["pubmedbert"]["conf"].append(pub["confidence"])
                model_data["pubmedbert"]["true"].append(true_label)
            else:
                model_data["pubmedbert"]["errors"] += 1

            # Traditional ML
            if "traditional_ml" in result and "error" not in result["traditional_ml"]:
                trad = result["traditional_ml"]

                # Logistic Regression
                if "log_reg_prediction" in trad:
                    model_data["log_reg"]["preds"].append(trad["log_reg_prediction"])
                    model_data["log_reg"]["conf"].append(trad["log_reg_confidence"])
                    model_data["log_reg"]["true"].append(true_label)
                else:
                    model_data["log_reg"]["errors"] += 1

                # SVM
                if "svm_prediction" in trad:
                    model_data["svm"]["preds"].append(trad["svm_prediction"])
                    model_data["svm"]["conf"].append(trad["svm_confidence"])
                    model_data["svm"]["true"].append(true_label)
                else:
                    model_data["svm"]["errors"] += 1
            else:
                model_data["log_reg"]["errors"] += 1
                model_data["svm"]["errors"] += 1

            # Zero-shot
            if "zero_shot" in result and "error" not in result["zero_shot"]:
                zero = result["zero_shot"]
                model_data["zero_shot"]["preds"].append(zero["prediction"])
                model_data["zero_shot"]["conf"].append(zero["confidence"])
                model_data["zero_shot"]["true"].append(true_label)
            else:
                model_data["zero_shot"]["errors"] += 1

        # Calculate metrics for each model
        for model_name, data in model_data.items():
            if data["preds"]:  # Only calculate metrics if we have predictions
                try:
                    # Calculate basic metrics
                    accuracy = accuracy_score(data["true"], data["preds"])
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        data["true"], data["preds"],
                        average='weighted',
                        zero_division=0
                    )

                    summary["model_metrics"][model_name] = {
                        "accuracy": accuracy * 100,
                        "precision": precision * 100,
                        "recall": recall * 100,
                        "f1_score": f1 * 100
                    }

                    # Calculate confusion matrix
                    conf_matrix = confusion_matrix(
                        data["true"],
                        data["preds"],
                        labels=["cancer", "non-cancer"]
                    )
                    summary["confusion_matrices"][model_name] = conf_matrix.tolist()

                    # Calculate confidence statistics
                    if data["conf"]:
                        summary["confidence_stats"][model_name] = {
                            "mean": np.mean(data["conf"]),
                            "median": np.median(data["conf"]),
                            "std": np.std(data["conf"]),
                            "min": min(data["conf"]),
                            "max": max(data["conf"])
                        }

                except Exception as e:
                    self.logger.error(f"Error calculating metrics for {model_name}: {e}")

            # Calculate error rate
            summary["error_rates"][model_name] = (data["errors"] / summary["total_documents"]) * 100

        return summary


def format_summary(summary: Dict) -> str:
    """Format performance summary for display."""
    output = ["\n=== Model Performance Summary ===\n"]

    # Basic stats
    output.append(f"Total documents processed: {summary['total_documents']}")

    # Model metrics
    output.append("\nModel Performance Metrics:")
    metrics_table = []
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    for model, metrics in summary["model_metrics"].items():
        row = [
            model,
            f"{metrics['accuracy']:.2f}%",
            f"{metrics['precision']:.2f}%",
            f"{metrics['recall']:.2f}%",
            f"{metrics['f1_score']:.2f}%"
        ]
        metrics_table.append(row)
    output.append(tabulate(metrics_table, headers=headers, tablefmt="simple"))

    # Confusion matrices
    output.append("\nConfusion Matrices:")
    for model, matrix in summary["confusion_matrices"].items():
        output.append(f"\n{model.upper()}:")
        conf_table = tabulate(matrix,
                              headers=["Predicted Cancer", "Predicted Non-cancer"],
                              showindex=["Actual Cancer", "Actual Non-cancer"],
                              tablefmt="simple")
        output.append(conf_table)

    # Confidence statistics
    if summary["confidence_stats"]:
        output.append("\nConfidence Statistics:")
        conf_table = []
        headers = ["Model", "Mean", "Median", "Std Dev", "Min", "Max"]
        for model, stats in summary["confidence_stats"].items():
            row = [
                model,
                f"{stats['mean']:.2f}%",
                f"{stats['median']:.2f}%",
                f"{stats['std']:.2f}%",
                f"{stats['min']:.2f}%",
                f"{stats['max']:.2f}%"
            ]
            conf_table.append(row)
        output.append(tabulate(conf_table, headers=headers, tablefmt="simple"))

    # Error rates
    output.append("\nError Rates:")
    error_table = [[model, f"{rate:.2f}%"]
                   for model, rate in summary["error_rates"].items()]
    output.append(tabulate(error_table, headers=["Model", "Error Rate"], tablefmt="simple"))

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description='Classify and evaluate clinical trial protocols')
    parser.add_argument('--cancer-dir', required=True, help='Directory containing cancer protocol PDFs')
    parser.add_argument('--non-cancer-dir', required=True, help='Directory containing non-cancer protocol PDFs')
    parser.add_argument('--output', help='Path to save results JSON file')
    parser.add_argument('--max-length', type=int, default=8000,
                        help='Maximum text length to process')
    parser.add_argument('--pubmedbert-path', default='./protocol_classifier',
                        help='Path to PubMedBERT model')
    parser.add_argument('--baseline-path', default='./baseline_models',
                        help='Path to baseline models')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize classifier ensemble
    classifier = ProtocolClassifierEnsemble(
        pubmedbert_path=args.pubmedbert_path,
        baseline_path=args.baseline_path,
        max_length=args.max_length
    )

    # Process directories and generate results
    results = classifier.process_protocol_directories(args.cancer_dir, args.non_cancer_dir)

    # Generate and display performance summary
    summary = classifier.generate_performance_summary(results)
    print(format_summary(summary))

    # Save results if output path provided
    if args.output:
        output_data = {
            "results": results,
            "summary": summary
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults and summary saved to {args.output}")


if __name__ == "__main__":
    main()