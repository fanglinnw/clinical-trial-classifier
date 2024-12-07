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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

        Args:
            pubmedbert_path: Path to PubMedBERT model
            baseline_path: Path to baseline models
            max_length: Maximum text length to process
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

    def classify_pdf(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Classify a single PDF using all available models.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing classifications from all models
        """
        results = {"file_name": str(pdf_path)}

        # Get PubMedBERT prediction
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

        # Get baseline predictions
        if self.baseline:
            try:
                baseline_result = self.baseline.classify_pdf(pdf_path)
                results["traditional_ml"] = baseline_result["traditional_ml"]
                results["zero_shot"] = baseline_result["zero_shot"]
            except Exception as e:
                self.logger.error(f"Baseline classification failed: {e}")
                results["baseline"] = {"error": str(e)}

        return results

    def batch_classify_pdfs(self, pdf_dir: Union[str, Path]) -> List[Dict]:
        """
        Classify all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files

        Returns:
            List of dictionaries containing classifications for each PDF
        """
        pdf_dir = Path(pdf_dir)
        results = []

        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        for pdf_path in pdf_files:
            result = self.classify_pdf(pdf_path)
            results.append(result)

        return results

    def generate_performance_summary(self, results: List[Dict], ground_truth: str = None) -> Dict:
        """
        Generate a comprehensive summary of model performance.

        Args:
            results: List of classification results
            ground_truth: Expected classification label for all documents ("cancer" or "non-cancer")

        Returns:
            Dictionary containing performance metrics
        """
        summary = {
            "total_documents": len(results),
            "model_predictions": {},
            "confidence_stats": {},
            "agreement_analysis": {},
            "error_rates": {}
        }

        if ground_truth:
            summary["ground_truth_evaluation"] = {}

        # Initialize counters
        predictions = {
            "pubmedbert": [],
            "log_reg": [],
            "svm": [],
            "zero_shot": []
        }
        confidences = {
            "pubmedbert": [],
            "log_reg": [],
            "svm": [],
            "zero_shot": []
        }
        errors = {
            "pubmedbert": 0,
            "traditional_ml": 0,
            "zero_shot": 0
        }

        # Collect predictions and confidences
        for result in results:
            if "pubmedbert" in result and "error" not in result["pubmedbert"]:
                predictions["pubmedbert"].append(result["pubmedbert"]["classification"])
                confidences["pubmedbert"].append(result["pubmedbert"]["confidence"])
            else:
                errors["pubmedbert"] += 1

            if "traditional_ml" in result and "error" not in result["traditional_ml"]:
                trad = result["traditional_ml"]
                predictions["log_reg"].append(trad.get("log_reg_prediction", ""))
                predictions["svm"].append(trad.get("svm_prediction", ""))
                confidences["log_reg"].append(trad.get("log_reg_confidence", 0))
                confidences["svm"].append(trad.get("svm_confidence", 0))
            else:
                errors["traditional_ml"] += 1

            if "zero_shot" in result and "error" not in result["zero_shot"]:
                zero = result["zero_shot"]
                predictions["zero_shot"].append(zero["prediction"])
                confidences["zero_shot"].append(zero["confidence"])
            else:
                errors["zero_shot"] += 1

        # Calculate prediction distributions
        for model, preds in predictions.items():
            if preds:
                summary["model_predictions"][model] = dict(Counter(preds))

        # Calculate confidence statistics
        for model, confs in confidences.items():
            if confs:
                summary["confidence_stats"][model] = {
                    "mean": np.mean(confs),
                    "median": np.median(confs),
                    "std": np.std(confs),
                    "min": min(confs),
                    "max": max(confs)
                }

        # Calculate agreement between models
        for m1 in predictions.keys():
            for m2 in predictions.keys():
                if m1 < m2:  # avoid duplicate comparisons
                    common_predictions = [
                        (p1, p2) for p1, p2 in zip(predictions[m1], predictions[m2])
                        if p1 and p2  # Only compare when both predictions exist
                    ]
                    if common_predictions:
                        agreement = sum(p1 == p2 for p1, p2 in common_predictions) / len(common_predictions)
                        summary["agreement_analysis"][f"{m1}_vs_{m2}"] = agreement * 100

        # Calculate error rates
        for model, error_count in errors.items():
            summary["error_rates"][model] = (error_count / summary["total_documents"]) * 100

        # Calculate metrics against ground truth if provided
        if ground_truth:
            for model, preds in predictions.items():
                if preds:
                    try:
                        # Filter out empty predictions
                        valid_preds = [(pred, truth) for pred, truth in zip(preds, [ground_truth] * len(preds))
                                       if pred]
                        if valid_preds:
                            y_pred, y_true = zip(*valid_preds)

                            # Calculate accuracy
                            accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_pred)

                            # Calculate precision, recall, and F1 with appropriate average
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                y_true, y_pred,
                                average='weighted',  # Use weighted average for multiclass
                                zero_division=0
                            )

                            summary["ground_truth_evaluation"][model] = {
                                "accuracy": accuracy * 100,
                                "precision": precision * 100,
                                "recall": recall * 100,
                                "f1_score": f1 * 100
                            }
                    except Exception as e:
                        self.logger.error(f"Error calculating metrics for {model}: {e}")
                        summary["ground_truth_evaluation"][model] = {
                            "error": str(e)
                        }

        return summary

def format_result(result: Dict) -> str:
    """Format classification result for display."""
    output = [f"\nResults for {result['file_name']}:"]

    if "pubmedbert" in result:
        pub = result["pubmedbert"]
        if "error" not in pub:
            output.append("\nPubMedBERT Classification:")
            output.append(f"  Classification: {pub['classification']}")
            output.append(f"  Confidence: {pub['confidence']}%")

    if "traditional_ml" in result:
        trad = result["traditional_ml"]
        output.append("\nTraditional ML Classifications:")
        output.append(f"  LogisticRegression: {trad['log_reg_prediction']} ({trad['log_reg_confidence']}%)")
        output.append(f"  SVM: {trad['svm_prediction']} ({trad['svm_confidence']}%)")

    if "zero_shot" in result:
        zero = result["zero_shot"]
        output.append("\nZero-shot Classification:")
        output.append(f"  Classification: {zero['prediction']}")
        output.append(f"  Confidence: {zero['confidence']}%")

    return "\n".join(output)


def format_summary(summary: Dict) -> str:
    """Format performance summary for display."""
    output = ["\n=== Model Performance Summary ===\n"]

    # Basic stats
    output.append(f"Total documents processed: {summary['total_documents']}")

    # Ground truth evaluation if available
    if "ground_truth_evaluation" in summary:
        output.append("\nGround Truth Evaluation:")
        metrics_table = []
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        for model, metrics in summary["ground_truth_evaluation"].items():
            row = [
                model,
                f"{metrics['accuracy']:.2f}%",
                f"{metrics['precision']:.2f}%",
                f"{metrics['recall']:.2f}%",
                f"{metrics['f1_score']:.2f}%"
            ]
            metrics_table.append(row)
        output.append(tabulate(metrics_table, headers=headers, tablefmt="simple"))

    # Prediction distributions
    output.append("\nPrediction Distributions:")
    for model, dist in summary["model_predictions"].items():
        output.append(f"\n{model.upper()}:")
        table = [[label, count] for label, count in dist.items()]
        output.append(tabulate(table, headers=["Label", "Count"], tablefmt="simple"))

    # Confidence statistics
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

    # Model agreement
    output.append("\nModel Agreement:")
    agreement_table = [[pair, f"{agreement:.2f}%"]
                       for pair, agreement in summary["agreement_analysis"].items()]
    output.append(tabulate(agreement_table, headers=["Models", "Agreement"], tablefmt="simple"))

    # Error rates
    output.append("\nError Rates:")
    error_table = [[model, f"{rate:.2f}%"]
                   for model, rate in summary["error_rates"].items()]
    output.append(tabulate(error_table, headers=["Model", "Error Rate"], tablefmt="simple"))

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description='Classify clinical trial protocols')
    parser.add_argument('--input', required=True, help='Path to PDF file or directory')
    parser.add_argument('--output', help='Path to save results JSON file')
    parser.add_argument('--ground-truth', choices=['cancer', 'non-cancer'],
                        help='Expected classification for all documents in the input')
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

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        results = [classifier.classify_pdf(input_path)]
    else:
        results = classifier.batch_classify_pdfs(input_path)

    # Generate and display performance summary
    summary = classifier.generate_performance_summary(results, args.ground_truth)
    print(format_summary(summary))

    # Display individual results
    print("\n=== Individual Results ===")
    for result in results:
        print(format_result(result))

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