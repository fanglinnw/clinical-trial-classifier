import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Union, List
import sys
import os

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


def main():
    parser = argparse.ArgumentParser(description='Classify clinical trial protocols')
    parser.add_argument('--input', required=True, help='Path to PDF file or directory')
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

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        results = [classifier.classify_pdf(input_path)]
    else:
        results = classifier.batch_classify_pdfs(input_path)

    # Display results
    for result in results:
        print(format_result(result))

    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()