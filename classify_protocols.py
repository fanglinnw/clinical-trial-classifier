import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import fitz
import argparse
import json
from typing import Union, List, Dict
import logging
from tqdm import tqdm
import platform


class ProtocolClassifier:
    def __init__(self, model_path: str = "./protocol_classifier"):
        """
        Initialize the protocol classifier.
        
        Args:
            model_path: Path to the directory containing the trained model and tokenizer
        """
        # Device setup for M1 Mac
        if platform.processor() == 'arm':
            # M1/M2 Mac setup
            device_name = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            # Regular setup for other machines
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device_name)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {device_name}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

        # Set up logging
        logging.basicConfig(level=logging.INFO)

    def extract_text_from_pdf(self, pdf_path: str, max_length: int = 4000) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            max_length: Maximum number of characters to extract
            
        Returns:
            Extracted text from the PDF
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Truncate while keeping whole words
            if len(text) > max_length:
                text = ' '.join(text[:max_length].split()[:-1])
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return ""
        finally:
            if 'doc' in locals():
                doc.close()

    def classify_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Classify a single text.
        
        Args:
            text: The text to classify
            
        Returns:
            Dictionary containing classification results
        """
        if not text:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "error": "Empty or invalid text"
            }

        try:
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Move outputs to CPU for numpy conversion
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
                confidence = probabilities[0][prediction].item()

            # Convert prediction to label
            label = "cancer" if prediction.item() == 1 else "non-cancer"
            
            return {
                "classification": label,
                "confidence": round(confidence * 100, 2)  # Convert to percentage
            }

        except Exception as e:
            self.logger.error(f"Error during classification: {e}")
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def classify_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Union[str, float]]:
        """
        Classify a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing classification results
        """
        text = self.extract_text_from_pdf(str(pdf_path))
        result = self.classify_text(text)
        result["file_name"] = str(pdf_path)
        return result

    def batch_classify_pdfs(self, pdf_dir: Union[str, Path]) -> List[Dict[str, Union[str, float]]]:
        """
        Classify all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDF files
            
        Returns:
            List of dictionaries containing classification results
        """
        pdf_dir = Path(pdf_dir)
        results = []
        
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        for pdf_path in tqdm(pdf_files, desc="Classifying protocols"):
            result = self.classify_pdf(pdf_path)
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description='Classify clinical trial protocols')
    parser.add_argument('--input', required=True, help='Path to PDF file or directory of PDFs')
    parser.add_argument('--output', help='Path to save results JSON file (optional)')
    parser.add_argument('--model-path', default='./protocol_classifier',
                      help='Path to trained model directory (default: ./protocol_classifier)')
    args = parser.parse_args()

    # Initialize classifier
    classifier = ProtocolClassifier(model_path=args.model_path)

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        result = classifier.classify_pdf(input_path)
        results = [result]
        print(f"\nResults for {input_path}:")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']}%")
    else:
        # Directory
        results = classifier.batch_classify_pdfs(input_path)
        print("\nClassification Results Summary:")
        for result in results:
            print(f"\nFile: {result['file_name']}")
            print(f"Classification: {result['classification']}")
            print(f"Confidence: {result['confidence']}%")

    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()