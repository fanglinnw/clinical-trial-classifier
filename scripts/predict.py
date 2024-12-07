import argparse
import json
import logging
from pathlib import Path
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from models import PubMedBERTClassifier

def setup_logging():
    """Configure logging for the prediction script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def predict_cancer_relevance(
    input_path: str,
    model_path: str = "./protocol_classifier",
    max_length: int = 8000,
    output_file: str = None
) -> dict:
    """
    Predict cancer relevance for a single PDF or directory of PDFs.
    
    Args:
        input_path: Path to a PDF file or directory containing PDFs
        model_path: Path to the trained PubMedBERT model
        max_length: Maximum sequence length for the model
        output_file: Optional path to save results as JSON
        
    Returns:
        Dictionary containing predictions for each input file
    """
    logger = setup_logging()
    
    try:
        # Initialize the model
        logger.info("Loading PubMedBERT classifier...")
        classifier = PubMedBERTClassifier(
            model_path=model_path,
            max_length=max_length
        )
        
        # Process input path
        input_path = Path(input_path)
        results = []
        
        if input_path.is_file():
            # Single file prediction
            if input_path.suffix.lower() != '.pdf':
                raise ValueError(f"Input file must be a PDF: {input_path}")
            
            logger.info(f"Processing file: {input_path}")
            prediction = classifier.classify_pdf(input_path)
            results.append({
                "file_name": str(input_path),
                "prediction": prediction["classification"],
                "confidence": prediction["confidence"]
            })
            
        elif input_path.is_dir():
            # Directory prediction
            pdf_files = list(input_path.glob("**/*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files in {input_path}")
            
            for pdf_path in pdf_files:
                logger.info(f"Processing file: {pdf_path}")
                try:
                    prediction = classifier.classify_pdf(pdf_path)
                    results.append({
                        "file_name": str(pdf_path),
                        "prediction": prediction["classification"],
                        "confidence": prediction["confidence"]
                    })
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {e}")
                    results.append({
                        "file_name": str(pdf_path),
                        "error": str(e)
                    })
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
            
        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({"predictions": results}, f, indent=2)
            logger.info(f"Results saved to {output_file}")
            
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Predict cancer relevance for clinical trial protocols"
    )
    parser.add_argument(
        "input_path",
        help="Path to a PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--model-path",
        default="./protocol_classifier",
        help="Path to the trained PubMedBERT model"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--output",
        help="Optional path to save results as JSON"
    )
    
    args = parser.parse_args()
    
    try:
        results = predict_cancer_relevance(
            args.input_path,
            args.model_path,
            args.max_length,
            args.output
        )
        
        # Print results to console
        for pred in results["predictions"]:
            if "error" in pred:
                print(f"\n{pred['file_name']}: Error - {pred['error']}")
            else:
                print(f"\n{pred['file_name']}:")
                print(f"  Prediction: {pred['prediction']}")
                print(f"  Confidence: {pred['confidence']:.2%}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
