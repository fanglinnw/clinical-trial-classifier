import fitz
import logging
from pathlib import Path
import gc
from typing import Optional, List, Dict, Union
import re


class ProtocolTextExtractor:
    def __init__(self, max_length: Optional[int] = None):
        """
        Initialize the protocol text extractor.

        Args:
            max_length: Maximum number of characters to extract. If None, extracts all text.
        """
        self.max_length = max_length

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Common section headers in clinical trial protocols
        self.important_sections = [
            "objective",
            "purpose",
            "study design",
            "methodology",
            "inclusion criteria",
            "exclusion criteria",
            "intervention",
            "treatment",
            "primary outcome",
            "secondary outcome",
            "statistical analysis",
            "eligibility"
        ]

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:-]', '', text)

        return text.strip()

    def extract_from_pdf(self, pdf_path: Union[str, Path],
                         extract_sections: bool = False) -> Dict[str, str]:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            extract_sections: If True, attempt to identify and extract specific sections

        Returns:
            Dictionary containing extracted text and optionally section-specific text
        """
        try:
            doc = fitz.open(str(pdf_path))
            full_text = []
            section_texts = {}
            current_section = None

            for page in doc:
                page_text = page.get_text()

                if extract_sections:
                    # Look for section headers
                    lines = page_text.split('\n')
                    for line in lines:
                        lower_line = line.lower().strip()

                        # Check if line is a section header
                        for section in self.important_sections:
                            if section in lower_line and len(
                                    lower_line) < 100:  # Avoid matching in middle of paragraphs
                                current_section = section
                                section_texts[current_section] = []
                                break

                        # Add text to current section if we're in one
                        if current_section:
                            section_texts[current_section].append(line)

                full_text.append(page_text)

            # Join all text
            complete_text = ' '.join(full_text)
            complete_text = self.clean_text(complete_text)

            # Truncate if needed
            if self.max_length and len(complete_text) > self.max_length:
                complete_text = ' '.join(complete_text[:self.max_length].split()[:-1])

            result = {"full_text": complete_text}

            # Add sections if extracted
            if extract_sections:
                for section, lines in section_texts.items():
                    section_text = ' '.join(lines)
                    section_text = self.clean_text(section_text)
                    result[f"section_{section}"] = section_text

            return result

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return {"full_text": "", "error": str(e)}

        finally:
            if 'doc' in locals():
                doc.close()
            gc.collect()

    def batch_extract_from_pdfs(self, pdf_dir: Union[str, Path],
                                extract_sections: bool = False) -> List[Dict[str, str]]:
        """
        Extract text from all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files
            extract_sections: If True, attempt to identify and extract specific sections

        Returns:
            List of dictionaries containing extracted text for each PDF
        """
        pdf_dir = Path(pdf_dir)
        results = []

        for pdf_path in pdf_dir.glob("**/*.pdf"):
            result = self.extract_from_pdf(pdf_path, extract_sections)
            result["file_name"] = str(pdf_path)
            results.append(result)

        return results

    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """Get basic statistics about the extracted text."""
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.splitlines())
        }


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract text from clinical trial protocols')
    parser.add_argument('--input', required=True, help='PDF file or directory path')
    parser.add_argument('--max-length', type=int, help='Maximum text length to extract')
    parser.add_argument('--extract-sections', action='store_true',
                        help='Extract text by sections')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize extractor
    extractor = ProtocolTextExtractor(max_length=args.max_length)

    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        result = extractor.extract_from_pdf(input_path, args.extract_sections)
        print(f"\nExtracted text statistics:")
        stats = extractor.get_text_statistics(result["full_text"])
        for key, value in stats.items():
            print(f"{key}: {value}")

        if args.extract_sections:
            print("\nExtracted sections:")
            for key, value in result.items():
                if key.startswith("section_"):
                    print(f"\n{key}:")
                    print(value[:200] + "..." if len(value) > 200 else value)
    else:
        results = extractor.batch_extract_from_pdfs(input_path, args.extract_sections)
        print(f"\nProcessed {len(results)} files")
