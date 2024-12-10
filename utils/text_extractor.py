import fitz
import logging
from pathlib import Path
import gc
from typing import Optional, List, Dict, Union
import re
from tqdm import tqdm
import argparse
from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """Abstract base class for protocol extractors"""

    @abstractmethod
    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, str]:
        pass

    @abstractmethod
    def process_directory(self, input_dir: str, output_dir: str):
        pass


class SimpleHeadExtractor(BaseExtractor):
    """Simple extractor that gets the first N characters"""

    def __init__(self, max_chars: int = 8000):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.max_chars = max_chars

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:-]', '', text)
        return text.strip()

    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, str]:
        try:
            doc = fitz.open(str(pdf_path))
            text = ""

            # Extract text page by page until we reach max_chars
            for page in doc:
                page_text = page.get_text()
                text += page_text

                # Stop if we have enough text
                if len(text) >= self.max_chars:
                    break

            # Clean and truncate text
            text = self.clean_text(text)
            if len(text) > self.max_chars:
                text = text[:self.max_chars]
                # Trim to last complete word
                text = ' '.join(text.split()[:-1])

            return {"full_text": text}

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return {"full_text": "", "error": str(e)}

        finally:
            if 'doc' in locals():
                doc.close()
            gc.collect()

    def process_directory(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for pdf_file in tqdm(list(input_path.glob("*.pdf")), desc="Processing (Simple Head)"):
            try:
                result = self.extract_from_pdf(pdf_file)
                if result.get("full_text"):
                    output_file = output_path / f"{pdf_file.stem}_simple.txt"
                    self.save_analysis(result, output_file)
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file}: {str(e)}")

    def save_analysis(self, result: Dict[str, str], output_path: Path):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("PROTOCOL HEAD EXTRACTION\n")
                f.write("=====================\n\n")
                f.write(f"First {self.max_chars} characters:\n\n")
                f.write(result["full_text"])
        except Exception as e:
            self.logger.error(f"Error saving to {output_path}: {str(e)}")


class SectionMatchExtractor(BaseExtractor):
    """Sophisticated extractor that matches key sections"""

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Define important sections with their patterns
        self.section_patterns = {
            'synopsis': [
                r'^protocol synopsis',
                r'^study synopsis',
                r'^synopsis$',
                r'^summary$'
            ],
            'background': [
                r'^background$',
                r'^introduction$',
                r'^background and rationale'
            ],
            'objectives': [
                r'^objectives?$',
                r'^study objectives?',
                r'^primary objectives?',
                r'^secondary objectives?',
                r'^\d+\.\d*\s*objectives?'
            ],
            'study_design': [
                r'^study design',
                r'^trial design',
                r'^methodology',
                r'^\d+\.\d*\s*study design'
            ],
            'population': [
                r'^study population',
                r'^target population',
                r'^subject selection'
            ],
            'eligibility': [
                r'^eligibility criteria',
                r'^inclusion criteria',
                r'^exclusion criteria',
                r'^subject eligibility'
            ],
            'treatment': [
                r'^treatment',
                r'^intervention',
                r'^study medication',
                r'^study treatment'
            ],
            'outcomes': [
                r'^outcomes?$',
                r'^endpoints?$',
                r'^primary endpoint',
                r'^secondary endpoint',
                r'^outcome measures?'
            ],
            'statistical_analysis': [
                r'^statistical analysis',
                r'^statistical considerations',
                r'^statistical methods'
            ]
        }

        # Compile patterns
        self.compiled_patterns = {
            section: [re.compile(pattern, re.IGNORECASE)
                      for pattern in patterns]
            for section, patterns in self.section_patterns.items()
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:-]', '', text)
        return text.strip()

    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, str]:
        try:
            doc = fitz.open(str(pdf_path))
            full_text = []
            section_texts = {}
            current_section = None
            current_section_text = []

            for page in doc:
                page_text = page.get_text()
                lines = page_text.split('\n')

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check for section headers
                    found_section = False
                    for section, patterns in self.compiled_patterns.items():
                        if any(pattern.match(line) for pattern in patterns):
                            # Save previous section if exists
                            if current_section and current_section_text:
                                section_texts[current_section] = self.clean_text('\n'.join(current_section_text))

                            # Start new section
                            current_section = section
                            current_section_text = []
                            found_section = True
                            break

                    # Add line to current section
                    if current_section and not found_section:
                        current_section_text.append(line)
                    full_text.append(line)

            # Save last section
            if current_section and current_section_text:
                section_texts[current_section] = self.clean_text('\n'.join(current_section_text))

            # Prepare result
            result = {
                "full_text": self.clean_text(' '.join(full_text))
            }

            # Add sections
            for section, text in section_texts.items():
                result[f"section_{section}"] = text

            return result

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")
            return {"full_text": "", "error": str(e)}

        finally:
            if 'doc' in locals():
                doc.close()
            gc.collect()

    def process_directory(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for pdf_file in tqdm(list(input_path.glob("*.pdf")), desc="Processing (Section Match)"):
            try:
                result = self.extract_from_pdf(pdf_file)
                if result.get("full_text"):
                    output_file = output_path / f"{pdf_file.stem}_sections.txt"
                    self.save_analysis(result, output_file)
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file}: {str(e)}")

    def save_analysis(self, result: Dict[str, str], output_path: Path):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("PROTOCOL SECTION ANALYSIS\n")
                f.write("=======================\n\n")

                # Write sections in specific order
                section_order = [
                    'synopsis', 'background', 'objectives', 'study_design',
                    'population', 'eligibility', 'treatment', 'outcomes',
                    'statistical_analysis'
                ]

                for section in section_order:
                    section_key = f"section_{section}"
                    if section_key in result:
                        title = section.replace('_', ' ').upper()
                        f.write(f"{title}\n")
                        f.write("=" * len(title) + "\n\n")
                        f.write(result[section_key] + "\n\n")

        except Exception as e:
            self.logger.error(f"Error saving to {output_path}: {str(e)}")


def get_extractor(extractor_type: str) -> BaseExtractor:
    """Factory function to get the desired extractor"""
    extractors = {
        'simple': SimpleHeadExtractor,
        'sections': SectionMatchExtractor
    }

    if extractor_type not in extractors:
        raise ValueError(f"Unknown extractor type: {extractor_type}")

    return extractors[extractor_type]()


def main():
    parser = argparse.ArgumentParser(description='Extract text from clinical trial protocols')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing PDF protocols')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save results')
    parser.add_argument('--extractor', type=str,
                        choices=['simple', 'sections', 'both'],
                        default='both',
                        help='Which extractor to use')
    parser.add_argument('--max_chars', type=int, default=8000,
                        help='Maximum characters to extract in simple mode')

    args = parser.parse_args()

    if args.extractor == 'both':
        # Run both extractors
        simple_ext = SimpleHeadExtractor(max_chars=args.max_chars)
        simple_ext.process_directory(args.input_dir, args.output_dir)

        section_ext = SectionMatchExtractor()
        section_ext.process_directory(args.input_dir, args.output_dir)
    else:
        # Run single extractor
        extractor = get_extractor(args.extractor)
        if args.extractor == 'simple':
            extractor.max_chars = args.max_chars
        extractor.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()