import os
import re
from utils import read_pdf, load_dataset
from collections import Counter
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm

class RuleBasedCancerClassifier:
    def __init__(self):
        # Cancer-specific keywords
        self.cancer_keywords = {
            'cancer', 'tumor', 'tumour', 'neoplasm', 'carcinoma', 'sarcoma', 'leukemia', 
            'lymphoma', 'melanoma', 'oncology', 'metastasis', 'malignant', 'chemotherapy',
            'radiotherapy', 'immunotherapy'
        }
        
        # Cancer types
        self.cancer_types = {
            'breast cancer', 'lung cancer', 'prostate cancer', 'colon cancer', 
            'melanoma', 'lymphoma', 'leukemia', 'pancreatic', 'ovarian', 
            'glioblastoma', 'myeloma'
        }
        
        # Cancer-related treatments and procedures
        self.treatment_keywords = {
            'chemotherapy', 'radiation therapy', 'immunotherapy', 'targeted therapy',
            'hormone therapy', 'stem cell transplant', 'car-t', 'biopsy',
            'resection', 'mastectomy'
        }
        
        # Cancer-related measurements and terminology
        self.medical_terms = {
            'staging', 'grade', 'metastatic', 'tnm', 'recurrent', 'remission',
            'progression', 'survival', 'prognosis', 'oncologist'
        }

    def _count_keyword_matches(self, text: str) -> Dict[str, int]:
        """Count matches for different categories of keywords"""
        text = text.lower()
        
        counts = {
            'cancer_keywords': sum(1 for word in self.cancer_keywords if word in text),
            'cancer_types': sum(1 for cancer_type in self.cancer_types if cancer_type in text),
            'treatment_keywords': sum(1 for treatment in self.treatment_keywords if treatment in text),
            'medical_terms': sum(1 for term in self.medical_terms if term in text)
        }
        
        return counts

    def classify_document(self, text: str) -> Tuple[bool, float, Dict]:
        """
        Classify a document as cancer-related or not using rule-based approach.
        Returns: (is_cancer, confidence_score, details)
        """
        # Count matches in each category
        counts = self._count_keyword_matches(text)
        
        # Calculate confidence score (0 to 1)
        total_matches = sum(counts.values())
        
        # Weighted scoring system
        score = (
            counts['cancer_keywords'] * 2.0 +
            counts['cancer_types'] * 2.5 +
            counts['treatment_keywords'] * 1.5 +
            counts['medical_terms'] * 1.0
        ) / 20.0  # Normalize to 0-1 range
        
        score = min(1.0, score)  # Cap at 1.0
        
        # Classification rules
        is_cancer = score >= 0.3  # Threshold determined empirically
        
        return is_cancer, score, counts

def verify_dataset(protocol_dir: str, output_file: str = "verification_report.txt"):
    """Verify the labels in the dataset and generate a report."""
    # Load the dataset
    texts, labels, file_paths = load_dataset(protocol_dir)
    
    classifier = RuleBasedCancerClassifier()
    results = []
    
    print(f"\nAnalyzing {len(texts)} documents...")
    
    for text, label, file_path in tqdm(zip(texts, labels, file_paths)):
        is_cancer, confidence, details = classifier.classify_document(text)
        predicted_label = 1 if is_cancer else 0
        
        results.append({
            'file': os.path.basename(file_path),
            'actual_label': label,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'details': details,
            'matches': predicted_label == label
        })
    
    # Generate report
    with open(output_file, 'w') as f:
        f.write("Dataset Verification Report\n")
        f.write("=========================\n\n")
        
        # Overall statistics
        total = len(results)
        matches = sum(1 for r in results if r['matches'])
        agreement_rate = (matches / total) * 100
        
        f.write(f"Total documents analyzed: {total}\n")
        f.write(f"Labels matching rule-based classifier: {matches} ({agreement_rate:.1f}%)\n\n")
        
        # List potential mislabeled documents
        f.write("Potential Mislabeled Documents:\n")
        f.write("-----------------------------\n")
        for r in results:
            if not r['matches']:
                f.write(f"\nFile: {r['file']}\n")
                f.write(f"Current label: {'Cancer' if r['actual_label'] == 1 else 'Non-cancer'}\n")
                f.write(f"Suggested label: {'Cancer' if r['predicted_label'] == 1 else 'Non-cancer'}\n")
                f.write(f"Confidence: {r['confidence']:.2f}\n")
                f.write("Keyword matches:\n")
                for category, count in r['details'].items():
                    f.write(f"  - {category}: {count}\n")
                f.write("\n")
    
    print(f"\nVerification complete! Report saved to {output_file}")
    print(f"Agreement rate with current labels: {agreement_rate:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Verify dataset labels using rule-based classification')
    parser.add_argument('--protocol-dir', type=str, required=True,
                      help='Directory containing the protocol documents')
    parser.add_argument('--output-file', type=str, default='verification_report.txt',
                      help='Output file for the verification report')
    
    args = parser.parse_args()
    verify_dataset(args.protocol_dir, args.output_file)

if __name__ == "__main__":
    main()
