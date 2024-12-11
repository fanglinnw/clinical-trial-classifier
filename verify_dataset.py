import os
import argparse
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from tqdm import tqdm
from utils import read_pdf, load_dataset

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

def get_nct_ids_from_dir(directory: str) -> Set[str]:
    """Extract NCT IDs from filenames in a directory."""
    nct_ids = set()
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                # Assuming filename format is NCT[0-9]+.pdf
                nct_id = filename.split('.')[0]
                nct_ids.add(nct_id)
    return nct_ids

def check_dataset_overlap(train_dir: str, test_dir: str) -> Dict:
    """Check for any overlap between training and test datasets."""
    # Get all subdirectories
    train_cancer_dir = os.path.join(train_dir, 'cancer')
    train_non_cancer_dir = os.path.join(train_dir, 'non_cancer')
    test_cancer_dir = os.path.join(test_dir, 'cancer')
    test_non_cancer_dir = os.path.join(test_dir, 'non_cancer')

    # Get NCT IDs from each directory
    train_cancer_ids = get_nct_ids_from_dir(train_cancer_dir)
    train_non_cancer_ids = get_nct_ids_from_dir(train_non_cancer_dir)
    test_cancer_ids = get_nct_ids_from_dir(test_cancer_dir)
    test_non_cancer_ids = get_nct_ids_from_dir(test_non_cancer_dir)

    # Check for overlaps
    overlaps = {
        'train_cancer_vs_non_cancer': train_cancer_ids & train_non_cancer_ids,
        'test_cancer_vs_non_cancer': test_cancer_ids & test_non_cancer_ids,
        'train_vs_test_cancer': train_cancer_ids & test_cancer_ids,
        'train_vs_test_non_cancer': train_non_cancer_ids & test_non_cancer_ids,
        'train_cancer_vs_test_non_cancer': train_cancer_ids & test_non_cancer_ids,
        'train_non_cancer_vs_test_cancer': train_non_cancer_ids & test_cancer_ids
    }

    return overlaps

def verify_dataset(train_dir: str, test_dir: str, output_file: str = "dataset_verification_report.txt"):
    """Verify the dataset and generate a comprehensive report."""
    print("\nStarting dataset verification...")
    
    # Initialize report sections
    report_sections = []
    
    # 1. Check directory structure
    report_sections.append("1. Directory Structure Check")
    report_sections.append("==========================")
    
    dirs_to_check = [
        (train_dir, "Training directory"),
        (os.path.join(train_dir, 'cancer'), "Training cancer directory"),
        (os.path.join(train_dir, 'non_cancer'), "Training non-cancer directory"),
        (test_dir, "Test directory"),
        (os.path.join(test_dir, 'cancer'), "Test cancer directory"),
        (os.path.join(test_dir, 'non_cancer'), "Test non-cancer directory")
    ]
    
    for dir_path, dir_name in dirs_to_check:
        exists = os.path.exists(dir_path)
        report_sections.append(f"{dir_name}: {'✓ exists' if exists else '✗ missing'}")
    
    # 2. Dataset size check
    report_sections.append("\n2. Dataset Size Check")
    report_sections.append("===================")
    
    size_stats = {}
    for dir_path, dir_name in dirs_to_check:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.pdf')]
            size_stats[dir_name] = len(files)
            report_sections.append(f"{dir_name}: {len(files)} files")
    
    # 3. Check for overlaps
    report_sections.append("\n3. Dataset Overlap Check")
    report_sections.append("======================")
    
    overlaps = check_dataset_overlap(train_dir, test_dir)
    has_overlaps = False
    
    for overlap_type, overlap_ids in overlaps.items():
        if overlap_ids:
            has_overlaps = True
            report_sections.append(f"\n⚠️ Found overlap in {overlap_type}:")
            report_sections.append(f"Overlapping NCT IDs: {', '.join(sorted(overlap_ids))}")
    
    if not has_overlaps:
        report_sections.append("✓ No overlaps found between any directories")
    
    # 4. Label verification using rule-based classifier
    report_sections.append("\n4. Label Verification")
    report_sections.append("===================")
    
    classifier = RuleBasedCancerClassifier()
    verification_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'mismatches': []})
    
    # Verify both training and test sets
    for dataset_type, base_dir in [('Training', train_dir), ('Test', test_dir)]:
        if not os.path.exists(base_dir):
            continue
            
        texts, labels, file_paths = load_dataset(base_dir)
        
        print(f"\nVerifying {dataset_type} dataset labels...")
        for text, label, file_path in tqdm(zip(texts, labels, file_paths)):
            is_cancer, confidence, details = classifier.classify_document(text)
            predicted_label = 1 if is_cancer else 0
            
            stats = verification_stats[dataset_type]
            stats['total'] += 1
            if predicted_label == label:
                stats['correct'] += 1
            else:
                stats['mismatches'].append({
                    'file': os.path.basename(file_path),
                    'expected': 'Cancer' if label == 1 else 'Non-cancer',
                    'predicted': 'Cancer' if predicted_label == 1 else 'Non-cancer',
                    'confidence': confidence
                })
    
    # Add verification results to report
    for dataset_type, stats in verification_stats.items():
        if stats['total'] > 0:
            agreement_rate = (stats['correct'] / stats['total']) * 100
            report_sections.append(f"\n{dataset_type} Dataset:")
            report_sections.append(f"Total documents: {stats['total']}")
            report_sections.append(f"Label agreement rate: {agreement_rate:.1f}%")
            
            if stats['mismatches']:
                report_sections.append("\nPotential mislabeled documents:")
                for mismatch in stats['mismatches'][:10]:  # Show only first 10 mismatches
                    report_sections.append(
                        f"\n- {mismatch['file']}: "
                        f"Expected {mismatch['expected']}, "
                        f"Predicted {mismatch['predicted']} "
                        f"(confidence: {mismatch['confidence']:.2f})"
                    )
                if len(stats['mismatches']) > 10:
                    report_sections.append(f"... and {len(stats['mismatches']) - 10} more")
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write("Dataset Verification Report\n")
        f.write("=========================\n\n")
        f.write('\n'.join(report_sections))
    
    print(f"\nVerification complete! Report saved to {output_file}")
    
    # Return verification status
    verification_passed = (
        not has_overlaps and
        all(stats['correct'] / stats['total'] >= 0.8 for stats in verification_stats.values() if stats['total'] > 0)
    )
    
    if verification_passed:
        print("\n✓ Dataset verification PASSED")
    else:
        print("\n⚠️ Dataset verification FAILED - Please check the report for details")
    
    return verification_passed

def main():
    parser = argparse.ArgumentParser(description='Verify dataset integrity and label correctness')
    parser.add_argument('--train-dir', type=str, default="protocol_documents",
                      help='Training data directory (default: protocol_documents)')
    parser.add_argument('--test-dir', type=str, default="protocol_documents_test",
                      help='Test data directory (default: protocol_documents_test)')
    parser.add_argument('--output-file', type=str, default="dataset_verification_report.txt",
                      help='Output file for the verification report')
    
    args = parser.parse_args()
    verify_dataset(args.train_dir, args.test_dir, args.output_file)

if __name__ == "__main__":
    main()
