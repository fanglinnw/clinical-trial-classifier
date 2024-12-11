import os
import re
import fitz
from torch.utils.data import Dataset
import torch

class ProtocolDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def preprocess_text(text):
    """Clean and preprocess text from PDF."""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters and keep only alphanumeric, basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,;:?!()\[\]{}\-\'\"]+', ' ', text)
    
    # Remove extra spaces around punctuation
    text = re.sub(r'\s*([.,;:?!()\[\]{}\-])\s*', r'\1 ', text)
    
    # Fix common PDF parsing artifacts
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between camelCase
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Add space between letters and numbers
    
    # Strip extra whitespace
    text = text.strip()
    
    # Collapse multiple spaces (again, just to be sure)
    text = ' '.join(text.split())
    
    return text

def read_pdf(file_path, max_chars=8000):
    """Extract text from first few pages of a PDF file until we reach max_chars."""
    doc = fitz.open(file_path)
    text = ""
    total_text = ""  # Store complete text to check truncation ratio
    
    try:
        for page in doc:
            page_text = page.get_text()
            # Clean each page's text individually to maintain max_chars accuracy
            page_text = preprocess_text(page_text)
            total_text += page_text
            text += page_text
            if len(text) >= max_chars:
                # Truncate to exactly max_chars and break
                text = text[:max_chars]
                break
        
        # Warn if we're truncating more than 75% of the document
        if len(total_text) > 4 * max_chars:
            print(f"Warning: Document {os.path.basename(file_path)} is {len(total_text)} chars long, "
                  f"truncating {len(total_text) - max_chars} chars ({(1 - max_chars/len(total_text))*100:.1f}%)")
        
        return text
    
    finally:
        doc.close()

def load_dataset(protocol_dir, debug=False, debug_samples=5):
    """
    Load protocol documents and their labels.
    
    Args:
        protocol_dir: Directory containing the protocol documents
        debug: If True, only load a small subset of data for debugging
        debug_samples: Number of samples per class to load in debug mode
    """
    texts = []
    labels = []
    
    # Assuming cancer-relevant protocols are in a subdirectory named 'cancer'
    # and non-relevant ones are in 'non_cancer'
    cancer_dir = os.path.join(protocol_dir, 'cancer')
    non_cancer_dir = os.path.join(protocol_dir, 'non_cancer')
    
    if debug:
        print("\nRunning in DEBUG mode")
        print(f"Loading {debug_samples} samples per class...")
    
    # Load cancer protocols
    print("\nLoading cancer protocols...")
    for filename in os.listdir(cancer_dir):
        if filename.endswith('.pdf'):
            if debug and len([l for l in labels if l == 1]) >= debug_samples:
                break
                
            file_path = os.path.join(cancer_dir, filename)
            try:
                text = read_pdf(file_path)
                if text.strip():  # Only add if text is not empty after cleaning
                    texts.append(text)
                    labels.append(1)
                else:
                    print(f"Warning: Empty text after preprocessing in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Load non-cancer protocols
    print("\nLoading non-cancer protocols...")
    for filename in os.listdir(non_cancer_dir):
        if filename.endswith('.pdf'):
            if debug and len([l for l in labels if l == 0]) >= debug_samples:
                break
                
            file_path = os.path.join(non_cancer_dir, filename)
            try:
                text = read_pdf(file_path)
                if text.strip():  # Only add if text is not empty after cleaning
                    texts.append(text)
                    labels.append(0)
                else:
                    print(f"Warning: Empty text after preprocessing in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nLoaded {len(texts)} total documents:")
    print(f"- Cancer protocols: {sum(labels)}")
    print(f"- Non-cancer protocols: {len(labels) - sum(labels)}")
    
    return texts, labels
