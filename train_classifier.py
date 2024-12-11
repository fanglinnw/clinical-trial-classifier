import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import random
from tqdm import tqdm
import pypdf
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import platform

def get_device_settings():
    """Determine device type and optimal settings"""
    if torch.cuda.is_available():  # NVIDIA GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        return {
            'device_type': 'cuda',
            'batch_size': 8 if gpu_memory >= 16 else 4,
            'mixed_precision': 'fp16',
            'gradient_accumulation_steps': 4
        }
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # Apple Silicon
        return {
            'device_type': 'mps',
            'batch_size': 4,  # M3 Max can handle this well
            'mixed_precision': 'no',  # MPS doesn't support mixed precision training yet
            'gradient_accumulation_steps': 8
        }
    else:  # CPU
        return {
            'device_type': 'cpu',
            'batch_size': 2,
            'mixed_precision': 'no',
            'gradient_accumulation_steps': 16
        }

class ProtocolDataset(Dataset):
    def __init__(self, root_dir, split, max_samples=None):
        """
        Args:
            root_dir: Root directory containing cancer/non_cancer subdirs
            split: One of 'train', 'val', or 'test'
            max_samples: Maximum number of samples to use per class (for controlling dataset size)
        """
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.max_length = 512  # Maximum sequence length for BERT
        
        # Get file paths and labels
        cancer_dir = os.path.join(root_dir, 'cancer', split)
        non_cancer_dir = os.path.join(root_dir, 'non_cancer', split)
        
        cancer_files = [(f, 1) for f in os.listdir(cancer_dir) if f.endswith('.pdf')]
        non_cancer_files = [(f, 0) for f in os.listdir(non_cancer_dir) if f.endswith('.pdf')]
        
        # Limit dataset size if specified
        if max_samples:
            cancer_files = cancer_files[:max_samples]
            non_cancer_files = non_cancer_files[:max_samples]
        
        self.samples = cancer_files + non_cancer_files
        random.shuffle(self.samples)
        
        self.cancer_dir = cancer_dir
        self.non_cancer_dir = non_cancer_dir

    def __len__(self):
        return len(self.samples)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        try:
            pdf_reader = pypdf.PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
        return text.strip()

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        pdf_path = os.path.join(self.cancer_dir if label == 1 else self.non_cancer_dir, filename)
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(data_dir, output_dir, max_samples=None, epochs=3):
    """Train the model"""
    # Get device-specific settings
    device_settings = get_device_settings()
    print(f"\nTraining on {device_settings['device_type'].upper()} with settings:")
    print(f"Batch size: {device_settings['batch_size']}")
    print(f"Mixed precision: {device_settings['mixed_precision']}")
    print(f"Gradient accumulation steps: {device_settings['gradient_accumulation_steps']}\n")
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        num_labels=2
    )
    
    # Create datasets
    train_dataset = ProtocolDataset(data_dir, 'train', max_samples)
    val_dataset = ProtocolDataset(data_dir, 'val', max_samples)
    
    # Calculate warmup steps
    num_training_steps = (len(train_dataset) // device_settings['batch_size']) * epochs
    warmup_steps = num_training_steps // 10  # 10% of training steps
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=device_settings['batch_size'],
        per_device_eval_batch_size=device_settings['batch_size'],
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        gradient_accumulation_steps=device_settings['gradient_accumulation_steps'],
        fp16=(device_settings['mixed_precision'] == 'fp16'),
        gradient_checkpointing=True,
        # Disable logging to wandb
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    trainer.train()
    
    # Save the model
    trainer.save_model()
    
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Training parameters
    DATA_DIR = "protocol_documents"
    OUTPUT_DIR = "model_output"
    MAX_SAMPLES = 4000  # Assuming 8000 total docs split between cancer/non-cancer
    EPOCHS = 5  # Increased epochs for larger dataset
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train the model
    train_model(DATA_DIR, OUTPUT_DIR, MAX_SAMPLES, EPOCHS)