import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm
import random
import pypdf

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustProtocolDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, max_samples=None):
        """
        Args:
            root_dir: Root directory containing cancer/non_cancer subdirs
            split: One of 'train', 'val', or 'test'
            max_samples: Maximum number of samples to use per class
        """
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.max_length = 512
        
        # Get file paths and labels
        cancer_dir = os.path.join(root_dir, 'cancer', split)
        non_cancer_dir = os.path.join(root_dir, 'non_cancer', split)
        
        # Validate and filter files
        cancer_files = self._validate_files(cancer_dir, label=1, max_samples=max_samples)
        non_cancer_files = self._validate_files(non_cancer_dir, label=0, max_samples=max_samples)
        
        self.samples = cancer_files + non_cancer_files
        if not self.samples:
            raise ValueError(f"No valid files found in {split} split")
            
        random.shuffle(self.samples)
        logger.info(f"Loaded {len(self.samples)} valid files for {split} split")
        
    def _validate_files(self, directory, label, max_samples=None):
        """Validate and filter PDF files, checking each one can be read"""
        valid_files = []
        all_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        
        for file in tqdm(all_files, desc=f"Validating files in {os.path.basename(directory)}"):
            try:
                pdf_path = os.path.join(directory, file)
                text = self._extract_text_from_pdf(pdf_path)
                if text.strip():  # Check if text is non-empty after stripping whitespace
                    valid_files.append((file, label, text))  # Store the extracted text
                else:
                    logger.warning(f"Empty text in file: {file}")
            except Exception as e:
                logger.warning(f"Error processing {file}: {str(e)}")
        
        if max_samples:
            valid_files = valid_files[:max_samples]
            
        return valid_files

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file with better error handling"""
        text = ""
        try:
            pdf_reader = pypdf.PdfReader(pdf_path)
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                except Exception as e:
                    logger.warning(f"Error extracting text from page in {pdf_path}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error reading PDF {pdf_path}: {str(e)}")
            raise  # Re-raise to be caught by _validate_files
            
        return text.strip()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label, text = self.samples[idx]  # Use pre-extracted text
        
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

def analyze_learning_curves(data_dir, output_dir, max_total_samples, num_points=6):
    """
    Analyze learning curves with robust error handling
    """
    # Create sample sizes to test (exponential scale)
    sample_sizes = np.geomspace(100, max_total_samples//2, num_points, dtype=int)
    
    results = {
        'train_size': [],
        'train_f1': [],
        'val_f1': [],
        'test_f1': [],
        'train_time': [],
        'valid_samples': []  # Track number of valid samples
    }
    
    # Get device settings
    device_settings = get_device_settings()
    
    # Create test dataset once
    try:
        test_dataset = RobustProtocolDataset(data_dir, 'test', max_total_samples//10)
        logger.info(f"Created test dataset with {len(test_dataset)} samples")
    except Exception as e:
        logger.error(f"Error creating test dataset: {str(e)}")
        return None
    
    for samples_per_class in sample_sizes:
        logger.info(f"\nTraining with {samples_per_class*2} total samples ({samples_per_class} per class)")
        
        try:
            # Create datasets
            train_dataset = RobustProtocolDataset(data_dir, 'train', samples_per_class)
            val_dataset = RobustProtocolDataset(data_dir, 'val', samples_per_class//5)
            
            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(
                'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                num_labels=2
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"temp_{samples_per_class}"),
                num_train_epochs=3,
                per_device_train_batch_size=device_settings['batch_size'],
                per_device_eval_batch_size=device_settings['batch_size'],
                evaluation_strategy="epoch",
                save_strategy="no",
                learning_rate=2e-5,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=50,
                gradient_accumulation_steps=device_settings['gradient_accumulation_steps'],
                fp16=(device_settings['mixed_precision'] == 'fp16'),
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
            
            # Train and measure time
            start_time = datetime.now()
            train_result = trainer.train()
            train_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate
            train_metrics = trainer.evaluate(train_dataset)
            val_metrics = trainer.evaluate(val_dataset)
            test_metrics = trainer.evaluate(test_dataset)
            
            # Store results
            results['train_size'].append(samples_per_class * 2)
            results['train_f1'].append(train_metrics['eval_f1'])
            results['val_f1'].append(val_metrics['eval_f1'])
            results['test_f1'].append(test_metrics['eval_f1'])
            results['train_time'].append(train_time)
            results['valid_samples'].append(len(train_dataset))
            
        except Exception as e:
            logger.error(f"Error in training loop for {samples_per_class} samples: {str(e)}")
            continue
    
    return results

def plot_learning_curves(results, output_dir):
    """Plot learning curves and save results"""
    plt.figure(figsize=(12, 8))
    
    # Plot F1 scores
    plt.subplot(2, 1, 1)
    plt.plot(results['train_size'], results['train_f1'], 'bo-', label='Training F1')
    plt.plot(results['train_size'], results['val_f1'], 'ro-', label='Validation F1')
    plt.plot(results['train_size'], results['test_f1'], 'go-', label='Test F1')
    plt.xscale('log')
    plt.xlabel('Training Set Size (total samples)')
    plt.ylabel('F1 Score')
    plt.title('Learning Curves: F1 Score vs Dataset Size')
    plt.grid(True)
    plt.legend()
    
    # Plot training time
    plt.subplot(2, 1, 2)
    plt.plot(results['train_size'], results['train_time'], 'mo-')
    plt.xscale('log')
    plt.xlabel('Training Set Size (total samples)')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Dataset Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    
    # Save numerical results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'learning_curves_data.csv'), index=False)
    
    # Print analysis
    print("\nLearning Curves Analysis:")
    print("-" * 50)
    
    # Calculate metrics
    max_test_f1 = max(results['test_f1'])
    max_test_f1_idx = results['test_f1'].index(max_test_f1)
    optimal_size = results['train_size'][max_test_f1_idx]
    
    # Find point of diminishing returns (where improvement rate drops below threshold)
    improvements = np.diff(results['test_f1'])
    improvement_rates = improvements / np.array(results['test_f1'][:-1])
    threshold = 0.01  # 1% improvement threshold
    
    try:
        diminishing_returns_idx = np.where(improvement_rates < threshold)[0][0] + 1
        diminishing_returns_size = results['train_size'][diminishing_returns_idx]
    except IndexError:
        diminishing_returns_size = results['train_size'][-1]
    
    print(f"Best test F1: {max_test_f1:.3f} (achieved with {optimal_size} samples)")
    print(f"Point of diminishing returns: ~{diminishing_returns_size} samples")
    
    # Calculate generalization gaps
    train_test_gap = np.array(results['train_f1']) - np.array(results['test_f1'])
    worst_gap_idx = np.argmax(train_test_gap)
    
    print(f"\nGeneralization Analysis:")
    print(f"Maximum train-test gap: {train_test_gap[worst_gap_idx]:.3f} at {results['train_size'][worst_gap_idx]} samples")
    
    if optimal_size >= results['train_size'][-1]:
        print("\nRecommendation: Consider collecting more data as test performance might still improve")
    else:
        print(f"\nRecommendation: A dataset size of {optimal_size} samples appears optimal")

if __name__ == "__main__":
    DATA_DIR = "protocol_documents"
    OUTPUT_DIR = "learning_curves_output"
    MAX_TOTAL_SAMPLES = 8000
    NUM_POINTS = 6
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set up file logging
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, 'learning_curves.log'))
    logger.addHandler(file_handler)
    
    # Run analysis
    results = analyze_learning_curves(DATA_DIR, OUTPUT_DIR, MAX_TOTAL_SAMPLES, NUM_POINTS)
    
    if results:
        plot_learning_curves(results, OUTPUT_DIR)
    else:
        logger.error("Analysis failed to complete")