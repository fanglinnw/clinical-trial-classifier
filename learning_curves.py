import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datetime import datetime
import logging
import random
import json
import pypdf
from tqdm import tqdm
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device_settings():
    """Determine device type and optimal settings"""
    if torch.cuda.is_available():  # NVIDIA GPU
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            'device_type': 'cuda',
            'batch_size': 8 if gpu_memory >= 16 else 4,
            'mixed_precision': 'fp16',
            'gradient_accumulation_steps': 4
        }
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # Apple Silicon
        return {
            'device_type': 'mps',
            'batch_size': 4,
            'mixed_precision': 'no',
            'gradient_accumulation_steps': 8
        }
    else:  # CPU
        return {
            'device_type': 'cpu',
            'batch_size': 2,
            'mixed_precision': 'no',
            'gradient_accumulation_steps': 16
        }

class ProtocolDataManager:
    """Handles one-time validation and caching of protocol documents"""
    
    def __init__(self, root_dir, cache_dir="protocol_cache"):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "validated_protocols.json")
        self.target_chars = 4000
        
        os.makedirs(cache_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def validate_and_cache_data(self, force_revalidation=False):
        """Validate all PDFs once and cache the results"""
        if not force_revalidation and os.path.exists(self.cache_file):
            self.logger.info("Using existing cached data")
            return self.load_cache()
            
        self.logger.info("Starting validation of all PDF files...")
        
        validated_data = {split: {'cancer': [], 'non_cancer': []} 
                         for split in ['train', 'val', 'test']}
        
        for split in ['train', 'val', 'test']:
            for category in ['cancer', 'non_cancer']:
                directory = os.path.join(self.root_dir, category, split)
                if not os.path.exists(directory):
                    continue
                    
                label = 1 if category == 'cancer' else 0
                
                files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
                for file in tqdm(files, desc=f"Validating {split}/{category}"):
                    try:
                        pdf_path = os.path.join(directory, file)
                        text = self._extract_text_from_pdf(pdf_path)
                        
                        if text.strip():
                            validated_data[split][category].append({
                                'filename': file,
                                'text': text,
                                'label': label
                            })
                        else:
                            self.logger.warning(f"Empty text in file: {file}")
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing {file}: {str(e)}")
        
        self._save_cache(validated_data)
        return validated_data
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file with target character limit"""
        text = ""
        try:
            pdf_reader = pypdf.PdfReader(pdf_path)
            
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                        if len(text) >= self.target_chars:
                            text = text[:self.target_chars]
                            break
                except Exception as e:
                    self.logger.warning(f"Error extracting page text: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            raise
            
        return text.strip()
    
    def _save_cache(self, data):
        """Save validated data to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(data, f)
            self.logger.info(f"Cache saved to {self.cache_file}")
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")
    
    def load_cache(self):
        """Load validated data from cache file"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            return None

class OptimizedProtocolDataset(torch.utils.data.Dataset):
    """Dataset class that uses pre-validated and cached data"""
    
    def __init__(self, validated_data, split, tokenizer, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = 512
        
        # Combine cancer and non_cancer samples
        samples = (validated_data[split]['cancer'] + 
                  validated_data[split]['non_cancer'])
        
        # Shuffle and limit samples if specified
        random.shuffle(samples)
        if max_samples:
            # Ensure balanced classes when limiting samples
            cancer_samples = [s for s in samples if s['label'] == 1][:max_samples//2]
            non_cancer_samples = [s for s in samples if s['label'] == 0][:max_samples//2]
            samples = cancer_samples + non_cancer_samples
            random.shuffle(samples)
            
        self.samples = samples
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created {split} dataset with {len(samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['label'], dtype=torch.long)
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

def analyze_learning_curves(data_dir, output_dir, max_total_samples, num_points=6):
    """Analyze learning curves with improved data handling"""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    
    # Initialize data manager and validate files once
    data_manager = ProtocolDataManager(data_dir)
    validated_data = data_manager.validate_and_cache_data()
    
    if not validated_data:
        logger.error("Failed to load or validate data")
        return None
    
    # Create sample sizes to test (exponential scale)
    sample_sizes = np.geomspace(100, max_total_samples//2, num_points, dtype=int)
    
    results = {
        'train_size': [],
        'train_f1': [],
        'val_f1': [],
        'test_f1': [],
        'train_time': [],
        'valid_samples': []
    }
    
    # Get device settings
    device_settings = get_device_settings()
    
    # Create test dataset once
    test_dataset = OptimizedProtocolDataset(validated_data, 'test', tokenizer)
    logger.info(f"Created test dataset with {len(test_dataset)} samples")
    
    for samples_per_class in sample_sizes:
        logger.info(f"\nTraining with {samples_per_class*2} total samples ({samples_per_class} per class)")
        
        try:
            # Create datasets using cached data
            train_dataset = OptimizedProtocolDataset(validated_data, 'train', tokenizer, samples_per_class*2)
            val_dataset = OptimizedProtocolDataset(validated_data, 'val', tokenizer, samples_per_class//2)
            
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
    
    # Find point of diminishing returns
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
    MAX_TOTAL_SAMPLES = 20000
    NUM_POINTS = 10 
    
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
