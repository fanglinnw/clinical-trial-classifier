import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from train_classifier import ProtocolDataset, compute_metrics, get_device_settings
import pandas as pd
from datetime import datetime

def analyze_learning_curves(data_dir, output_dir, max_total_samples, num_points=6):
    """
    Analyze learning curves by training on different dataset sizes
    
    Args:
        data_dir: Root directory containing the dataset
        output_dir: Directory to save results
        max_total_samples: Maximum number of samples to consider
        num_points: Number of different dataset sizes to try
    """
    # Create sample sizes to test (exponential scale)
    sample_sizes = np.geomspace(100, max_total_samples//2, num_points, dtype=int)
    
    # Store results
    results = {
        'train_size': [],
        'train_f1': [],
        'val_f1': [],
        'test_f1': [],
        'train_time': []
    }
    
    # Get device settings
    device_settings = get_device_settings()
    
    # Create test dataset once - we'll use the same test set for all evaluations
    test_dataset = ProtocolDataset(data_dir, 'test', max_total_samples//10)  # Using 10% of max data for test
    print(f"Test set size: {len(test_dataset)} samples")
    
    for samples_per_class in sample_sizes:
        print(f"\nTraining with {samples_per_class*2} total samples ({samples_per_class} per class)")
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            num_labels=2
        )
        
        # Create datasets
        train_dataset = ProtocolDataset(data_dir, 'train', samples_per_class)
        val_dataset = ProtocolDataset(data_dir, 'val', samples_per_class//5)  # 20% of train size
        
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
        
        # Evaluate on all sets
        train_metrics = trainer.evaluate(train_dataset)
        val_metrics = trainer.evaluate(val_dataset)
        test_metrics = trainer.evaluate(test_dataset)
        
        # Store results
        results['train_size'].append(samples_per_class * 2)
        results['train_f1'].append(train_metrics['eval_f1'])
        results['val_f1'].append(val_metrics['eval_f1'])
        results['test_f1'].append(test_metrics['eval_f1'])
        results['train_time'].append(train_time)
    
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
    MAX_TOTAL_SAMPLES = 8000  # Your total available samples
    NUM_POINTS = 6  # Number of different dataset sizes to try
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run analysis
    results = analyze_learning_curves(DATA_DIR, OUTPUT_DIR, MAX_TOTAL_SAMPLES, NUM_POINTS)
    
    # Plot and analyze results
    plot_learning_curves(results, OUTPUT_DIR)