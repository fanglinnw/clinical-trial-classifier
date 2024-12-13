import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from utils import ProtocolDataset, load_dataset
from train_classifier import compute_metrics

def calculate_experiment_sizes(total_size_per_class, num_points=6):
    """
    Calculate experiment sizes based on actual dataset size.
    Uses a logarithmic scale to better distribute the test points.
    
    Args:
        total_size_per_class: Total number of samples per class
        num_points: Number of different sizes to try
    
    Returns:
        List of dataset sizes to experiment with
    """
    # Ensure minimum size is at least 100
    min_size = min(100, total_size_per_class // 10)
    
    # Generate logarithmically spaced points
    sizes = np.logspace(
        np.log10(min_size),
        np.log10(total_size_per_class),
        num_points
    ).astype(int)
    
    # Round to nearest 50 for cleaner numbers
    sizes = np.unique((sizes / 50).round() * 50)
    
    return sorted(sizes.tolist())

def run_experiment(
    train_dir: str,
    output_dir: str,
    dataset_sizes: list = None,
    epochs: int = 5,
    learning_rate: float = 1e-5,
    num_runs: int = 3,
    max_size_per_class: int = None
):
    """
    Run experiments with different dataset sizes and track performance metrics.
    
    Args:
        train_dir: Directory containing the training data
        output_dir: Directory to save results
        dataset_sizes: List of dataset sizes to try (samples per class). If None,
                      automatically determined based on available data
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        num_runs: Number of runs per dataset size
        max_size_per_class: Maximum samples per class to consider. If None,
                           uses all available data
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Load full dataset first to determine total size
    print("\nLoading full dataset to determine size...")
    full_texts, full_labels, _ = load_dataset(train_dir)
    class_counts = np.bincount(full_labels)
    min_class_size = min(class_counts)
    
    if max_size_per_class is None:
        max_size_per_class = min_class_size
    else:
        max_size_per_class = min(max_size_per_class, min_class_size)
    
    # Calculate experiment sizes if not provided
    if dataset_sizes is None:
        dataset_sizes = calculate_experiment_sizes(max_size_per_class)
    
    print(f"\nTotal samples available per class: {min_class_size}")
    print(f"Will experiment with following sizes: {dataset_sizes}")
    
    # Check available hardware
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple M1 GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    # Initialize tokenizer
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for size in dataset_sizes:
        print(f"\nExperimenting with dataset size: {size} samples per class")
        
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")
            
            # Load and subsample dataset
            texts, labels, _ = load_dataset(train_dir, debug=True, debug_samples=size)
            
            # Split dataset
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42 + run, stratify=labels
            )
            
            # Create datasets
            train_dataset = ProtocolDataset(train_texts, train_labels, tokenizer)
            val_dataset = ProtocolDataset(val_texts, val_labels, tokenizer)
            
            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                problem_type="single_label_classification"
            ).to(device)
            
            # Training arguments
            run_output_dir = os.path.join(output_dir, f"size_{size}_run_{run}")
            training_args = TrainingArguments(
                output_dir=run_output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=8 if device.type == 'cpu' else 16,
                per_device_eval_batch_size=8 if device.type == 'cpu' else 16,
                weight_decay=0.01,
                learning_rate=learning_rate,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                logging_dir=os.path.join(run_output_dir, 'logs'),
                logging_steps=10,
                fp16=device.type == 'cuda',
                fp16_full_eval=False
            )
            
            # Early stopping
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.01
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping_callback]
            )
            
            # Train and evaluate
            train_result = trainer.train()
            eval_result = trainer.evaluate()
            
            # Record results
            results.append({
                'dataset_size': size,
                'run': run + 1,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'eval_accuracy': eval_result['eval_accuracy'],
                'eval_f1': eval_result['eval_f1'],
                'eval_precision': eval_result['eval_precision'],
                'eval_recall': eval_result['eval_recall'],
                'epochs_trained': train_result.global_step / len(train_dataset)
            })
            
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'experiment_results.csv'), index=False)
    
    # Plot learning curves
    plot_learning_curves(results_df, output_dir)

def plot_learning_curves(results_df, output_dir):
    """Plot learning curves for different metrics."""
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        
        # Calculate mean and std for each dataset size
        stats = results_df.groupby('dataset_size')[f'eval_{metric}'].agg(['mean', 'std']).reset_index()
        
        plt.errorbar(stats['dataset_size'], stats['mean'], 
                    yerr=stats['std'], capsize=5, marker='o')
        plt.fill_between(stats['dataset_size'], 
                        stats['mean'] - stats['std'],
                        stats['mean'] + stats['std'], 
                        alpha=0.2)
        
        plt.xlabel('Dataset Size (samples per class)')
        plt.ylabel(f'Validation {metric.capitalize()}')
        plt.title(f'Learning Curve - {metric.capitalize()}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()

def plot_spacing_comparison(output_dir):
    """
    Create a visual comparison of linear vs logarithmic spacing.
    """
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Generate example sizes
    total_size = 7000
    num_points = 6
    
    # Linear spacing
    linear_sizes = np.linspace(100, total_size, num_points).astype(int)
    
    # Logarithmic spacing
    log_sizes = np.logspace(
        np.log10(100),
        np.log10(total_size),
        num_points
    ).astype(int)
    
    # Plot both
    plt.subplot(1, 2, 1)
    plt.plot(range(num_points), linear_sizes, 'bo-', label='Linear')
    plt.plot(range(num_points), log_sizes, 'ro-', label='Logarithmic')
    plt.xlabel('Sample Point')
    plt.ylabel('Dataset Size')
    plt.title('Linear vs Logarithmic Spacing')
    plt.legend()
    plt.grid(True)
    
    # Plot the same data with log scale y-axis
    plt.subplot(1, 2, 2)
    plt.plot(range(num_points), linear_sizes, 'bo-', label='Linear')
    plt.plot(range(num_points), log_sizes, 'ro-', label='Logarithmic')
    plt.yscale('log')
    plt.xlabel('Sample Point')
    plt.ylabel('Dataset Size (Log Scale)')
    plt.title('Linear vs Logarithmic Spacing (Log Scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spacing_comparison.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run dataset size experiments')
    parser.add_argument('--train-dir', type=str, default='protocol_documents',
                      help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='size_experiment_results',
                      help='Directory to save experiment results')
    parser.add_argument('--sizes', type=int, nargs='+', 
                      help='List of dataset sizes to try (samples per class). If not provided, '
                           'sizes will be automatically determined based on available data')
    parser.add_argument('--max-size', type=int,
                      help='Maximum samples per class to consider. If not provided, '
                           'uses all available data')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                      help='Learning rate for training')
    parser.add_argument('--num-runs', type=int, default=3,
                      help='Number of runs per dataset size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate spacing comparison plot
    plot_spacing_comparison(args.output_dir)
    
    print("\nStarting dataset size experiments...")
    
    run_experiment(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        dataset_sizes=args.sizes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_runs=args.num_runs,
        max_size_per_class=args.max_size
    )
    
    print(f"\nExperiments completed! Results saved in {args.output_dir}/")
    print("Check 'experiment_results.csv' for detailed metrics")
    print("Check 'learning_curves.png' for visualization of results")

if __name__ == "__main__":
    main()
