import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import matplotlib.pyplot as plt
from utils import ProtocolDataset, load_dataset, preprocess_text
from sklearn.model_selection import train_test_split
from train_classifier import compute_metrics
import pandas as pd

def train_with_subset(train_dataset, eval_dataset, train_size, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
    """Train model with a subset of training data and return evaluation metrics."""
    # Take a subset of training data
    subset_size = int(len(train_dataset) * train_size)
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    subset_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./temp_model_{train_size}",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"./logs_{train_size}",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=subset_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    
    return eval_results

def analyze_learning_curve(dataset_path="data/processed"):
    """Analyze and plot learning curves with different training dataset sizes."""
    # Load and prepare dataset
    full_dataset = load_dataset(dataset_path)
    train_dataset, eval_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
    
    # Different proportions of training data to try
    train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []
    
    print("Starting learning curve analysis...")
    for size in train_sizes:
        print(f"\nTraining with {size*100}% of training data...")
        metrics = train_with_subset(train_dataset, eval_dataset, size)
        results.append({
            'train_size': size,
            'num_samples': int(len(train_dataset) * size),
            'accuracy': metrics['eval_accuracy'],
            'f1': metrics['eval_f1'],
            'precision': metrics['eval_precision'],
            'recall': metrics['eval_recall']
        })
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Plot learning curves
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    for metric in metrics:
        plt.plot(df_results['num_samples'], df_results[metric], marker='o', label=metric)
    
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Score')
    plt.title('Learning Curves: Model Performance vs Training Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    plt.close()
    
    # Save numerical results
    df_results.to_csv('learning_curve_results.csv', index=False)
    
    # Analyze point of diminishing returns
    diffs = df_results['f1'].diff() / df_results['train_size'].diff()
    min_improvement_threshold = 0.05  # minimum improvement in F1 score per 10% increase in data
    optimal_idx = np.where(diffs < min_improvement_threshold)[0]
    optimal_size = train_sizes[optimal_idx[0]] if len(optimal_idx) > 0 else 1.0
    
    print("\nLearning Curve Analysis Results:")
    print("--------------------------------")
    print(f"Optimal training dataset size: {optimal_size*100:.0f}% of full dataset")
    print(f"This corresponds to approximately {int(len(train_dataset) * optimal_size)} samples")
    print("\nResults saved to 'learning_curves.png' and 'learning_curve_results.csv'")
    
    return optimal_size

if __name__ == "__main__":
    analyze_learning_curve()
