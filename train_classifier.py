import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from utils import ProtocolDataset, load_dataset, preprocess_text, read_pdf
import argparse

def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description='Train a PubMedBERT model for protocol classification')
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode with smaller dataset and fewer epochs')
    parser.add_argument('--debug-samples', type=int, default=5,
                      help='Number of samples per class to use in debug mode')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs (default: 5)')
    parser.add_argument('--output-dir', type=str, default='./final_model',
                      help='Directory to save the final model')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                      help='Learning rate for training')
    args = parser.parse_args()

    if args.debug:
        print("\nRunning in DEBUG mode")
        print(f"Using {args.debug_samples} samples per class")
        print("Using 2 epochs for quick testing")
        
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

    # Initialize tokenizer and model
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Load and split dataset
    texts, labels, _ = load_dataset('protocol_documents', debug=args.debug, debug_samples=args.debug_samples)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create datasets
    train_dataset = ProtocolDataset(train_texts, train_labels, tokenizer)
    val_dataset = ProtocolDataset(val_texts, val_labels, tokenizer)

    # Define training arguments with device-specific settings
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2 if args.debug else args.epochs,
        per_device_train_batch_size=8 if device.type == 'cpu' else 16,
        per_device_eval_batch_size=8 if device.type == 'cpu' else 16,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir='./logs',
        logging_steps=1 if args.debug else 10,
        fp16=device.type == 'cuda',
        fp16_full_eval=False
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.01
    )
    
    model = model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
