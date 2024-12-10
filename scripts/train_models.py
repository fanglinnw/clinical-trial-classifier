import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from models.bert_classifier import BERTClassifier
from models.baseline_classifiers import BaselineClassifiers
from utils.text_extractor import get_extractor

def load_data_from_directory(base_dir: str, extractor, split: str = "train", use_cache: bool = True) -> pd.DataFrame:
    """
    Load data from protocol_documents directory structure for a specific split (train/val/test).
    Uses caching to avoid re-processing PDFs.
    
    Args:
        base_dir: Base directory containing cancer and non_cancer subdirectories
        extractor: Text extractor instance
        split: Which split to load ('train', 'val', or 'test')
        use_cache: Whether to use caching or not
    
    Returns:
        DataFrame containing the data for the specified split
    """
    logger = logging.getLogger(__name__)
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(base_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(cache_dir, f'{split}_data.pkl')
    
    # Try to load from cache first
    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading {split} data from cache...")
        try:
            df = pd.read_pickle(cache_file)
            logger.info(f"Successfully loaded {len(df)} samples from cache")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache, will process PDFs: {e}")
    
    data = []
    
    # Process cancer protocols
    cancer_dir = Path(base_dir) / 'cancer' / split
    cancer_files = list(cancer_dir.glob('*.pdf'))
    logger.info(f"Found {len(cancer_files)} cancer protocol PDFs in {split} set")
    
    for pdf_path in tqdm(cancer_files, desc=f"Processing cancer protocols ({split})"):
        result = extractor.extract_from_pdf(pdf_path)
        if result["full_text"]:
            data.append({
                'text': result["full_text"],
                'label': 1,  # cancer
                'file_name': pdf_path.name
            })
    
    # Process non-cancer protocols
    non_cancer_dir = Path(base_dir) / 'non_cancer' / split
    non_cancer_files = list(non_cancer_dir.glob('*.pdf'))
    logger.info(f"Found {len(non_cancer_files)} non-cancer protocol PDFs in {split} set")
    
    for pdf_path in tqdm(non_cancer_files, desc=f"Processing non-cancer protocols ({split})"):
        result = extractor.extract_from_pdf(pdf_path)
        if result["full_text"]:
            data.append({
                'text': result["full_text"],
                'label': 0,  # non-cancer
                'file_name': pdf_path.name
            })
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"{split.capitalize()} set size: {len(df)}")
    
    # Save to cache
    if use_cache:
        try:
            logger.info(f"Saving {split} data to cache...")
            df.to_pickle(cache_file)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    return df

def train_baseline_models(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: str,
    max_length: int = 8000
) -> Dict[str, Any]:
    """Train baseline models including traditional ML and zero-shot."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize baseline classifiers
        baseline = BaselineClassifiers(model_dir=output_dir, max_length=max_length)
        
        # Train models with the data
        texts = train_data["text"].tolist()
        labels = train_data["label"].tolist()
        
        baseline.train_with_data(texts, labels)
        
        # Evaluate on test set
        logger.info("\nEvaluating baseline models on test set...")
        
        # Process test data in batches
        batch_size = 32
        predictions = []
        test_texts = test_data["text"].tolist()
        
        for i in tqdm(range(0, len(test_texts), batch_size), desc="Evaluating on test set"):
            batch_texts = test_texts[i:i + batch_size]
            batch_results = baseline.classify_text_batch(batch_texts)
            
            # Extract predictions using logistic regression as primary classifier
            for result in batch_results:
                pred = 1 if result['traditional_ml']['log_reg_prediction'] == 'cancer' else 0
                predictions.append(pred)
        
        accuracy = accuracy_score(test_data['label'].tolist(), predictions) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(test_data['label'].tolist(), predictions, average='binary')
        
        metrics = {
            'accuracy': round(accuracy, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2)
        }
        
        logger.info("\nTest Set Metrics:")
        logger.info("─" * 40)
        for metric, value in metrics.items():
            logger.info(f"{metric.capitalize():12} : {value:>6.2f}%")
        logger.info("─" * 40)
        
        return {
            "model_path": output_dir,
            "status": "success",
            "test_metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error training baseline models: {e}")
        return {
            "error": str(e)
        }

def evaluate_model(model, test_data: pd.DataFrame) -> Dict[str, float]:
    """Evaluate model on test data and return metrics."""
    logger = logging.getLogger(__name__)
    
    predictions = []
    labels = test_data['label'].tolist()
    texts = test_data['text'].tolist()
    
    # Process in batches
    batch_size = 32
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating on test set"):
        batch_texts = texts[i:i + batch_size]
        
        if isinstance(model, BERTClassifier):
            # BERT models handle batching internally
            results = [model.classify_text(text) for text in batch_texts]
            batch_predictions = [1 if result['classification'] == 'cancer' else 0 
                               for result in results]
        else:
            # For baseline models, use the batch classification
            results = model.classify_text_batch(batch_texts)
            batch_predictions = [1 if result['traditional_ml']['log_reg_prediction'] == 'cancer' 
                               else 0 for result in results]
        
        predictions.extend(batch_predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    metrics = {
        'accuracy': round(accuracy, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1': round(f1 * 100, 2)
    }
    
    # Print metrics in a nice format
    logger.info("\nTest Set Metrics:")
    logger.info("─" * 40)
    for metric, value in metrics.items():
        logger.info(f"{metric.capitalize():12} : {value:>6.2f}%")
    logger.info("─" * 40)
    
    return metrics

def train_bert_model(
    model_type: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: str,
    max_length: int = 512,
    batch_size: int = 4,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 3
) -> Dict[str, Any]:
    """Train a BERT model for protocol classification."""
    logger = logging.getLogger(__name__)
    
    # Initialize model and tokenizer
    model_name = BERTClassifier.MODEL_PATHS[model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # Tokenize data
    def tokenize_data(texts):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    train_encodings = tokenize_data(train_data["text"].tolist())
    val_encodings = tokenize_data(val_data["text"].tolist())

    # Create dataset class
    class ProtocolDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ProtocolDataset(train_encodings, train_data["label"].tolist())
    val_dataset = ProtocolDataset(val_encodings, val_data["label"].tolist())

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )

    # Train the model
    logger.info(f"Starting {model_type.upper()} training...")
    train_result = trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Load the trained model for evaluation
    trained_model = BERTClassifier(model_type=model_type, model_path=output_dir)
    
    # Evaluate on test set
    logger.info("\nEvaluating model on test set...")
    test_metrics = evaluate_model(trained_model, test_data)
    
    return {
        "model_path": output_dir,
        "training_args": training_args.to_dict(),
        "test_metrics": test_metrics,
        "train_loss": train_result.training_loss,
        "train_steps": train_result.global_step,
        "eval_results": eval_result,
        "status": "success"
    }

def main():
    parser = argparse.ArgumentParser(description="Train BERT models for protocol classification")
    parser.add_argument("--data-dir", default="./protocol_documents", help="Directory containing protocol documents")
    parser.add_argument("--output-dir", default="./trained_models", help="Directory to save models")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for BERT models")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-text-length", type=int, default=8000, help="Maximum text length for extraction")
    parser.add_argument("--model", choices=['biobert', 'clinicalbert', 'pubmedbert', 'baseline'],
                      help="Train only a specific model. If not provided, all models will be trained.")
    parser.add_argument("--no-cache", action="store_true", help="Force re-processing of PDFs without using cache")
    parser.add_argument("--extractor-type", choices=['simple', 'sections'], default='simple',
                      help="Type of text extractor to use (default: simple)")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Set random seed
    torch.manual_seed(args.seed)
    
    # Initialize text extractor
    extractor = get_extractor(args.extractor_type)
    
    # Load train, validation and test data
    train_data = load_data_from_directory(args.data_dir, extractor, "train", not args.no_cache)
    val_data = load_data_from_directory(args.data_dir, extractor, "val", not args.no_cache)
    test_data = load_data_from_directory(args.data_dir, extractor, "test", not args.no_cache)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    # Train only specified model or all models
    if args.model == 'baseline':
        # Train baseline models
        logger.info("Training baseline models...")
        baseline_output_dir = os.path.join(args.output_dir, "baseline")
        os.makedirs(baseline_output_dir, exist_ok=True)
        
        results["baseline"] = train_baseline_models(
            train_data=train_data,
            test_data=test_data,
            output_dir=baseline_output_dir,
            max_length=args.max_text_length
        )
    elif args.model in ['biobert', 'clinicalbert', 'pubmedbert']:
        # Train specific BERT model
        logger.info(f"\nTraining {args.model.upper()}...")
        model_output_dir = os.path.join(args.output_dir, args.model)
        os.makedirs(model_output_dir, exist_ok=True)
        
        try:
            results[args.model] = train_bert_model(
                model_type=args.model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                output_dir=model_output_dir,
                max_length=args.max_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                warmup_ratio=args.warmup_ratio,
                weight_decay=args.weight_decay,
                early_stopping_patience=args.early_stopping_patience
            )
        except Exception as e:
            logger.error(f"Error training {args.model}: {e}")
            results[args.model] = {"error": str(e)}
    else:
        # Train all models
        # Train baseline models
        logger.info("Training baseline models...")
        baseline_output_dir = os.path.join(args.output_dir, "baseline")
        os.makedirs(baseline_output_dir, exist_ok=True)
        
        results["baseline"] = train_baseline_models(
            train_data=train_data,
            test_data=test_data,
            output_dir=baseline_output_dir,
            max_length=args.max_text_length
        )
        
        # Train all BERT models
        model_types = ['biobert', 'clinicalbert', 'pubmedbert']
        for model_type in model_types:
            logger.info(f"\nTraining {model_type.upper()}...")
            model_output_dir = os.path.join(args.output_dir, model_type)
            os.makedirs(model_output_dir, exist_ok=True)
            
            try:
                results[model_type] = train_bert_model(
                    model_type=model_type,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    output_dir=model_output_dir,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    warmup_ratio=args.warmup_ratio,
                    weight_decay=args.weight_decay,
                    early_stopping_patience=args.early_stopping_patience
                )
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                results[model_type] = {"error": str(e)}
    
    # Save training results
    results_file = os.path.join(args.output_dir, "training_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info("\nTraining complete. Results saved to training_results.json")

if __name__ == "__main__":
    main()