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

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from models.bert_classifier import BERTClassifier
from models.baseline_classifiers import BaselineClassifiers
from utils.text_extractor import ProtocolTextExtractor

def load_data_from_directory(base_dir: str, extractor: ProtocolTextExtractor, split: str = "train") -> pd.DataFrame:
    """
    Load data from protocol_documents directory structure for a specific split (train/val/test).
    
    Args:
        base_dir: Base directory containing cancer and non_cancer subdirectories
        extractor: Text extractor instance
        split: Which split to load ('train', 'val', or 'test')
    
    Returns:
        DataFrame containing the data for the specified split
    """
    logger = logging.getLogger(__name__)
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
    
    return df

def train_baseline_models(
    train_data: pd.DataFrame,
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
        
        return {
            "model_path": output_dir,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error training baseline models: {e}")
        return {
            "error": str(e)
        }

def train_bert_model(
    model_type: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
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
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return {
        "train_loss": train_result.training_loss,
        "train_steps": train_result.global_step,
        "model_path": output_dir
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
    extractor = ProtocolTextExtractor(max_length=args.max_text_length)
    
    # Load train and validation data
    train_data = load_data_from_directory(args.data_dir, extractor, "train")
    val_data = load_data_from_directory(args.data_dir, extractor, "val")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train baseline models
    logger.info("Training baseline models...")
    baseline_output_dir = os.path.join(args.output_dir, "baseline")
    os.makedirs(baseline_output_dir, exist_ok=True)
    
    results = {
        "baseline": train_baseline_models(
            train_data=train_data,
            output_dir=baseline_output_dir,
            max_length=args.max_text_length
        )
    }
    
    # Train BERT models
    model_types = ['biobert', 'clinicalbert', 'pubmedbert']
    
    for model_type in model_types:
        model_output_dir = os.path.join(args.output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        try:
            results[model_type] = train_bert_model(
                model_type=model_type,
                train_data=train_data,
                val_data=val_data,
                output_dir=model_output_dir,
                max_length=args.max_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                warmup_ratio=args.warmup_ratio,
                weight_decay=args.weight_decay,
                early_stopping_patience=args.early_stopping_patience
            )
            logger.info(f"Successfully trained {model_type}")
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            results[model_type] = {"error": str(e)}
    
    # Save training results
    with open(os.path.join(args.output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()