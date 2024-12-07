import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
from tqdm import tqdm
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.text_extractor import ProtocolTextExtractor
from models.pubmedbert_classifier import ProtocolDataset
from models.baseline_classifiers import BaselineClassifiers


def prepare_datasets(base_dir: str, extractor: ProtocolTextExtractor):
    """Prepare datasets from train/val/test splits."""
    splits = ['train', 'val', 'test']
    datasets = {}

    for split in splits:
        data = []

        # Process cancer protocols
        cancer_dir = Path(base_dir) / 'cancer' / split
        cancer_files = list(cancer_dir.glob('*.pdf'))
        for pdf_path in tqdm(cancer_files, desc=f"Processing cancer protocols ({split})"):
            result = extractor.extract_from_pdf(pdf_path)
            if result["full_text"]:
                data.append({
                    'text': result["full_text"],
                    'label': 1,
                    'file_name': pdf_path.name
                })

        # Process non-cancer protocols
        non_cancer_dir = Path(base_dir) / 'non_cancer' / split
        non_cancer_files = list(non_cancer_dir.glob('*.pdf'))
        for pdf_path in tqdm(non_cancer_files, desc=f"Processing non-cancer protocols ({split})"):
            result = extractor.extract_from_pdf(pdf_path)
            if result["full_text"]:
                data.append({
                    'text': result["full_text"],
                    'label': 0,
                    'file_name': pdf_path.name
                })

        datasets[split] = data
        print(f"{split} set size: {len(datasets[split])} protocols")
        print(f"Class distribution in {split} set:")
        labels = [d['label'] for d in data]
        print(f"Cancer: {sum(labels)}, Non-cancer: {len(labels) - sum(labels)}")

    return datasets


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    metrics = {
        'accuracy': evaluate.load('accuracy'),
        'f1': evaluate.load('f1'),
        'precision': evaluate.load('precision'),
        'recall': evaluate.load('recall')
    }

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        'accuracy': metrics['accuracy'].compute(predictions=predictions, references=labels)['accuracy'],
        'f1': metrics['f1'].compute(predictions=predictions, references=labels, average='binary')['f1'],
        'precision': metrics['precision'].compute(predictions=predictions, references=labels, average='binary')[
            'precision'],
        'recall': metrics['recall'].compute(predictions=predictions, references=labels, average='binary')['recall']
    }


def train_pubmedbert(base_dir: str, extractor: ProtocolTextExtractor):
    """Train PubMedBERT classifier."""
    logging.info("Training PubMedBERT classifier...")

    # Initialize tokenizer and model
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Prepare datasets
    logging.info("Preparing datasets...")
    datasets = prepare_datasets(base_dir, extractor)

    train_dataset = ProtocolDataset(
        texts=[d['text'] for d in datasets['train']],
        labels=[d['label'] for d in datasets['train']],
        tokenizer=tokenizer
    )

    val_dataset = ProtocolDataset(
        texts=[d['text'] for d in datasets['val']],
        labels=[d['label'] for d in datasets['val']],
        tokenizer=tokenizer
    )

    test_dataset = ProtocolDataset(
        texts=[d['text'] for d in datasets['test']],
        labels=[d['label'] for d in datasets['test']],
        tokenizer=tokenizer
    )

    # Detect device type
    device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard",
        # Only enable fp16 on CUDA devices
        fp16=device_type == 'cuda',
        gradient_checkpointing=True,
        save_total_limit=2,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logging.info("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model("./protocol_classifier")
    tokenizer.save_pretrained("./protocol_classifier")

    # Evaluate on validation set
    logging.info("Evaluating on validation set...")
    val_results = trainer.evaluate()

    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)

    # Save evaluation results
    os.makedirs("./protocol_classifier", exist_ok=True)
    with open('./protocol_classifier/eval_results.txt', 'w') as f:
        f.write("Validation Results:\n")
        for key, value in val_results.items():
            f.write(f"{key}: {value}\n")
        f.write("\nTest Results:\n")
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")

    logging.info("Training completed!")
    return val_results, test_results


def train_baseline_models(base_dir: str):
    """Train baseline models."""
    logging.info("Training baseline models...")
    baseline_classifiers = BaselineClassifiers()
    baseline_classifiers.train_traditional_models(base_dir)
    return baseline_classifiers


def main():
    logging.basicConfig(level=logging.INFO)
    base_dir = "./protocol_documents"

    # Initialize text extractor
    extractor = ProtocolTextExtractor(max_length=8000)

    # Train PubMedBERT
    train_pubmedbert(base_dir, extractor)

    # Train baseline models
    train_baseline_models(base_dir)


if __name__ == "__main__":
    main()