import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
from tqdm import tqdm
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
        cancer_files = list(Path(base_dir) / 'cancer' / split / '*.pdf')
        for pdf_path in tqdm(cancer_files, desc=f"Processing cancer protocols ({split})"):
            result = extractor.extract_from_pdf(pdf_path)
            if result["full_text"]:
                data.append({
                    'text': result["full_text"],
                    'label': 1,
                    'file_name': pdf_path.name
                })

        # Process non-cancer protocols
        non_cancer_files = list(Path(base_dir) / 'non_cancer' / split / '*.pdf')
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
        'precision': metrics['precision'].compute(predictions=predictions, references=labels, average='binary')['precision'],
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
    datasets = prepare_datasets(base_dir, extractor)

    # Create datasets
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

    # Set up training arguments optimized for larger dataset
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
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
        fp16=True,  # Mixed precision training
        gradient_checkpointing=True,  # Memory optimization
        save_total_limit=2,  # Keep only the last 2 checkpoints
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
    with open('./protocol_classifier/eval_results.txt', 'w') as f:
        f.write("Validation Results:\n")
        for key, value in val_results.items():
            f.write(f"{key}: {value}\n")
        f.write("\nTest Results:\n")
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")

    logging.info("Training completed!")

    print("\nValidation Results:")
    for key, value in val_results.items():
        print(f"{key}: {value}")

    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"{key}: {value}")


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