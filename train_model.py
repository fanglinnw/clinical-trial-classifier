import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
import pandas as pd
import numpy as np
import evaluate
import logging
import fitz
import gc
from tqdm import tqdm


class ProtocolDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def extract_text_from_pdf(pdf_path, max_length=4000):
    """Extract and truncate text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        if len(text) > max_length:
            text = ' '.join(text[:max_length].split()[:-1])
        return text.strip()
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        return ""
    finally:
        if 'doc' in locals():
            doc.close()
        gc.collect()  # Help manage memory for large datasets


def prepare_split_datasets(base_dir):
    """Prepare datasets from train/val/test splits."""
    splits = ['train', 'val', 'test']
    datasets = {}
    
    for split in splits:
        data = []
        
        # Process cancer protocols
        cancer_files = list(Path(base_dir) / 'cancer' / split / '*.pdf')
        non_cancer_files = list(Path(base_dir) / 'non_cancer' / split / '*.pdf')

        # Process cancer protocols
        for pdf_path in tqdm(cancer_files, desc=f"Processing cancer protocols ({split})"):
            text = extract_text_from_pdf(pdf_path)
            if text:
                data.append({
                    'text': text,
                    'label': 1,
                    'file_name': pdf_path.name
                })
            gc.collect()  # Regular memory cleanup

        # Process non-cancer protocols
        for pdf_path in tqdm(non_cancer_files, desc=f"Processing non-cancer protocols ({split})"):
            text = extract_text_from_pdf(pdf_path)
            if text:
                data.append({
                    'text': text,
                    'label': 0,
                    'file_name': pdf_path.name
                })
            gc.collect()  # Regular memory cleanup

        datasets[split] = pd.DataFrame(data)
        print(f"{split} set size: {len(datasets[split])} protocols")
        print(f"Class distribution in {split} set:")
        print(datasets[split]['label'].value_counts())
        
        gc.collect()  # Clean up after creating DataFrame
    
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


def train_model():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize tokenizer and model
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Prepare datasets
    logging.info("Preparing datasets...")
    datasets = prepare_split_datasets(base_dir='./protocol_documents')

    # Create datasets
    train_dataset = ProtocolDataset(
        texts=datasets['train']['text'].tolist(),
        labels=datasets['train']['label'].tolist(),
        tokenizer=tokenizer
    )

    val_dataset = ProtocolDataset(
        texts=datasets['val']['text'].tolist(),
        labels=datasets['val']['label'].tolist(),
        tokenizer=tokenizer
    )

    test_dataset = ProtocolDataset(
        texts=datasets['test']['text'].tolist(),
        labels=datasets['test']['label'].tolist(),
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
    return val_results, test_results


if __name__ == "__main__":
    val_results, test_results = train_model()
    
    print("\nValidation Results:")
    for key, value in val_results.items():
        print(f"{key}: {value}")
        
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"{key}: {value}")
