import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from dataclasses import dataclass
from typing import Dict, List, Optional
from utils.text_extractor import get_extractor

@dataclass
class TrainingConfig:
    max_length: int = 512
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    seed: int = 42

class ProtocolDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class ProtocolTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_data(self, base_dir: str, extractor_type: str = "simple") -> Dict[str, pd.DataFrame]:
        """Load and process all data at once."""
        extractor = get_extractor(extractor_type)
        data = {split: [] for split in ["train", "val", "test"]}
        
        for split in data:
            for label, class_name in [(1, "cancer"), (0, "non_cancer")]:
                dir_path = Path(base_dir) / class_name / split
                for pdf_path in dir_path.glob("*.pdf"):
                    result = extractor.extract_from_pdf(pdf_path)
                    if result["full_text"]:
                        data[split].append({
                            "text": result["full_text"],
                            "label": label,
                            "file_name": pdf_path.name
                        })
                        
            # Convert to DataFrame and shuffle
            data[split] = pd.DataFrame(data[split]).sample(frac=1, random_state=self.config.seed)
            self.logger.info(f"Loaded {len(data[split])} documents for {split} set")
            
        return data

    def train_model(
        self, 
        model_type: str,
        data: Dict[str, pd.DataFrame],
        output_dir: Path
    ) -> Dict:
        """Train a single model."""
        model_name = BERTClassifier.MODEL_PATHS[model_type]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Create datasets
        train_dataset = ProtocolDataset(
            data["train"]["text"].tolist(),
            data["train"]["label"].tolist(),
            tokenizer,
            self.config.max_length
        )
        
        val_dataset = ProtocolDataset(
            data["val"]["text"].tolist(),
            data["val"]["label"].tolist(),
            tokenizer,
            self.config.max_length
        )

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )

        # Initialize and run trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )

        self.logger.info(f"Training {model_type}...")
        train_result = trainer.train()
        eval_result = trainer.evaluate()

        # Save model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        return {
            "training_args": training_args.to_dict(),
            "train_loss": train_result.training_loss,
            "train_steps": train_result.global_step,
            "eval_results": eval_result,
            "status": "success"
        }

def main():
    parser = argparse.ArgumentParser(description="Train protocol classifiers")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", choices=["biobert", "clinicalbert", "pubmedbert"])
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    args = parser.parse_args()

    # Load config
    config = TrainingConfig()
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
            config = TrainingConfig(**config_dict)

    # Initialize trainer
    trainer = ProtocolTrainer(config)
    
    # Load all data
    data = trainer.load_data(args.data_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train models
    results = {}
    models_to_train = [args.model] if args.model else ["biobert", "clinicalbert", "pubmedbert"]
    
    for model_type in models_to_train:
        model_dir = output_dir / model_type
        model_dir.mkdir(exist_ok=True)
        
        try:
            results[model_type] = trainer.train_model(model_type, data, model_dir)
        except Exception as e:
            trainer.logger.error(f"Error training {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    # Save results
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()