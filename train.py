import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import fitz  # PyMuPDF for PDF processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProtocolDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length=512):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load cancer and non-cancer documents
        self.cancer_dir = self.data_dir / 'cancer' / split
        self.non_cancer_dir = self.data_dir / 'non_cancer' / split
        
        self.cancer_files = list(self.cancer_dir.glob('*.pdf'))
        self.non_cancer_files = list(self.non_cancer_dir.glob('*.pdf'))
        
        self.files = [(f, 1) for f in self.cancer_files] + [(f, 0) for f in self.non_cancer_files]
    
    def __len__(self):
        return len(self.files)
    
    def extract_text_from_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text[:100000]  # Limit text length to prevent memory issues
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return ""
    
    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        text = self.extract_text_from_pdf(file_path)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, device, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
        
        logger.info(f'Epoch {epoch + 1}:')
        logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_save_path = os.path.join('model_output', 'best_model')
            model.save_pretrained(model_save_path)
            logger.info(f'Saved best model with F1: {val_f1:.4f}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load tokenizer and model
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    model.to(device)
    
    # Create datasets
    train_dataset = ProtocolDataset('protocol_documents', 'train', tokenizer)
    val_dataset = ProtocolDataset('protocol_documents', 'val', tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Train model
    train_model(model, train_loader, val_loader, device)

if __name__ == '__main__':
    main()
