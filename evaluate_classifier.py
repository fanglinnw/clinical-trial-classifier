import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from train_classifier import ProtocolDataset  # Import the dataset class from training script

def evaluate_model(model_path, test_data_dir, max_samples=None):
    """
    Evaluate the trained model on a test dataset
    
    Args:
        model_path: Path to the saved model
        test_data_dir: Directory containing test data
        max_samples: Maximum number of samples to use per class
    """
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Create test dataset
    test_dataset = ProtocolDataset(test_data_dir, 'test', max_samples)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Evaluate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Cancer', 'Cancer']))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(
        cm, 
        index=['Actual Non-Cancer', 'Actual Cancer'],
        columns=['Predicted Non-Cancer', 'Predicted Cancer']
    )
    print("\nConfusion Matrix:")
    print(cm_df)
    
    return {
        'predictions': all_preds,
        'true_labels': all_labels
    }

if __name__ == "__main__":
    MODEL_PATH = "model_output"  # Path to saved model
    TEST_DATA_DIR = "protocol_documents"  # Directory containing test data
    MAX_SAMPLES = 100  # Set to None to use all available data
    
    results = evaluate_model(MODEL_PATH, TEST_DATA_DIR, MAX_SAMPLES)
