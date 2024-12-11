import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import argparse
import os
from utils import ProtocolDataset, load_dataset
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dataset, device, batch_size=8):
    """Evaluate model on a dataset and return predictions and true labels."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--model-dir', type=str, default='./final_model',
                      help='Directory containing the trained model')
    parser.add_argument('--test-dir', type=str, default='protocol_documents_test',
                      help='Directory containing test data')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode with smaller dataset')
    parser.add_argument('--debug-samples', type=int, default=5,
                      help='Number of samples per class to use in debug mode')
    args = parser.parse_args()

    if args.debug:
        print("\nRunning in DEBUG mode")
        print(f"Using {args.debug_samples} samples per class")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check available hardware
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        batch_size = args.batch_size * 2  # Double batch size for GPU
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple M1 GPU")
        batch_size = args.batch_size * 2  # Double batch size for GPU
    else:
        device = torch.device('cpu')
        print("Using CPU")
        batch_size = args.batch_size

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = model.to(device)

    # Load test dataset
    print("Loading test dataset...")
    test_texts, test_labels = load_dataset(args.test_dir, debug=args.debug, debug_samples=args.debug_samples)
    test_dataset = ProtocolDataset(test_texts, test_labels, tokenizer)

    # Evaluate model
    print("Evaluating model...")
    predictions, true_labels = evaluate_model(model, test_dataset, device, batch_size)

    # Calculate and save metrics
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, target_names=['Non-Cancer', 'Cancer'])
    print(report)

    # Save classification report
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Create and save confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)

    print(f"\nResults saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
