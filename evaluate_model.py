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
import pandas as pd

def evaluate_model(model, dataset, device, batch_size=8):
    """Evaluate model on a dataset and return predictions and true labels."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_paths = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            file_paths = batch['file_path']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_paths.extend(file_paths)
            all_probs.extend(probs)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_paths), np.array(all_probs)

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
    test_texts, test_labels, test_paths = load_dataset(args.test_dir, debug=args.debug, debug_samples=args.debug_samples)
    test_dataset = ProtocolDataset(test_texts, test_labels, tokenizer, file_paths=test_paths)

    # Evaluate model
    print("Evaluating model...")
    predictions, true_labels, file_paths, probabilities = evaluate_model(model, test_dataset, device, batch_size)

    # Calculate and save metrics
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, target_names=['Non-Cancer', 'Cancer'])
    print(report)

    # Save classification report
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Save confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm, os.path.join(args.output_dir, 'confusion_matrix.png'))

    # Save detailed predictions to CSV
    results_df = pd.DataFrame({
        'file_path': file_paths,
        'true_label': ['Cancer' if label == 1 else 'Non-Cancer' for label in true_labels],
        'predicted_label': ['Cancer' if pred == 1 else 'Non-Cancer' for pred in predictions],
        'confidence_non_cancer': probabilities[:, 0],
        'confidence_cancer': probabilities[:, 1],
        'correct_prediction': true_labels == predictions
    })
    
    csv_path = os.path.join(args.output_dir, 'detailed_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed predictions saved to: {csv_path}")
    
    # Print summary of incorrect predictions
    incorrect_predictions = results_df[~results_df['correct_prediction']].copy()  # Create an explicit copy
    if len(incorrect_predictions) > 0:
        print("\nSummary of incorrect predictions:")
        print(f"Total incorrect predictions: {len(incorrect_predictions)}")
        print("\nTop 5 most confident incorrect predictions:")
        
        # Calculate the maximum confidence for each prediction
        incorrect_predictions.loc[:, 'max_confidence'] = incorrect_predictions.apply(
            lambda x: max(x['confidence_cancer'], x['confidence_non_cancer']), axis=1
        )
        
        # Show top 5 by confidence
        for _, row in incorrect_predictions.nlargest(5, 'max_confidence').iterrows():
            print(f"\nFile: {os.path.basename(row['file_path'])}")
            print(f"True label: {row['true_label']}")
            print(f"Predicted: {row['predicted_label']} (confidence: {row['max_confidence']:.2%})")
            
        # Add analysis of incorrect predictions
        cancer_as_non = len(incorrect_predictions[
            (incorrect_predictions['true_label'] == 'Cancer') & 
            (incorrect_predictions['predicted_label'] == 'Non-Cancer')
        ])
        non_as_cancer = len(incorrect_predictions[
            (incorrect_predictions['true_label'] == 'Non-Cancer') & 
            (incorrect_predictions['predicted_label'] == 'Cancer')
        ])
        
        print("\nBreakdown of incorrect predictions:")
        print(f"Cancer protocols classified as Non-Cancer: {cancer_as_non}")
        print(f"Non-Cancer protocols classified as Cancer: {non_as_cancer}")
    
    print(f"\nResults saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
