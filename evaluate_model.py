import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

def evaluate_zero_shot(texts, device):
    """Evaluate using zero-shot classification with RoBERTa."""
    # Determine device for pipeline
    if device.type == 'cuda':
        pipeline_device = 0  # Use CUDA
    elif device.type == 'mps':
        try:
            # Try to use MPS
            pipeline_device = 'mps'
        except Exception as e:
            print(f"Warning: MPS device detected but encountered error: {e}")
            print("Falling back to CPU for zero-shot classification")
            pipeline_device = -1
    else:
        pipeline_device = -1  # Use CPU

    print(f"\nInitializing zero-shot classifier on device: {pipeline_device}")
    
    try:
        classifier = pipeline("zero-shot-classification",
                            model="roberta-large-mnli",
                            device=pipeline_device)
    except Exception as e:
        print(f"Error initializing pipeline on {pipeline_device}, falling back to CPU: {e}")
        classifier = pipeline("zero-shot-classification",
                            model="roberta-large-mnli",
                            device=-1)
    
    candidate_labels = ["clinical trial not related to cancer research", "clinical trial focused on cancer research"]
    hypothesis_template = "This clinical trial protocol describes a {}."
    
    all_preds = []
    all_probs = []
    
    for text in texts:
        result = classifier(text, 
                          candidate_labels, 
                          multi_label=False,
                          hypothesis_template=hypothesis_template)
        pred_label = 1 if result['labels'][0] == "clinical trial focused on cancer research" else 0
        prob = result['scores'][0]  # probability of the highest scoring label
        all_preds.append(pred_label)
        all_probs.append([1 - prob, prob] if pred_label == 1 else [prob, 1 - prob])
    
    return np.array(all_preds), np.array(all_probs)

def calculate_binary_metrics(y_true, y_pred):
    """Calculate binary classification metrics treating cancer (1) as positive class."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics_str = (
        f"Binary Classification Metrics (Cancer as Positive Class):\n"
        f"{'='*50}\n"
        f"Accuracy:  {accuracy:.1%} ({tp + tn}/{len(y_true)} correct classifications)\n"
        f"Precision: {precision:.1%} ({tp} true positives out of {tp + fp} predicted cancer protocols)\n"
        f"Recall:    {recall:.1%} ({tp} true positives out of {tp + fn} actual cancer protocols)\n"
        f"F1-Score:  {f1:.1%} (harmonic mean of precision and recall)\n\n"
        f"Confusion Matrix:\n"
        f"{'='*50}\n"
        f"True Negatives (TN): {tn} | False Positives (FP): {fp}\n"
        f"False Negatives (FN): {fn} | True Positives (TP): {tp}\n"
    )
    
    return metrics_str, cm

def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label (0: Non-Cancer, 1: Cancer)')
    plt.xlabel('Predicted Label (0: Non-Cancer, 1: Cancer)')
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--test-dir', type=str, default='protocol_documents_test',
                      help='Directory containing test data (default: protocol_documents_test)')
    parser.add_argument('--model-dir', type=str, default='./final_model',
                      help='Directory containing the trained model (default: ./final_model)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results (default: evaluation_results)')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for evaluation (default: 8)')
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
        batch_size = args.batch_size * 2
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple M1 GPU")
        batch_size = args.batch_size * 2
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

    # Evaluate trained model
    print("\nEvaluating fine-tuned model...")
    predictions, true_labels, file_paths, probabilities = evaluate_model(model, test_dataset, device, batch_size)

    # Evaluate zero-shot model
    print("\nEvaluating zero-shot classifier...")
    zero_shot_predictions, zero_shot_probabilities = evaluate_zero_shot(test_texts, device)

    # Calculate and save metrics for both models
    print("\nFine-tuned Model Results:")
    finetuned_metrics, cm_finetuned = calculate_binary_metrics(true_labels, predictions)
    print(finetuned_metrics)

    print("\nZero-shot Model Results:")
    zeroshot_metrics, cm_zeroshot = calculate_binary_metrics(test_labels, zero_shot_predictions)
    print(zeroshot_metrics)

    # Save metrics to file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Fine-tuned Model Results:\n")
        f.write(finetuned_metrics)
        f.write("\nZero-shot Model Results:\n")
        f.write(zeroshot_metrics)

    # Save confusion matrices
    plot_confusion_matrix(cm_finetuned, os.path.join(args.output_dir, 'confusion_matrix_finetuned.png'))
    plot_confusion_matrix(cm_zeroshot, os.path.join(args.output_dir, 'confusion_matrix_zeroshot.png'))

    # Save detailed predictions to CSV
    results_df = pd.DataFrame({
        'file_path': file_paths,
        'true_label': ['Cancer' if label == 1 else 'Non-Cancer' for label in true_labels],
        'finetuned_predicted': ['Cancer' if pred == 1 else 'Non-Cancer' for pred in predictions],
        'zeroshot_predicted': ['Cancer' if pred == 1 else 'Non-Cancer' for pred in zero_shot_predictions],
        'finetuned_confidence_non_cancer': probabilities[:, 0],
        'finetuned_confidence_cancer': probabilities[:, 1],
        'zeroshot_confidence_non_cancer': zero_shot_probabilities[:, 0],
        'zeroshot_confidence_cancer': zero_shot_probabilities[:, 1],
        'finetuned_correct': true_labels == predictions,
        'zeroshot_correct': test_labels == zero_shot_predictions
    })
    
    csv_path = os.path.join(args.output_dir, 'detailed_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed predictions saved to: {csv_path}")
    
    # Print summary of incorrect predictions
    incorrect_predictions = results_df[~results_df['finetuned_correct']].copy()
    if len(incorrect_predictions) > 0:
        print("\nSummary of incorrect predictions:")
        print(f"Total incorrect predictions: {len(incorrect_predictions)}")
        print("\nSample of incorrect predictions:")
        for _, row in incorrect_predictions.head().iterrows():
            print(f"\nFile: {os.path.basename(row['file_path'])}")
            print(f"True label: {row['true_label']}")
            print(f"Predicted: {row['finetuned_predicted']}")
            print(f"Confidence: {row['finetuned_confidence_cancer']:.2f}")

if __name__ == "__main__":
    main()
