import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from utils import load_dataset
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def setup_llm_pipeline():
    """Set up local LLM pipeline using Mistral."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Latest Mistral instruct model
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision for memory efficiency
        device_map="auto"  # Automatically handle model placement on GPU
    )
    
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=False
    )

def classify_with_llm(pipe, text):
    """Classify a single text using local LLM."""
    prompt = f"""<s>[INST] You are a clinical trial document classifier. Your task is to determine if a clinical trial protocol is cancer-related or not.
    Please classify the following clinical trial protocol text as either cancer-related or not.
    Focus on whether the primary purpose or main endpoints of the trial are related to cancer research, treatment, or prevention.
    Respond with either "CANCER" or "NON-CANCER" followed by a brief explanation.

    Here is the protocol text:
    {text[:2000]}  # Limiting text length to manage memory
    [/INST]"""
    
    try:
        response = pipe(prompt)[0]['generated_text']
        # Extract the part after the prompt
        response_text = response[len(prompt):].strip()
        # Extract the first line and clean it up
        first_line = response_text.split('\n')[0].strip().upper()
        # Check for exact matches
        if first_line == "CANCER":
            classification = "CANCER"
        elif first_line == "NON-CANCER":
            classification = "NON-CANCER"
        else:
            # If no exact match, look for the label within the first line
            if "NON-CANCER" in first_line:
                classification = "NON-CANCER"
            elif "CANCER" in first_line and "NON-CANCER" not in first_line:
                classification = "CANCER"
            else:
                # Default to None if we can't determine the classification
                return None, "Could not determine classification from response: " + first_line
        return classification, response_text
        
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return None, str(e)

def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('LLM Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Classify clinical trial protocols using local LLM')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing the protocol documents')
    parser.add_argument('--output-dir', type=str, default='llm_results',
                      help='Directory to save classification results')
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode with smaller dataset')
    parser.add_argument('--debug-samples', type=int, default=5,
                      help='Number of samples per class to use in debug mode')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for processing')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up LLM pipeline
    pipe = setup_llm_pipeline()
    
    # Load dataset
    print("Loading dataset...")
    texts, labels, paths = load_dataset(args.data_dir, debug=args.debug, debug_samples=args.debug_samples)
    
    # Classify documents
    print("\nClassifying documents using local LLM...")
    results = []
    for text, label, path in tqdm(zip(texts, labels, paths), total=len(texts)):
        classification, explanation = classify_with_llm(pipe, text)
        if classification:
            results.append({
                'file_path': path,
                'true_label': 'CANCER' if label == 1 else 'NON-CANCER',
                'predicted_label': classification,
                'explanation': explanation,
                'correct': (classification == 'CANCER') == (label == 1)
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    y_true = [1 if label == 'CANCER' else 0 for label in results_df['true_label']]
    y_pred = [1 if pred == 'CANCER' else 0 for pred in results_df['predicted_label']]
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=['Non-Cancer', 'Cancer'])
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(args.output_dir, 'llm_classification_report.txt'), 'w') as f:
        f.write("Local LLM Classification Results:\n")
        f.write("="*50 + "\n")
        f.write(report)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, os.path.join(args.output_dir, 'llm_confusion_matrix.png'))
    
    # Save detailed results to CSV
    csv_path = os.path.join(args.output_dir, 'llm_detailed_predictions.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed predictions saved to: {csv_path}")
    
    # Print summary of incorrect predictions
    incorrect_predictions = results_df[~results_df['correct']].copy()
    if len(incorrect_predictions) > 0:
        print("\nSummary of incorrect predictions:")
        print(f"Total incorrect predictions: {len(incorrect_predictions)}")
        print("\nSample of incorrect predictions:")
        for _, row in incorrect_predictions.head().iterrows():
            print(f"\nFile: {os.path.basename(row['file_path'])}")
            print(f"True label: {row['true_label']}")
            print(f"Predicted: {row['predicted_label']}")
            print(f"Explanation: {row['explanation']}")

if __name__ == "__main__":
    main()
