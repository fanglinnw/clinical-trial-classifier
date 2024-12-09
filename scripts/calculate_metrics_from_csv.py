import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import sys
from tabulate import tabulate

def analyze_model_performance(df):
    """
    Analyze performance metrics for different models predicting cancer relevance.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing model predictions and actual labels
    
    Returns:
    dict: Dictionary containing performance metrics for each model
    """
    # List of models to analyze
    models = ['biobert', 'clinicalbert', 'pubmedbert', 'baseline_log_reg', 
             'baseline_svm', 'baseline_zero_shot']
    
    # Convert dataset column to binary (1 for cancer, 0 for non_cancer)
    y_true = (df['dataset'] == 'cancer').astype(int)
    
    results = {}
    
    for model in models:
        # Get predictions and confidence scores
        y_pred = (df[f'{model}_prediction'] == 'cancer').astype(int)
        confidence_scores = df[f'{model}_confidence'].values
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'avg_confidence': np.mean(confidence_scores),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Calculate confidence metrics for correct/incorrect predictions
        correct_mask = y_true == y_pred
        metrics['avg_confidence_correct'] = np.mean(confidence_scores[correct_mask]) if any(correct_mask) else 0
        metrics['avg_confidence_incorrect'] = np.mean(confidence_scores[~correct_mask]) if any(~correct_mask) else 0
        
        results[model] = metrics
    
    return results

def print_analysis(results):
    """
    Print formatted analysis results in tables.
    
    Parameters:
    results (dict): Dictionary containing performance metrics for each model
    """
    print("\nModel Performance Analysis")
    print("=" * 80)

    # Prepare data for the main metrics table
    main_metrics = []
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Avg Conf %']
    
    for model, metrics in results.items():
        main_metrics.append([
            model,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['avg_confidence']:.2f}"
        ])
    
    # Print main metrics table
    print("\nMain Performance Metrics")
    print(tabulate(main_metrics, headers=headers, tablefmt='grid'))
    
    # Prepare data for the confidence analysis table
    conf_metrics = []
    conf_headers = ['Model', 'Avg Conf (Correct) %', 'Avg Conf (Incorrect) %']
    
    for model, metrics in results.items():
        conf_metrics.append([
            model,
            f"{metrics['avg_confidence_correct']:.2f}",
            f"{metrics['avg_confidence_incorrect']:.2f}"
        ])
    
    # Print confidence analysis table
    print("\nConfidence Analysis")
    print(tabulate(conf_metrics, headers=conf_headers, tablefmt='grid'))
    
    # Print confusion matrices
    print("\nConfusion Matrices")
    print("=" * 80)
    for model, metrics in results.items():
        cm = metrics['confusion_matrix']
        cm_data = [
            ['', 'Pred Negative', 'Pred Positive'],
            ['Act Negative', cm[0][0], cm[0][1]],
            ['Act Positive', cm[1][0], cm[1][1]]
        ]
        print(f"\n{model.upper()}")
        print(tabulate(cm_data, headers='firstrow', tablefmt='grid'))

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze model performance metrics from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing model predictions')
    parser.add_argument('-o', '--output', type=str, help='Optional: Path to save results as CSV', default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Read the CSV file
        df = pd.read_csv(args.csv_file)
        
        # Analyze performance
        results = analyze_model_performance(df)
        
        # Print results
        print_analysis(results)
        
        # If output path is provided, save results to CSV
        if args.output:
            # Convert results to DataFrame for easy CSV export
            results_df = pd.DataFrame.from_dict(results, orient='index')
            # Convert confusion matrix to string to avoid nested structures
            results_df['confusion_matrix'] = results_df['confusion_matrix'].apply(str)
            results_df.to_csv(args.output)
            print(f"\nResults saved to {args.output}")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{args.csv_file}'")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{args.csv_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
