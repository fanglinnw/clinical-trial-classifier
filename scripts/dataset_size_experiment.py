import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from train_models import train_bert_model, load_data_from_directory, BERTClassifier
from utils.text_extractor import get_extractor
from evaluate_models import ProtocolClassifierEnsemble

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_sampled_data(base_dir, extractor, split, sample_size):
    """
    Load a sampled subset of data efficiently by sampling file paths first,
    then only processing the selected files.
    """
    logger = logging.getLogger(__name__)
    data = []
    
    # Get cancer files
    cancer_dir = Path(base_dir) / 'cancer' / split
    cancer_files = list(cancer_dir.glob('*.pdf'))
    
    # Get non-cancer files
    non_cancer_dir = Path(base_dir) / 'non_cancer' / split
    non_cancer_files = list(non_cancer_dir.glob('*.pdf'))
    
    # Calculate how many files to sample from each class (50-50 split)
    samples_per_class = sample_size // 2
    
    # Sample files from each class
    sampled_cancer = pd.Series(cancer_files).sample(n=min(samples_per_class, len(cancer_files)), random_state=42)
    sampled_non_cancer = pd.Series(non_cancer_files).sample(n=min(samples_per_class, len(non_cancer_files)), random_state=42)
    
    # Process only the sampled cancer files
    logger.info(f"Processing {len(sampled_cancer)} sampled cancer protocols")
    for pdf_path in tqdm(sampled_cancer, desc=f"Processing sampled cancer protocols ({split})"):
        result = extractor.extract_from_pdf(pdf_path)
        if result["full_text"]:
            data.append({
                'text': result["full_text"],
                'label': 1,  # cancer
                'file_name': pdf_path.name
            })
    
    # Process only the sampled non-cancer files
    logger.info(f"Processing {len(sampled_non_cancer)} sampled non-cancer protocols")
    for pdf_path in tqdm(sampled_non_cancer, desc=f"Processing sampled non-cancer protocols ({split})"):
        result = extractor.extract_from_pdf(pdf_path)
        if result["full_text"]:
            data.append({
                'text': result["full_text"],
                'label': 0,  # non-cancer
                'file_name': pdf_path.name
            })
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Final sampled dataset size: {len(df)}")
    return df

def run_experiment(dataset_sizes, base_dir, output_base_dir):
    logger = setup_logging()
    extractor = get_extractor("simple")
    
    # Load validation and test data only once
    val_df = load_data_from_directory(base_dir, extractor, "val")
    test_df = load_data_from_directory(base_dir, extractor, "test")
    
    results = []
    
    for size in dataset_sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running experiment with dataset size: {size}")
        logger.info(f"{'='*50}")
        
        # Use the new efficient loading function
        sampled_train_df = load_sampled_data(base_dir, extractor, "train", size)
        
        # Create experiment-specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base_dir, f"pubmedbert_size_{size}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Train the model
        train_bert_model(
            model_type="pubmedbert",
            train_data=sampled_train_df,
            val_data=val_df,
            test_data=test_df,
            output_dir=output_dir,
            max_length=512,
            batch_size=4,
            epochs=5,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            early_stopping_patience=3
        )
        
        # Evaluate the model
        class SingleModelEnsemble(ProtocolClassifierEnsemble):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Keep only PubMedBERT model and make sure it's loaded from the correct directory
                model_path = os.path.join(output_dir)  # Use the specific experiment output directory
                self.classifiers = {
                    'pubmedbert': BERTClassifier(
                        model_type='pubmedbert',
                        model_path=model_path,
                        max_length=512,
                        extractor_type="simple"
                    )
                }

        ensemble = SingleModelEnsemble(
            trained_models_dir=output_dir,  # Use the specific experiment output directory
            max_length=512,
            extractor_type="simple"
        )
        
        # Evaluate on test set
        test_cancer_dir = "../ppp_docs"
        test_non_cancer_dir = "./protocol_documents/non_cancer/test"
        
        # Get predictions for both cancer and non-cancer
        cancer_metrics, cancer_preds = ensemble.evaluate_directory(test_cancer_dir, "cancer")
        non_cancer_metrics, non_cancer_preds = ensemble.evaluate_directory(test_non_cancer_dir, "non_cancer")
        
        # Combine predictions and true labels
        y_true = [1] * len(cancer_preds['pubmedbert']) + [0] * len(non_cancer_preds['pubmedbert'])
        y_pred = cancer_preds['pubmedbert'] + non_cancer_preds['pubmedbert']
        
        # Calculate combined metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics = {
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2)
        }
        
        # Store results
        results.append({
            "dataset_size": size,
            "output_dir": output_dir,
            "metrics": metrics
        })
        
        # Save results after each experiment
        with open(os.path.join(output_base_dir, "experiment_results.txt"), "a") as f:
            f.write(f"\nResults for dataset size {size}:\n")
            f.write(f"Metrics: {metrics}\n")
            f.write("-" * 50 + "\n")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./protocol_documents",
                      help="Base directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="./experiment_results",
                      help="Directory to save experiment results")
    args = parser.parse_args()

    # Testing a wide range of sizes from 200 to 8000
    # Using roughly logarithmic scale to get good coverage
    dataset_sizes = [200, 400, 800, 1600, 3200, 4800, 6400, 8000]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write experiment configuration
    with open(os.path.join(args.output_dir, "experiment_config.txt"), "w") as f:
        f.write("Dataset Size Experiment Configuration:\n")
        f.write(f"Dataset sizes tested: {dataset_sizes}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    run_experiment(dataset_sizes, args.base_dir, args.output_dir)

if __name__ == "__main__":
    main()
