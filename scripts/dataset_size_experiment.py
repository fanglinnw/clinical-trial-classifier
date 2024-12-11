import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add the root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from train_models import train_bert_model
from utils.text_extractor import get_extractor
from evaluate_models import ProtocolEvaluator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def process_all_data(base_dir, extractor, split):
    """
    Process all PDF files in a directory once and cache the results.
    """
    logger = logging.getLogger(__name__)
    data = []
    
    # Process cancer files
    cancer_dir = Path(base_dir) / 'cancer' / split
    logger.info(f"Processing all cancer protocols in {split} split")
    for pdf_path in tqdm(list(cancer_dir.glob('*.pdf')), desc=f"Processing cancer protocols ({split})"):
        result = extractor.extract_from_pdf(pdf_path)
        if result["full_text"]:
            data.append({
                'text': result["full_text"],
                'label': 1,  # cancer
                'file_name': pdf_path.name
            })
    
    # Process non-cancer files
    non_cancer_dir = Path(base_dir) / 'non_cancer' / split
    logger.info(f"Processing all non-cancer protocols in {split} split")
    for pdf_path in tqdm(list(non_cancer_dir.glob('*.pdf')), desc=f"Processing non-cancer protocols ({split})"):
        result = extractor.extract_from_pdf(pdf_path)
        if result["full_text"]:
            data.append({
                'text': result["full_text"],
                'label': 0,  # non-cancer
                'file_name': pdf_path.name
            })
    
    return pd.DataFrame(data)

def sample_from_processed_data(df, sample_size):
    """
    Sample balanced dataset from already processed data.
    """
    samples_per_class = sample_size // 2
    
    cancer_df = df[df['label'] == 1].sample(n=min(samples_per_class, len(df[df['label'] == 1])), random_state=42)
    non_cancer_df = df[df['label'] == 0].sample(n=min(samples_per_class, len(df[df['label'] == 0])), random_state=42)
    
    return pd.concat([cancer_df, non_cancer_df]).sample(frac=1, random_state=42).reset_index(drop=True)

def run_experiment(dataset_sizes, base_dir, output_base_dir):
    logger = setup_logging()
    extractor = get_extractor("simple")
    
    # Process all data once
    logger.info("Processing all training data once...")
    full_train_df = process_all_data(base_dir, extractor, "train")
    
    logger.info("Processing validation data...")
    val_df = process_all_data(base_dir, extractor, "val")
    
    logger.info("Processing test data...")
    test_df = process_all_data(base_dir, extractor, "test")
    
    # Cache the processed data
    cache_dir = os.path.join(output_base_dir, "processed_data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    full_train_df.to_parquet(os.path.join(cache_dir, "full_train.parquet"))
    val_df.to_parquet(os.path.join(cache_dir, "val.parquet"))
    test_df.to_parquet(os.path.join(cache_dir, "test.parquet"))
    
    results = []
    
    for size in dataset_sizes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running experiment with dataset size: {size}")
        logger.info(f"{'='*50}")
        
        # Sample from processed data
        sampled_train_df = sample_from_processed_data(full_train_df, size)
        
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
        
        # Initialize evaluator
        evaluator = ProtocolEvaluator(output_dir)
        
        # Evaluate on test sets
        test_cancer_dir = "../ppp_docs"
        test_non_cancer_dir = "./protocol_documents/non_cancer/test"
        
        cancer_results = evaluator.evaluate_directory(Path(test_cancer_dir), is_cancer=True)
        non_cancer_results = evaluator.evaluate_directory(Path(test_non_cancer_dir), is_cancer=False)
        all_results = cancer_results + non_cancer_results
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(all_results)
        
        # Store results
        result = {
            "dataset_size": size,
            "output_dir": output_dir,
            "metrics": metrics['pubmedbert']  # We only care about PubMedBERT metrics for this experiment
        }
        results.append(result)
        
        # Save detailed results after each experiment
        with open(os.path.join(output_base_dir, "experiment_results.txt"), "a") as f:
            f.write(f"\nResults for dataset size {size}:\n")
            f.write(f"Metrics: {metrics['pubmedbert']}\n")
            f.write("-" * 50 + "\n")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./protocol_documents",
                      help="Base directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="./experiment_results",
                      help="Directory to save experiment results")
    parser.add_argument("--use_cached_data", action="store_true",
                      help="Use cached processed data if available")
    args = parser.parse_args()

    dataset_sizes = [200, 400, 800, 1600, 3200, 4800, 6400, 8000]
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "experiment_config.txt"), "w") as f:
        f.write("Dataset Size Experiment Configuration:\n")
        f.write(f"Dataset sizes tested: {dataset_sizes}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    run_experiment(dataset_sizes, args.base_dir, args.output_dir)

if __name__ == "__main__":
    main()