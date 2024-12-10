# Clinical Trial Protocol Classifier

This repository contains a machine learning system for classifying clinical trial protocols from ClinicalTrials.gov as either cancer-related or non-cancer studies. The system implements multiple approaches for comparison:

1. **BERT-based Models**
   - BioBERT: Specialized for biomedical text
   - Bio_ClinicalBERT: Optimized for clinical notes
   - PubMedBERT: Pre-trained on PubMed abstracts and full-text
2. **Traditional ML Baselines**
   - TF-IDF with Logistic Regression
   - TF-IDF with SVM

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/clinical-trial-classifier.git
cd clinical-trial-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- transformers >= 4.30.0 (for BERT models)
- torch >= 2.0.0 (optimized for Apple Silicon)
- scikit-learn >= 1.0.0 (for baseline models)
- PyMuPDF >= 1.22.0 (for PDF processing)
- pandas >= 1.5.0 (for data handling)
- tqdm (for progress bars)
- requests (for API calls)

### Hardware Support
- Automatically uses MPS acceleration on Apple Silicon Macs
- Falls back to CPU if no GPU/MPS is available
- CUDA support for systems with NVIDIA GPUs

## Project Components

### 1. Data Collection (`download_protocols.py`)

Download and split clinical trial protocols:

```bash
python scripts/download_protocols.py [options]
```

Options:
- `--target-size`: Target number of protocols per type (cancer/non-cancer) (default: 1500)
- `--train-ratio`: Ratio of studies to use for training (default: 0.7)
- `--val-ratio`: Ratio of studies to use for validation (default: 0.15)
- `--test-ratio`: Ratio of studies to use for testing (default: 0.15)
- `--force-download`: Force re-download of existing files
- `--output-dir`: Output directory for protocol documents (default: protocol_documents)
- `--eval-mode`: Download evaluation dataset with no overlap with other datasets
- `--exclude-dirs`: Additional directories to check for existing NCT IDs to exclude

### 2. Model Training (`train_models.py`)

Train all models (BERT variants and baselines):

```bash
python scripts/train_models.py [options]
```

Options:
- `--data-dir`: Directory containing protocol documents (default: ./protocol_documents)
- `--output-dir`: Directory to save models (default: ./trained_models)
- `--max-length`: Maximum sequence length for BERT models (default: 512)
- `--batch-size`: Training batch size (default: 4)
- `--epochs`: Number of training epochs (default: 5)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--warmup-ratio`: Warmup ratio (default: 0.1)
- `--weight-decay`: Weight decay (default: 0.01)
- `--early-stopping-patience`: Early stopping patience (default: 3)
- `--seed`: Random seed (default: 42)
- `--max-text-length`: Maximum text length for extraction (default: 8000)

### 3. Model Evaluation (`evaluate_models.py`)

Evaluate model performance:

```bash
python scripts/evaluate_models.py [options]
```

Options:
- `--cancer-dir`: Directory containing cancer protocol PDFs (required)
- `--non-cancer-dir`: Directory containing non-cancer protocol PDFs (required)
- `--models-dir`: Directory containing trained models (default: ./trained_models)
- `--baseline-path`: Path to baseline models (default: ./baseline_models)
- `--max-length`: Maximum text length to process (default: 8000)

### 4. Prediction (`predict.py`)

Predict cancer relevance for new protocols:

```bash
python scripts/predict.py [options] input_path
```

Options:
- `input_path`: Path to a PDF file or directory containing PDFs (required)
- `--model-path`: Path to the trained PubMedBERT model (default: ./protocol_classifier)
- `--max-length`: Maximum sequence length for the model (default: 8000)
- `--output`: Optional path to save results as JSON

### 5. Classification

The classifier can process directories of cancer and non-cancer protocols:

```bash
python scripts/classify_protocol.py \
    --cancer-dir path/to/cancer/protocols \
    --non-cancer-dir path/to/non-cancer/protocols \
    --output results.json
```

Additional options:
```bash
python scripts/classify_protocol.py \
    --cancer-dir path/to/cancer/protocols \
    --non-cancer-dir path/to/non-cancer/protocols \
    --output results.json \
    --max-length 8000 \
    --pubmedbert-path ./protocol_classifier \
    --baseline-path ./baseline_models
```

The script will:
- Process all PDFs in the specified directories
- Run classification using both PubMedBERT and baseline models
- Generate performance metrics and confusion matrices
- Save detailed results and summary to the specified output file

## Directory Structure

When downloading data in normal mode (not eval mode):
```
clinical-trial-classifier/
├── protocol_documents/       # Base directory for protocol documents
│   ├── cancer/             # Cancer protocols
│   │   ├── train/         # Training set
│   │   ├── val/           # Validation set
│   │   └── test/          # Test set
│   └── non_cancer/        # Non-cancer protocols
│       ├── train/         # Training set
│       ├── val/           # Validation set
│       └── test/          # Test set
├── trained_models/         # Directory for trained models
│   ├── biobert/           # Fine-tuned BioBERT model
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── eval_results.txt
│   ├── clinicalbert/      # Fine-tuned Bio_ClinicalBERT model
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── eval_results.txt
│   ├── pubmedbert/        # Fine-tuned PubMedBERT model
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── eval_results.txt
│   └── baseline/          # Traditional ML models
│       ├── tfidf_vectorizer.pkl
│       ├── logistic_regression.pkl
│       └── svm.pkl
└── scripts/               # Python scripts
    ├── download_protocols.py
    ├── train_models.py
    ├── evaluate_models.py
    └── predict.py

When downloading in eval mode (--eval-mode):
```
clinical-trial-classifier/
├── protocol_documents/    # Base directory for protocol documents
│   ├── cancer/          # Cancer protocols (no splits)
│   └── non_cancer/     # Non-cancer protocols (no splits)

## Models

The project uses several models for clinical trial protocol classification:

### BERT-based Models
1. **BioBERT** (`dmis-lab/biobert-v1.1`)
   - Specialized for biomedical text
   - Pre-trained on PubMed abstracts and PMC full-text articles

2. **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`)
   - Optimized for clinical text
   - Pre-trained on clinical notes from MIMIC-III

3. **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)
   - Pre-trained from scratch on PubMed
   - Uses both abstracts and full-text articles

### Traditional ML Models (Baselines)
1. **TF-IDF + Logistic Regression**
   - Uses TF-IDF vectorization
   - Maximum text length: 8000 tokens

2. **TF-IDF + SVM**
   - Uses TF-IDF vectorization
   - Linear kernel for efficiency

All BERT models are fine-tuned on the clinical trial protocols with:
- Maximum sequence length: 512 tokens
- Batch size: 4
- Learning rate: 2e-5
- Warmup ratio: 0.1
- Weight decay: 0.01
- Early stopping with patience of 3

## Output Format

The output JSON file contains both individual results and performance summary:

```json
{
    "results": [
        {
            "file_name": "protocol.pdf",
            "true_label": "cancer",
            "predictions": {
                "pubmedbert": {
                    "prediction": "cancer",
                    "confidence": 0.95
                },
                "baseline": {
                    "prediction": "cancer",
                    "confidence": 0.92
                }
            }
        }
    ],
    "summary": {
        "accuracy": {
            "pubmedbert": 0.94,
            "baseline": 0.91
        },
        "precision": {
            "pubmedbert": 0.93,
            "baseline": 0.90
        },
        "recall": {
            "pubmedbert": 0.95,
            "baseline": 0.92
        },
        "f1": {
            "pubmedbert": 0.94,
            "baseline": 0.91
        }
    }
}
```

## Notes

- The download script includes appropriate delays to respect ClinicalTrials.gov's servers
- All models are optimized for Apple M1 chips and will automatically use the appropriate device
- Training the PubMedBERT model requires significant computational resources
- The traditional ML models provide a good balance of speed and accuracy
- The zero-shot classifier is useful for quick experimentation without training