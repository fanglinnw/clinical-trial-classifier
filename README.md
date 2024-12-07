# Clinical Trial Protocol Classifier

This repository contains scripts for downloading and classifying clinical trial protocols from ClinicalTrials.gov. It implements three different classification approaches:
1. Fine-tuned PubMedBERT (optimized for medical text)
2. Traditional ML baselines (TF-IDF with Logistic Regression and SVM)
3. Zero-shot classification using RoBERTa-MNLI

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/study_classifier.git
cd study_classifier
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

Key dependencies:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- PyMuPDF >= 1.22.0
- scikit-learn >= 1.0.0
- pandas >= 1.5.0
- accelerate >= 0.26.0

Note: The scripts are optimized for Apple M1 chips and will automatically use the appropriate device (MPS, CUDA, or CPU).

## Project Components

### 1. Data Collection

Download and split clinical trial protocols:

```bash
python scripts/download_protocols.py --target-size 1500
```

This will:
- Download approximately 3000 protocols (1500 cancer and 1500 non-cancer)
- Split them into train (70%), validation (15%), and test (15%) sets
- Organize them in the appropriate directories

### 2. Model Training

Train the PubMedBERT classifier:

```bash
python scripts/train_models.py
```

Train the traditional ML baseline models:

```bash
python scripts/classify_protocol.py --train --train-dir ./protocol_documents
```

Note: The zero-shot classifier doesn't require training.

### 3. Classification

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

```
.
├── scripts/                # Main execution scripts
│   ├── download_protocols.py  # Data collection script
│   ├── train_models.py       # Model training script
│   └── classify_protocol.py  # Protocol classification script
├── models/                 # Model checkpoints and configurations
├── logs/                  # Training and inference logs
├── results/               # Classification results output
├── utils/                 # Utility functions and helpers
├── protocol_documents/    # Downloaded protocols
│   ├── cancer/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── non_cancer/
│       ├── train/
│       ├── val/
│       └── test/
├── protocol_classifier/   # PubMedBERT model outputs
│   ├── pytorch_model.bin
│   ├── config.json
│   └── eval_results.txt
└── baseline_models/      # Traditional ML model outputs
    ├── tfidf.joblib
    ├── logistic_regression.joblib
    └── svm.joblib
```

## Model Details

1. **PubMedBERT Classifier**
   - Fine-tuned on medical text
   - Optimized for clinical trial protocols
   - Best performance but requires training
   - GPU/MPS accelerated when available

2. **Traditional ML Baseline**
   - TF-IDF vectorization with Logistic Regression and SVM
   - Faster training and inference
   - Lighter resource requirements
   - Good baseline performance

3. **Zero-shot Classifier**
   - Uses RoBERTa-MNLI model
   - No training required
   - Can adapt to new categories
   - Useful for quick prototyping

## Notes

- The download script includes appropriate delays to respect ClinicalTrials.gov's servers
- All models are optimized for Apple M1 chips and will automatically use the appropriate device
- Training the PubMedBERT model requires significant computational resources
- The traditional ML models provide a good balance of speed and accuracy
- The zero-shot classifier is useful for quick experimentation without training

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