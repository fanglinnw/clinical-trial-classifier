# Clinical Trial Protocol Classifier

This repository contains a comprehensive system for downloading, processing, and classifying clinical trial protocols from ClinicalTrials.gov. It implements multiple classification approaches and provides a unified interface for comparison.

## Features

- Multiple classification approaches:
  - Fine-tuned PubMedBERT (optimized for medical text)
  - Traditional ML (TF-IDF with Logistic Regression and SVM)
  - Zero-shot classification using BART-MNLI
- Centralized text extraction and processing
- Support for both individual and batch processing
- Optimized for Apple M1/M2 chips and CUDA devices
- Comprehensive evaluation metrics

## Project Structure

```
clinical-trial-classifier/
├── utils/
│   ├── __init__.py
│   └── text_extractor.py          # Centralized text extraction
├── models/
│   ├── __init__.py
│   ├── pubmedbert_classifier.py   # Fine-tuned BERT classifier
│   └── baseline_classifiers.py    # Traditional ML and zero-shot
├── scripts/
│   ├── download_protocols.py      # Data collection
│   ├── train_models.py           # Model training
│   └── classify_protocol.py      # Unified inference
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fanglinnw/clinical-trial-classifier.git
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

## Usage

### 1. Download Protocols

Download and split clinical trial protocols into train/val/test sets:

```bash
python scripts/download_protocols.py --target-size 1500
```

This will:
- Download approximately 3000 protocols (1500 cancer and 1500 non-cancer)
- Split them into train (70%), validation (15%), and test (15%) sets
- Organize them in the appropriate directories

### 2. Train Models

Train all classifiers:

```bash
python scripts/train_models.py
```

This trains:
- PubMedBERT classifier
- Traditional ML models (TF-IDF + LogisticRegression/SVM)
- Zero-shot classifier (no training required)

### 3. Classify Protocols

Use the unified classifier interface:

```bash
# Classify single PDF with all models
python scripts/classify_protocol.py --input protocol.pdf

# Batch classify directory
python scripts/classify_protocol.py --input protocols_dir/ --output results.json

# Adjust text processing length
python scripts/classify_protocol.py --input protocol.pdf --max-length 10000
```

## Output Format

The classifier produces structured JSON output:

```json
{
    "file_name": "protocol.pdf",
    "pubmedbert": {
        "classification": "cancer",
        "confidence": 95.67
    },
    "traditional_ml": {
        "log_reg_prediction": "cancer",
        "log_reg_confidence": 92.45,
        "svm_prediction": "cancer",
        "svm_confidence": 90.32
    },
    "zero_shot": {
        "prediction": "cancer",
        "confidence": 87.65
    }
}
```

## Model Details

### PubMedBERT Classifier
- Fine-tuned on medical text
- Best performance for medical domain
- GPU/MPS accelerated when available

### Traditional ML Baseline
- TF-IDF vectorization
- Logistic Regression and SVM classifiers
- Faster training and inference
- Good baseline performance

### Zero-shot Classifier
- Uses BART-MNLI model
- No training required
- Flexible for new categories
- Good for prototyping

## Directory Structure After Training

```
clinical-trial-classifier/
├── protocol_documents/         # Downloaded protocols
│   ├── cancer/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── non_cancer/
│       ├── train/
│       ├── val/
│       └── test/
├── protocol_classifier/        # PubMedBERT model
│   ├── pytorch_model.bin
│   ├── config.json
│   └── eval_results.txt
└── baseline_models/           # Traditional ML models
    ├── tfidf.joblib
    ├── logistic_regression.joblib
    └── svm.joblib
```

## Notes

- The download script includes appropriate delays to respect ClinicalTrials.gov's servers
- Text extraction is limited to 8000 characters by default (configurable)
- All models are optimized for Apple M1/M2 chips and will automatically use appropriate acceleration
- The zero-shot classifier can be used immediately without training
- Results can be exported to JSON for further analysis

## Hardware Requirements

- Minimum: 8GB RAM, multicore CPU
- Recommended: 16GB RAM, GPU or Apple M1/M2 chip
- Storage: ~5GB for full dataset and models

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.