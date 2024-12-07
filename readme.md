# Clinical Trial Protocol Classifier

This repository contains scripts for downloading and classifying clinical trial protocols from ClinicalTrials.gov. It implements three different classification approaches:
1. Fine-tuned PubMedBERT (optimized for medical text)
2. Traditional ML baselines (TF-IDF with Logistic Regression and SVM)
3. Zero-shot classification using RoBERTa-MNLI

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
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

Note: The scripts are optimized for Apple M1 chips and will automatically use the appropriate device (MPS, CUDA, or CPU).

## Project Components

### 1. Data Collection

Download and split clinical trial protocols:

```bash
python download_protocols.py --target-size 1500
```

This will:
- Download approximately 3000 protocols (1500 cancer and 1500 non-cancer)
- Split them into train (70%), validation (15%), and test (15%) sets
- Organize them in the appropriate directories

### 2. Model Training

Train the PubMedBERT classifier:

```bash
python train_model.py
```

Train the traditional ML baseline models:

```bash
python baseline_classifiers.py --train --train-dir ./protocol_documents
```

Note: The zero-shot classifier doesn't require training.

### 3. Classification

You can use any of the three classification approaches:

#### PubMedBERT Classifier
```bash
python classify_protocol.py --input path/to/protocol.pdf
python classify_protocol.py --input path/to/protocols/dir --output results.json
```

#### Baseline Classifiers (Traditional ML and Zero-shot)
```bash
python baseline_classifiers.py --input path/to/protocol.pdf
python baseline_classifiers.py --input path/to/protocols/dir --output results.json
```

## Directory Structure

```
.
├── download_protocols.py     # Data collection script
├── train_model.py           # PubMedBERT training script
├── classify_protocol.py     # PubMedBERT inference script
├── baseline_classifiers.py  # Traditional ML and zero-shot classifiers
├── requirements.txt
├── protocol_documents/      # Downloaded protocols
│   ├── cancer/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── non_cancer/
│       ├── train/
│       ├── val/
│       └── test/
├── protocol_classifier/     # PubMedBERT model outputs
│   ├── pytorch_model.bin
│   ├── config.json
│   └── eval_results.txt
└── baseline_models/        # Traditional ML model outputs
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

## Output Formats

### PubMedBERT Classifier
```json
{
    "file_name": "protocol.pdf",
    "classification": "cancer",
    "confidence": 95.67
}
```

### Baseline Classifiers
```json
{
    "file_name": "protocol.pdf",
    "traditional_ml": {
        "log_reg_prediction": "cancer",
        "log_reg_confidence": 95.67,
        "svm_prediction": "cancer",
        "svm_confidence": 93.45
    },
    "zero_shot": {
        "prediction": "cancer",
        "confidence": 87.23
    }
}
```