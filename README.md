# Clinical Trial Protocol Classifier

A deep learning system that uses PubMedBERT and LLM-based classifiers to classify clinical trial protocols as cancer-relevant or not relevant.

## 🔧 Requirements

- Python 3.10 (Required for PyTorch compatibility)
- Virtual environment recommended

```bash
# Install Python 3.10 if not already installed
brew install python@3.10

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quickstart

### 1. Download Protocol Documents
```bash
# Download the default dataset (750 protocols per category for training, 100 for testing)
python download_protocols.py

# For a smaller dataset during development
python download_protocols.py --train-size 100 --test-size 20

# Skip downloading test set
python download_protocols.py --no-test
```

### 2. Balance Dataset (Optional)
```bash
# Balance the number of protocols between cancer and non-cancer categories
python balance_datasets.py
```

The balancing process:
- Counts files in cancer and non-cancer directories
- If non-cancer has more files, randomly removes excess files to match cancer count
- Ensures equal representation of both categories

### 3. Verify Dataset
```bash
# Verify dataset integrity and label correctness
python verify_dataset.py

# Use custom directories
python verify_dataset.py --train-dir custom_train_dir --test-dir custom_test_dir
```

The verification process checks:
- Directory structure and dataset sizes
- No overlap between train/test sets
- Label correctness using rule-based classification
- Generates a comprehensive report

### 4. Train Model
```bash
# Train using PubMedBERT classifier
python train_classifier.py

# Quick test run with debug mode
python train_classifier.py --debug
```

### 5. Alternative: Use LLM Classifier
```bash
# Use Mistral-based LLM classifier
python llm_classifier.py

# Run with specific directories
python llm_classifier.py --train-dir custom_train_dir --test-dir custom_test_dir
```

The LLM classifier:
- Uses Mistral-7B-Instruct-v0.2 for classification
- Provides detailed predictions and confidence scores
- Generates comprehensive evaluation metrics

### 6. Evaluate Model
```bash
# Basic evaluation
python evaluate_model.py
```

## 📁 Directory Structure
```
protocol_documents/          # Training data
├── cancer/                 # Cancer-relevant protocols
└── non_cancer/            # Non-relevant protocols

protocol_documents_test/     # Test data (optional)
├── cancer/
└── non_cancer/
```

## 🛠️ Detailed Configuration

### Download Protocols (`download_protocols.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--train-size` | 750 | Number of protocols to download per category for training |
| `--test-size` | 100 | Number of protocols to download per category for testing |
| `--no-test` | False | Skip downloading test set |
| `--train-dir` | protocol_documents | Directory for training dataset |
| `--test-dir` | protocol_documents_test | Directory for test dataset |
| `--force-download` | False | Force re-download of existing files |
| `--exclude-dirs` | [] | Additional directories to check for existing NCT IDs |

### Balance Dataset (`balance_datasets.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--cancer-dir` | protocol_documents/cancer | Directory containing cancer protocols |
| `--non-cancer-dir` | protocol_documents/non_cancer | Directory containing non-cancer protocols |

### Verify Dataset (`verify_dataset.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--train-dir` | protocol_documents | Directory containing training data |
| `--test-dir` | protocol_documents_test | Directory containing test data |
| `--output-file` | dataset_verification_report.txt | Output file for verification report |

### Train Model (`train_classifier.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--train-dir` | protocol_documents | Directory containing training data |
| `--model-dir` | ./final_model | Directory to save trained model |
| `--debug` | False | Run in debug mode |
| `--debug-samples` | 5 | Number of samples per class in debug mode |
| `--epochs` | 5 | Number of training epochs |
| `--learning-rate` | 1e-5 | Learning rate for training |

### LLM Classifier (`llm_classifier.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--train-dir` | protocol_documents | Directory containing training data |
| `--test-dir` | protocol_documents_test | Directory containing test data |
| `--output-dir` | evaluation_results | Directory to save results |
| `--batch-size` | 8 | Batch size for evaluation |

### Evaluate Model (`evaluate_model.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--test-dir` | protocol_documents_test | Directory containing test data |
| `--model-dir` | ./final_model | Directory containing trained model |
| `--output-dir` | evaluation_results | Directory to save results |
| `--batch-size` | 8 | Batch size for evaluation |
| `--debug` | False | Run in debug mode |
| `--debug-samples` | 5 | Number of samples per class in debug mode |

## 📊 Evaluation Results

Here are the evaluation results for our clinical trial protocol classifier, treating cancer protocols as the positive class:

### PubMedBERT Model Performance
- **Accuracy**: 95.5% (1863/1950 correct classifications)
- **Precision**: 94.7% (941 true positives out of 994 predicted cancer protocols)
- **Recall**: 96.5% (941 true positives out of 975 actual cancer protocols)
- **F1-Score**: 95.6%

Confusion Matrix:
- True Negatives: 922 (correctly identified non-cancer protocols)
- False Positives: 53 (non-cancer protocols misclassified as cancer)
- False Negatives: 34 (cancer protocols misclassified as non-cancer)
- True Positives: 941 (correctly identified cancer protocols)

### Zero-Shot LLM Performance
- **Accuracy**: 87.1% (1699/1950 correct classifications)
- **Precision**: 86.7% (855 true positives out of 986 predicted cancer protocols)
- **Recall**: 87.7% (855 true positives out of 975 actual cancer protocols)
- **F1-Score**: 87.2%

Confusion Matrix:
- True Negatives: 844 (correctly identified non-cancer protocols)
- False Positives: 131 (non-cancer protocols misclassified as cancer)
- False Negatives: 120 (cancer protocols misclassified as non-cancer)
- True Positives: 855 (correctly identified cancer protocols)

### Key Findings
- PubMedBERT model performs better than the zero-shot approach (95.5% vs 87.1% accuracy)
- The model misses 34 cancer protocols (3.5% false negative rate)
- Both approaches show similar precision and recall scores
- Evaluation outputs include:
  1. Detailed classification metrics
  2. Confusion matrix visualizations
  3. Confidence scores for each prediction
  4. Complete results saved in `evaluation_results/`

## 📝 Notes

- The system provides two classification approaches:
  1. PubMedBERT-based classification (traditional approach)
  2. LLM-based classification using Mistral-7B (newer approach)
- Text is automatically cleaned and preprocessed before training
- Training automatically uses the best available hardware
- Balanced datasets are recommended for optimal performance

## 🤝 Contributing

Feel free to open issues or submit pull requests for:
- Bug fixes
- Feature additions
- Documentation improvements
- Performance optimizations

## 🔍 Debug Mode

Debug mode is available in both training and evaluation scripts for quick testing:

```bash
# Train with debug mode
python train_classifier.py --debug --debug-samples 5

# Evaluate with debug mode
python evaluate_model.py --debug --debug-samples 5
```

Debug mode features:
- Uses smaller dataset (default: 5 samples per class)
- Runs fewer epochs during training (2 instead of 3)
- More frequent logging
- Faster iteration for testing changes

## 💻 Hardware Support

The system automatically detects and utilizes available hardware:
- CPU: Default execution mode
- GPU: Automatically utilized when available (CUDA or MPS)
- Memory optimization: Uses half-precision (fp16) when possible
