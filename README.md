# Clinical Trial Protocol Classifier

A deep learning system that uses PubMedBERT to classify clinical trial protocols as cancer-relevant or not relevant.

## 🚀 Quickstart

### 1. Download Protocol Documents
```bash
# Download the recommended dataset size (750 protocols per category)
python download_protocols.py --target-size 750

# For a smaller dataset during development
python download_protocols.py --target-size 100 --output-dir protocol_documents_dev
```

### 2. Train Model
```bash
# Basic training
python train_classifier.py

# Quick test run with debug mode
python train_classifier.py --debug
```

### 3. Evaluate Model
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
| `--target-size` | 1500 | Number of protocols to download per category |
| `--test-size` | None | Number of protocols for test set (if specified) |
| `--output-dir` | protocol_documents | Directory for main dataset |
| `--test-dir` | protocol_documents_test | Directory for test dataset |
| `--force-download` | False | Force re-download of existing files |
| `--exclude-dirs` | [] | Additional directories to check for existing NCT IDs |

### Train Model (`train_classifier.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--debug` | False | Run in debug mode with smaller dataset |
| `--debug-samples` | 5 | Number of samples per class in debug mode |
| `--epochs` | 3 | Number of training epochs |
| `--output-dir` | ./final_model | Directory to save trained model |

### Evaluate Model (`evaluate_model.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | ./final_model | Directory containing trained model |
| `--test-dir` | protocol_documents_test | Directory containing test data |
| `--batch-size` | 8 | Batch size for evaluation |
| `--output-dir` | evaluation_results | Directory to save results |
| `--debug` | False | Run in debug mode |
| `--debug-samples` | 5 | Number of samples per class in debug mode |

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

## 🔬 Advanced Features

### Label Verification (`verify_labels.py`)
Verify and validate the dataset labels using a rule-based classifier:
```bash
python verify_labels.py --protocol-dir protocol_documents
```

The verification process:
- Uses cancer-specific keywords and patterns
- Identifies cancer-related treatments and procedures
- Generates a detailed verification report

### Learning Curve Analysis (`learning_curve_analysis.py`)
Analyze model performance with varying training dataset sizes:
```bash
python learning_curve_analysis.py
```

Features:
- Trains models with different subset sizes
- Plots learning curves for accuracy and loss
- Helps determine optimal dataset size

### Dataset Balancing (`balance_datasets.py`)
Balance the dataset between cancer and non-cancer protocols:
```bash
python balance_datasets.py
```

## 💻 Hardware Support

The system automatically detects and utilizes available hardware:
- NVIDIA GPUs: Uses mixed precision training (fp16)
- Apple M1 GPU: Uses MPS backend
- CPU: Falls back to CPU with optimized batch sizes

## 📊 Evaluation Results

Evaluation generates:
1. Classification report with precision, recall, and F1 scores
2. Confusion matrix visualization
3. Detailed metrics saved to `evaluation_results/`

## 📝 Notes

- The system uses PubMedBERT which has a maximum token length of 512
- Only the first ~8000 characters of each protocol are used for classification
- Text is automatically cleaned and preprocessed before training
- Training automatically uses the best available hardware (NVIDIA GPU > Apple M1 > CPU)

## 🤝 Contributing

Feel free to open issues or submit pull requests for:
- Bug fixes
- Feature additions
- Documentation improvements
- Performance optimizations
