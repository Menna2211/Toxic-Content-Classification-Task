# Transformer-based Toxic Content Classification with LoRA

A modular pipeline for fine-tuning Transformer models (DistilBERT and ALBERT) for toxicity classification using LoRA (Low-Rank Adaptation) technique. This project demonstrates efficient parameter-efficient fine-tuning for text classification tasks.

##  Project Overview

This project implements a complete workflow for:
- Loading and preprocessing toxic content data
- Fine-tuning Transformer models with LoRA
- Evaluating model performance comprehensively
- Making predictions on new text inputs
- Saving models and experiment configurations

##  Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Multiple Models**: Support for DistilBERT and ALBERT architectures
- **LoRA Fine-tuning**: Parameter-efficient training with reduced computational requirements
- **Comprehensive Evaluation**: Accuracy, F1 scores, confusion matrices, and visualizations
- **Reproducibility**: Full experiment tracking and configuration saving
- **GPU Support**: Automatic CUDA detection and utilization

##  Project Structure

```
transformer-toxic-classification/
├── main.py                           # Main execution script
├── config_transformer.py             # Configuration and settings
├── data_preprocessing_transformer.py  # Data loading and preprocessing
├── model_transformer.py              # Model training and LoRA setup
├── evaluation_transformer.py         # Evaluation metrics and visualization
├── utils_transformer.py              # Utility functions and prediction
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

##  Installation

### Prerequisites
- Python 3.7+
- pip package manager
- CUDA-enabled GPU (optional, but recommended)

### Installation Steps

1. **Install dependencies**
   ```bash
   pip install -r requirements_transformer.txt
   ```

2. **Prepare your data**
   - Place your CSV file named `cellula_toxic_data.csv` in the project root
   - Ensure it has columns: `query`, `image descriptions`, `Toxic Category`

##  Usage

### Basic Execution
```bash
python main.py
```

### Expected Output
The pipeline will:
1. Load and preprocess data
2. Train both DistilBERT and ALBERT models with LoRA
3. Generate performance metrics and visualizations
4. Save trained models and experiment configuration
5. Display example predictions

### Custom Training
```python
# For custom training, modify config_transformer.py
SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-4
MAX_LENGTH = 256
```

##  Configuration

### Key Settings (`config_transformer.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| SEED | 42 | Random seed for reproducibility |
| BATCH_SIZE | 16 | Training and evaluation batch size |
| EPOCHS | 10 | Number of training epochs |
| MAX_LENGTH | 256 | Maximum sequence length |
| LEARNING_RATE | 2e-4 | Learning rate for optimizer |
| LORA_R | 8 | LoRA attention dimension |
| LORA_ALPHA | 16 | LoRA scaling parameter |
| LORA_DROPOUT | 0.1 | Dropout probability for LoRA layers |

##  Modules Overview

### 1. Configuration (`config_transformer.py`)
- Seed management for reproducibility
- Device detection (CPU/GPU)
- Global configuration constants

### 2. Data Preprocessing (`data_preprocessing_transformer.py`)
- Text cleaning and normalization
- Label encoding
- Dataset splitting (60-20-20 train-val-test)
- HuggingFace dataset conversion

### 3. Model Training (`model_transformer.py`)
- LoRA configuration setup
- Tokenizer initialization
- Model loading and LoRA application
- Training workflow with HuggingFace Trainer

### 4. Evaluation (`evaluation_transformer.py`)
- Performance metrics calculation
- Model comparison utilities
- Visualization functions (confusion matrices, bar plots)

### 5. Utilities (`utils_transformer.py`)
- Prediction functions
- Model saving/loading
- Experiment configuration management

##  Model Architectures

### Supported Models
- **DistilBERT-base-uncased**: Smaller, faster BERT variant
- **ALBERT-base-v2**: Parameter-efficient Transformer architecture

### LoRA Configuration
```python
# DistilBERT LoRA targets
target_modules=["q_lin", "v_lin", "k_lin", "out_lin"]

# ALBERT LoRA targets  
target_modules=["query", "value", "key", "output"]
```

##  Evaluation Metrics

The pipeline evaluates models using:

- **Accuracy**: Overall classification accuracy
- **F1 Macro**: Unweighted mean F1 score across classes
- **F1 Weighted**: Weighted mean F1 score based on class support
- **Confusion Matrix**: Per-class performance visualization
- **Classification Report**: Detailed precision, recall, and F1 scores

##  Output Files

### Generated Files
- `distilbert-lora-toxic-seed42/`: Trained DistilBERT model
- `albert-lora-toxic-seed42/`: Trained ALBERT model
- `transformer_lora_experiment_seed42.json`: Experiment configuration
- **Visualizations**: Training plots and performance charts

### Example Prediction
```python
from utils_transformer import predict_toxicity_transformer

pred_class, confidence = predict_toxicity_transformer(
    model, tokenizer, label_encoder,
    "How to make a bomb?", 
    "Police tape across a crime scene"
)
print(f"Prediction: {pred_class} (Confidence: {confidence:.4f})")
```

##  Customization

### Adding New Models
1. Extend `get_lora_config()` in `model_transformer.py`
2. Add new model-specific LoRA target modules
3. Update model name constants

### Modifying Preprocessing
Edit `data_preprocessing_transformer.py`:
```python
def preprocess_text(text):
    # Add custom text cleaning rules
    return cleaned_text
```

### Changing Evaluation Metrics
Modify `compute_metrics()` in `model_transformer.py`:
```python
def compute_metrics(eval_pred):
    # Add custom metrics
    return metrics_dict
```


### Performance Tips
- Use GPU for significantly faster training
- Increase `MAX_LENGTH` for longer texts (if memory allows)
- Experiment with different LoRA parameters (r, alpha, dropout)
- Use mixed precision training (`fp16=True`)

##  Experimental Results

### Model Comparison
The pipeline automatically compares DistilBERT and ALBERT performance across multiple metrics, helping you choose the best model for your specific use case.

### Visualization
- Training loss curves
- Validation accuracy progression
- Confusion matrices for both models
- F1 score comparisons

---




