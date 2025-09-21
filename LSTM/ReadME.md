# Toxic Content Classification with BiLSTM

A modular deep learning pipeline for classifying toxic content using Bidirectional LSTM (BiLSTM) with text preprocessing, model training, evaluation, and prediction capabilities.

## 📁 Project Structure

```
toxic-classification/
├── main.py              # Main execution script
├── config.py            # Configuration and seed settings
├── data_preprocessing.py # Data loading and preprocessing functions
├── model.py             # Model building and training functions
├── evaluation.py        # Model evaluation and visualization functions
├── utils.py             # Utility functions for prediction and saving
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

##  Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline:**
   ```bash
   python main.py
   ```

##  Configuration

The project uses the following default configuration (`config.py`):

- **SEED:** 42 (for reproducibility)
- **MAX_FEATURES:** 1500 (vocabulary size)
- **MAX_LENGTH:** 75 (sequence length)
- **BATCH_SIZE:** 16
- **EPOCHS:** 100
- **PATIENCE:** 10 (early stopping)

##  Data Processing

### Input Data Format
The pipeline expects a CSV file with the following columns:
- `query`: Text query to classify
- `image descriptions`: Associated image description
- `Toxic Category`: Target label (toxic/non-toxic categories)

### Preprocessing Steps
1. **Text Cleaning:** Lowercasing, whitespace normalization, special character removal
2. **Tokenization:** Word tokenization with NLTK
3. **Stopword Removal:** Optional removal of common stopwords
4. **Lemmatization:** Word normalization using WordNet lemmatizer
5. **Text Combination:** Query and image descriptions combined with `[IMG]` separator

### Feature Engineering
- Tokenization with Keras Tokenizer
- Sequence padding to fixed length
- Label encoding for multi-class classification

##  Model Architecture

The BiLSTM model consists of:

1. **Embedding Layer:** 128-dimensional embeddings
2. **Bidirectional LSTM Layer:** 64 units with dropout (0.3)
3. **Bidirectional LSTM Layer:** 32 units with dropout (0.3)
4. **Dense Layers:** 64 and 32 units with ReLU activation
5. **Output Layer:** Softmax activation for multi-class classification

##  Training

### Training Strategy
- **Class Weighting:** Automatic class weight calculation for imbalanced data
- **Early Stopping:** Stops training when validation loss stops improving
- **Validation:** 60-20-20 split (Train-Validation-Test)

### Hyperparameters
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss:** Categorical Crossentropy
- **Metrics:** Accuracy

## Evaluation Metrics

The model is evaluated using:

- **Accuracy:** Overall classification accuracy
- **F1 Scores:** Macro and weighted F1 scores
- **Confusion Matrix:** Per-class performance visualization
- **Classification Report:** Detailed precision, recall, and F1 scores

##  Prediction

### Usage Example
```python
from utils import predict_toxicity

pred_class, confidence = predict_toxicity(
    model, tokenizer, label_encoder, 
    "How to make a bomb?", 
    "Police tape across a crime scene"
)
```

### Example Output
```
Query: 'How to make a bomb?'
Image: 'Police tape across a crime scene'
Prediction: toxic (Confidence: 0.9234)
```

##  Output Files

The pipeline generates:

- **Model File:** `bilstm_toxic_classifier_seed42.h5` (trained model)
- **Config File:** `experiment_config_seed42.json` (experiment details)
- **Visualizations:** Training history and evaluation plots

##  Dependencies

### Core Libraries
- TensorFlow 2.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK

### Required NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

##  Customization

### Modify Preprocessing
Edit `data_preprocessing.py` to change:
- Text cleaning rules
- Stopword removal behavior
- Lemmatization/stemming options

### Adjust Model Architecture
Modify `build_bilstm_model()` in `model.py` to:
- Change layer sizes
- Add/remove layers
- Modify dropout rates

### Update Training Parameters
Edit `config.py` to change:
- Batch size
- Epochs
- Early stopping patience
- Learning rate

##  Results Interpretation

The pipeline provides comprehensive evaluation including:

- **Training Curves:** Loss and accuracy over epochs
- **F1 Scores:** Macro and weighted averages across datasets
- **Confusion Matrix:** Visual representation of predictions
- **Class-wise Performance:** F1 scores for each category

##  Troubleshooting

### Common Issues
- **NLTK Data Missing:** Run NLTK downloads as shown above
- **Memory Issues:** Reduce batch size or sequence length
- **Slow Training:** Use GPU-enabled TensorFlow version

### Performance Tips
- Use GPU for faster training
- Adjust `MAX_FEATURES` and `MAX_LENGTH` based on your dataset
- Monitor class distribution for imbalanced data


