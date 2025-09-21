import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import random
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42

def set_seeds(seed=SEED):
    """Set seeds for reproducible results"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # For deterministic operations (optional, may slow training)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
set_seeds(SEED)
print(f" Random seeds set to {SEED} for reproducibility")


# Load data from CSV
df = pd.read_csv('cellula_toxic_data.csv')
# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text, use_lemmatization=True, remove_stopwords=True):
    """Clean and normalize text with stemming/lemmatization options"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\?\!,]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords (optional)
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # Apply stemming or lemmatization
    if use_lemmatization:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back
    text = ' '.join(tokens)
    
    # Strip whitespace
    text = text.strip()
    
    return text

# Display dataset info
print("Dataset Overview:")
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Categories: {df['Toxic Category'].unique()}")
print(f"Category distribution:\n{df['Toxic Category'].value_counts()}")
print(f"\nFirst few rows:")
print(df.head())

# Apply text preprocessing
print("\nApplying advanced text preprocessing (lemmatization + stopword removal)...")
df['query_clean'] = df['query'].apply(lambda x: preprocess_text(x, use_lemmatization=True, remove_stopwords=True))
df['image_desc_clean'] = df['image descriptions'].apply(lambda x: preprocess_text(x, use_lemmatization=True, remove_stopwords=True))

# Combine preprocessed query and image descriptions
df['combined_text'] = df['query_clean'] + ' [IMG] ' + df['image_desc_clean']

print(f"\nExample preprocessing comparison:")
print(f"Original Query: {df['query'].iloc[0]}")
print(f"Processed Query: {df['query_clean'].iloc[0]}")
print(f"Original Image: {df['image descriptions'].iloc[0]}")
print(f"Processed Image: {df['image_desc_clean'].iloc[0]}")
print(f"Combined Text: {df['combined_text'].iloc[0]}")

# Show preprocessing statistics
print(f"\nPreprocessing Statistics:")
original_words = sum(len(text.split()) for text in df['query'] + ' ' + df['image descriptions'])
processed_words = sum(len(text.split()) for text in df['combined_text'])
print(f"Original total words: {original_words}")
print(f"Processed total words: {processed_words}")
print(f"Word reduction: {((original_words - processed_words) / original_words * 100):.1f}%")

# Check for any empty texts after preprocessing
empty_texts = df['combined_text'].str.strip().eq('').sum()
if empty_texts > 0:
    print(f"\nWarning: {empty_texts} empty texts after preprocessing")
    df = df[df['combined_text'].str.strip() != ''].reset_index(drop=True)
    print(f"Remaining samples: {len(df)}")

# Preprocessing
max_features = 1500  # Increased for combined text
max_length = 75      # Increased for longer combined sequences

# Tokenize combined text (query + image descriptions)
tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(df['combined_text'])
X = tokenizer.texts_to_sequences(df['combined_text'])
X = pad_sequences(X, maxlen=max_length)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(df['Toxic Category'])
y = to_categorical(y_encoded)
num_classes = len(le.classes_)

print(f"\nLabel classes: {le.classes_}")
print(f"Number of classes: {num_classes}")

# Train-validation-test split (60-20-20) with fixed random state
print(f"\nSplitting data: Train/Val/Test")
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.25, random_state=SEED, 
    stratify=np.argmax(y_train_temp, axis=1)
)

print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Check class distribution in each split
y_train_classes = np.argmax(y_train, axis=1)
y_val_classes = np.argmax(y_val, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print(f"\nClass distribution:")
print("Train:", dict(zip(*np.unique(y_train_classes, return_counts=True))))
print("Validation:", dict(zip(*np.unique(y_val_classes, return_counts=True))))
print("Test:", dict(zip(*np.unique(y_test_classes, return_counts=True))))

# Build BiLSTM model with reproducible initialization
model = Sequential([
    Embedding(max_features, 128, input_length=max_length),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
    Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile with fixed seed
model.build((None, max_length))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
print("\nEnhanced BiLSTM Model Architecture:")
model.summary()

# Train model with early stopping
print("\nTraining BiLSTM model with early stopping...")

# Define callbacks with seeds
early_stopping = EarlyStopping(
    monitor='val_loss',           # Monitor validation loss
    patience=10,                  # Wait 10 epochs before stopping
    restore_best_weights=True,    # Keep best weights
    verbose=1                     # Print when stopping
)

from sklearn.utils.class_weight import compute_class_weight

# Encode labels first (you already have y_encoded from LabelEncoder)
classes_for_cw = np.unique(y_encoded)
cw = compute_class_weight(class_weight='balanced', classes=classes_for_cw, y=y_encoded)

# Convert to dict for Keras
class_weight = {i: w for i, w in enumerate(cw)}
print("Class weights:", class_weight)

# --- Train the model with class_weight ---
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,  
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    class_weight=class_weight,  
    verbose=1,
    shuffle=True
)

print(f"\nTraining completed!")
print(f"Total epochs trained: {len(history.history['loss'])}")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"Final learning rate: {history.model.optimizer.learning_rate.numpy():.6f}")

# Evaluate on all three sets
print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

# Training set evaluation
y_train_pred = model.predict(X_train, verbose=0)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)
train_f1_macro = f1_score(y_train_classes, y_train_pred_classes, average='macro')
train_f1_weighted = f1_score(y_train_classes, y_train_pred_classes, average='weighted')

# Validation set evaluation  
y_val_pred = model.predict(X_val, verbose=0)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
val_f1_macro = f1_score(y_val_classes, y_val_pred_classes, average='macro')
val_f1_weighted = f1_score(y_val_classes, y_val_pred_classes, average='weighted')

# Test set evaluation
y_test_pred = model.predict(X_test, verbose=0)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
test_f1_macro = f1_score(y_test_classes, y_test_pred_classes, average='macro')
test_f1_weighted = f1_score(y_test_classes, y_test_pred_classes, average='weighted')

# Print results
print(f"\nF1 SCORES:")
print(f"{'Dataset':<12} {'Macro F1':<10} {'Weighted F1':<12}")
print(f"{'-'*35}")
print(f"{'Train':<12} {train_f1_macro:<10.4f} {train_f1_weighted:<12.4f}")
print(f"{'Validation':<12} {val_f1_macro:<10.4f} {val_f1_weighted:<12.4f}")
print(f"{'Test':<12} {test_f1_macro:<10.4f} {test_f1_weighted:<12.4f}")

print(f"\nFINAL TEST SET CLASSIFICATION REPORT:")
print(classification_report(y_test_classes, y_test_pred_classes, target_names=le.classes_))

# Plotting results with train/val/test
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('BiLSTM Model Training & Evaluation Results', fontsize=16, fontweight='bold')

# Training history
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].axvline(x=len(history.history['loss'])-1, color='red', linestyle='--', alpha=0.7, label='Early Stop')
axes[0, 0].set_title('Model Loss Over Time (Early Stopping)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].axvline(x=len(history.history['accuracy'])-1, color='red', linestyle='--', alpha=0.7, label='Early Stop')
axes[0, 1].set_title('Model Accuracy Over Time (Early Stopping)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# F1 Score comparison across datasets
datasets = ['Train', 'Validation', 'Test']
f1_macro_scores = [train_f1_macro, val_f1_macro, test_f1_macro]
f1_weighted_scores = [train_f1_weighted, val_f1_weighted, test_f1_weighted]

x = np.arange(len(datasets))
width = 0.35

axes[0, 2].bar(x - width/2, f1_macro_scores, width, label='Macro F1', alpha=0.8)
axes[0, 2].bar(x + width/2, f1_weighted_scores, width, label='Weighted F1', alpha=0.8)
axes[0, 2].set_title('F1 Scores Across Datasets')
axes[0, 2].set_ylabel('F1 Score')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(datasets)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Confusion Matrix for Test Set
cm = confusion_matrix(y_test_classes, y_test_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1, 0])
axes[1, 0].set_title('Test Set Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# F1 Scores by class (Test Set)
f1_per_class = f1_score(y_test_classes, y_test_pred_classes, average=None)
axes[1, 1].bar(le.classes_, f1_per_class, color='skyblue', alpha=0.7)
axes[1, 1].set_title('Test Set F1 Score by Class')
axes[1, 1].set_xlabel('Categories')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

# Sample distribution across splits
split_counts = [len(X_train), len(X_val), len(X_test)]
colors = ['#ff9999', '#66b3ff', '#99ff99']
axes[1, 2].pie(split_counts, labels=datasets, autopct='%1.1f%%', colors=colors, startangle=90)
axes[1, 2].set_title('Data Split Distribution')

plt.tight_layout()
plt.show()

# Test function with preprocessing
def predict_toxicity(query, image_desc=""):
    # Apply same preprocessing
    query_clean = preprocess_text(query, use_lemmatization=True, remove_stopwords=True)
    image_desc_clean = preprocess_text(image_desc, use_lemmatization=True, remove_stopwords=True)
    combined = query_clean + ' [IMG] ' + image_desc_clean if image_desc else query_clean
    
    sequence = tokenizer.texts_to_sequences([combined])
    padded = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded, verbose=0)
    predicted_class = le.classes_[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Example predictions
print("\nExample Predictions:")
test_cases = [
    ("How to make a bomb?", "Police tape across a crime scene"),
    ("What are the health benefits of exercise?", "A child playing in a sunny meadow"),
    ("Tell me about election fraud methods", "People walking through a peaceful city street")
]

for query, img_desc in test_cases:
    pred_class, confidence = predict_toxicity(query, img_desc)
    print(f"Query: '{query}'")
    print(f"Image: '{img_desc}'")
    print(f"Prediction: {pred_class} (Confidence: {confidence:.4f})\n")

# Save model with seed info
model_filename = f'bilstm_toxic_classifier_seed{SEED}.h5'
model.save(model_filename)
print(f"Model saved as '{model_filename}'")

# Save experiment configuration
experiment_config = {
    'seed': SEED,
    'max_features': max_features,
    'max_length': max_length,
    'model_architecture': 'Stacked BiLSTM with Early Stopping',
    'preprocessing': 'Lemmatization + Stopword Removal',
    'data_split': '60-20-20 (Train-Val-Test)',
    'total_epochs': len(history.history['loss']),
    'best_val_loss': min(history.history['val_loss']),
    'test_f1_macro': test_f1_macro,
    'test_f1_weighted': test_f1_weighted
}

print(f"\n EXPERIMENT CONFIGURATION:")
print(f"{'Parameter':<20} {'Value':<30}")
print(f"{'-'*50}")
for key, value in experiment_config.items():
    print(f"{key:<20} {str(value):<30}")

# Save config to file
import json
with open(f'experiment_config_seed{SEED}.json', 'w') as f:
    json.dump(experiment_config, f, indent=2)
print(f"\nConfiguration saved to 'experiment_config_seed{SEED}.json'")