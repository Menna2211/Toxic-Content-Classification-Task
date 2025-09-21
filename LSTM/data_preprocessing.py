import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

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
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # Apply stemming or lemmatization
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back
    text = ' '.join(tokens)
    
    # Strip whitespace
    text = text.strip()
    
    return text

def load_and_preprocess_data(file_path):
    """Load data from CSV and apply preprocessing"""
    df = pd.read_csv(file_path)
    
    print("Dataset Overview:")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Categories: {df['Toxic Category'].unique()}")
    print(f"Category distribution:\n{df['Toxic Category'].value_counts()}")
    
    # Apply text preprocessing
    print("\nApplying advanced text preprocessing...")
    df['query_clean'] = df['query'].apply(lambda x: preprocess_text(x, use_lemmatization=True, remove_stopwords=True))
    df['image_desc_clean'] = df['image descriptions'].apply(lambda x: preprocess_text(x, use_lemmatization=True, remove_stopwords=True))
    
    # Combine preprocessed query and image descriptions
    df['combined_text'] = df['query_clean'] + ' ' + df['image_desc_clean']
    
    # Check for any empty texts after preprocessing
    empty_texts = df['combined_text'].str.strip().eq('').sum()
    if empty_texts > 0:
        print(f"\nWarning: {empty_texts} empty texts after preprocessing")
        df = df[df['combined_text'].str.strip() != ''].reset_index(drop=True)
        print(f"Remaining samples: {len(df)}")
        
    return df

def prepare_features(df, max_features, max_length):
    """Prepare features for model training"""
    # Tokenize combined text
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['combined_text'])
    X = tokenizer.texts_to_sequences(df['combined_text'])
    X = pad_sequences(X, maxlen=max_length)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['Toxic Category'])
    y = to_categorical(y_encoded)
    
    return X, y, y_encoded, tokenizer, label_encoder

def split_data(X, y, y_encoded, test_size=0.2, val_size=0.25, random_state=42):
    """Split data into train, validation, and test sets"""
    # First split: train+val vs test
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Get encoded labels for train_temp for stratification
    y_train_temp_encoded = np.argmax(y_train_temp, axis=1)
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=val_size, random_state=random_state, 
        stratify=y_train_temp_encoded
    )
    
    # Convert to class indices for analysis
    y_train_classes = np.argmax(y_train, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\nClass distribution:")
    print("Train:", dict(zip(*np.unique(y_train_classes, return_counts=True))))
    print("Validation:", dict(zip(*np.unique(y_val_classes, return_counts=True))))
    print("Test:", dict(zip(*np.unique(y_test_classes, return_counts=True))))
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            y_train_classes, y_val_classes, y_test_classes)