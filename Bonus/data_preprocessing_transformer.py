# data_preprocessing_transformer.py
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset

def preprocess_text(text):
    """Clean text: convert to lowercase, remove extra whitespace, and remove punctuation (except ., ?, !)"""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s\.\?\!,]', '', text)
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
    df['query_clean'] = df['query'].apply(preprocess_text)
    df['image_desc_clean'] = df['image descriptions'].apply(preprocess_text)
    
    # Combine query and image descriptions
    df['combined_text'] = df['query_clean'] + " " + df['image_desc_clean']
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Toxic Category'])
    num_classes = len(le.classes_)
    
    print(f"Label classes: {le.classes_}")
    print(f"Number of classes: {num_classes}")
    
    return df, le, num_classes

def prepare_datasets(df, test_size=0.2, val_size=0.25, random_state=42):
    """Prepare HuggingFace datasets with proper splitting"""
    # Convert to HuggingFace dataset
    dataset = HFDataset.from_pandas(df[['combined_text', 'label']])
    
    # Split the dataset
    train_test_split = dataset.train_test_split(test_size=test_size, seed=random_state)
    train_val_split = train_test_split['train'].train_test_split(test_size=val_size, seed=random_state)
    
    dataset_dict = {
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': train_test_split['test']
    }
    
    print(f"Training samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")
    print(f"Test samples: {len(dataset_dict['test'])}")
    
    return dataset_dict