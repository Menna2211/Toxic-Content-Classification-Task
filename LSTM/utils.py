import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preprocessing import preprocess_text

def predict_toxicity(model, tokenizer, label_encoder, query, image_desc="", max_length=75):
    """Predict toxicity for given query and image description"""
    # Apply same preprocessing
    query_clean = preprocess_text(query, use_lemmatization=True, remove_stopwords=True)
    image_desc_clean = preprocess_text(image_desc, use_lemmatization=True, remove_stopwords=True)
    combined = query_clean + '  ' + image_desc_clean if image_desc else query_clean
    
    sequence = tokenizer.texts_to_sequences([combined])
    padded = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded, verbose=0)
    predicted_class = label_encoder.classes_[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

def save_experiment_config(config, file_path):
    """Save experiment configuration to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to '{file_path}'")