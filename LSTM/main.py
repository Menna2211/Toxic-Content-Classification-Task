import warnings
warnings.filterwarnings('ignore')

import numpy as np
from config import set_seeds, SEED, MAX_FEATURES, MAX_LENGTH, BATCH_SIZE, EPOCHS, PATIENCE
from data_preprocessing import preprocess_text, load_and_preprocess_data, prepare_features, split_data
from model import build_bilstm_model, train_model
from evaluation import evaluate_model, plot_results
from utils import predict_toxicity, save_experiment_config

def main():
    # Set seeds for reproducibility
    set_seeds(SEED)
    print(f"Random seeds set to {SEED} for reproducibility")
    
    # Load and preprocess data
    df = load_and_preprocess_data('cellula_toxic_data.csv')
    
    # Display preprocessing examples
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
    
    # Prepare features
    X, y, y_encoded, tokenizer, label_encoder = prepare_features(df, MAX_FEATURES, MAX_LENGTH)
    
    # Get label information
    num_classes = len(label_encoder.classes_)
    print(f"\nLabel classes: {label_encoder.classes_}")
    print(f"Number of classes: {num_classes}")
    
    # Split data
    (X_train, X_val, X_test, y_train, y_val, y_test, 
     y_train_classes, y_val_classes, y_test_classes) = split_data(
        X, y, y_encoded, test_size=0.2, val_size=0.25, random_state=SEED
    )
    
    # Build and train model
    model = build_bilstm_model(MAX_FEATURES, MAX_LENGTH, num_classes)
    history = train_model(model, X_train, y_train, X_val, y_val, y_encoded, 
                         epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE)
    
    # Evaluate model
    (y_train_pred_classes, y_val_pred_classes, y_test_pred_classes,
     train_f1_macro, train_f1_weighted, val_f1_macro, val_f1_weighted,
     test_f1_macro, test_f1_weighted) = evaluate_model(
        model, X_train, X_val, X_test, 
        y_train_classes, y_val_classes, y_test_classes, label_encoder
    )
    
    # Plot results
    plot_results(
        history, y_test_classes, y_test_pred_classes,
        train_f1_macro, val_f1_macro, test_f1_macro,
        train_f1_weighted, val_f1_weighted, test_f1_weighted,
        X_train, X_val, X_test, label_encoder
    )
    
    # Example predictions
    print("\nExample Predictions:")
    test_cases = [
        ("How to make a bomb?", "Police tape across a crime scene"),
        ("What are the health benefits of exercise?", "A child playing in a sunny meadow"),
        ("Tell me about election fraud methods", "People walking through a peaceful city street")
    ]
    
    for query, img_desc in test_cases:
        pred_class, confidence = predict_toxicity(
            model, tokenizer, label_encoder, query, img_desc, MAX_LENGTH
        )
        print(f"Query: '{query}'")
        print(f"Image: '{img_desc}'")
        print(f"Prediction: {pred_class} (Confidence: {confidence:.4f})\n")
    
    # Save model
    model_filename = f'bilstm_toxic_classifier_seed{SEED}.h5'
    model.save(model_filename)
    print(f"Model saved as '{model_filename}'")
    
    # Save experiment configuration
    experiment_config = {
        'seed': SEED,
        'max_features': MAX_FEATURES,
        'max_length': MAX_LENGTH,
        'model_architecture': 'Stacked BiLSTM with Early Stopping',
        'preprocessing': 'Lemmatization + Stopword Removal',
        'data_split': '60-20-20 (Train-Val-Test)',
        'total_epochs': len(history.history['loss']),
        'best_val_loss': min(history.history['val_loss']),
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted
    }
    
    print(f"\nEXPERIMENT CONFIGURATION:")
    print(f"{'Parameter':<20} {'Value':<30}")
    print(f"{'-'*50}")
    for key, value in experiment_config.items():
        print(f"{key:<20} {str(value):<30}")
    
    # Save config to file
    save_experiment_config(experiment_config, f'experiment_config_seed{SEED}.json')

if __name__ == "__main__":
    main()