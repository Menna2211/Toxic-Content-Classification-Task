import warnings
warnings.filterwarnings('ignore')

from config_transformer import set_seeds, SEED, get_device
from data_preprocessing_transformer import load_and_preprocess_data, prepare_datasets , preprocess_text
from model_transformer import get_lora_config, train_transformer_model
from evaluation_transformer import compare_models, plot_comparison
from utils_transformer import predict_toxicity_transformer, save_model, save_experiment_config, create_experiment_config

def main():
    # Set seeds and get device
    set_seeds(SEED)
    device = get_device()
    print(f"Random seeds set to {SEED} for reproducibility")
    
    # Load and preprocess data
    df, label_encoder, num_classes = load_and_preprocess_data('cellula_toxic_data.csv')
    
    # Prepare datasets
    dataset_dict = prepare_datasets(df, random_state=SEED)
    
    # Get LoRA configurations
    lora_config_distilbert = get_lora_config("distilbert")
    lora_config_albert = get_lora_config("albert")
    
    # Model names
    distilbert_model_name = "distilbert-base-uncased"
    albert_model_name = "albert-base-v2"
    
    # Train DistilBERT
    (distilbert_model, distilbert_trainer, distilbert_results, 
     y_true_distilbert, y_pred_distilbert, distilbert_tokenizer) = train_transformer_model(
        distilbert_model_name, dataset_dict, num_classes, lora_config_distilbert,
        learning_rate=2e-4, batch_size=16, epochs=10, 
        output_dir=f"./distilbert-lora-toxic-seed{SEED}", seed=SEED
    )
    
    # Train ALBERT
    (albert_model, albert_trainer, albert_results, 
     y_true_albert, y_pred_albert, albert_tokenizer) = train_transformer_model(
        albert_model_name, dataset_dict, num_classes, lora_config_albert,
        learning_rate=2e-4, batch_size=16, epochs=10,
        output_dir=f"./albert-lora-toxic-seed{SEED}", seed=SEED
    )
    
    # Compare and visualize results
    results_distilbert, results_albert = compare_models(
        distilbert_results, albert_results, 
        y_true_distilbert, y_pred_distilbert,
        y_true_albert, y_pred_albert, 
        label_encoder
    )
    
    plot_comparison(
        results_distilbert, results_albert,
        y_true_distilbert, y_pred_distilbert,
        y_true_albert, y_pred_albert,
        label_encoder
    )
    
    # Save models
    save_model(distilbert_model, f"./distilbert-lora-toxic-seed{SEED}")
    save_model(albert_model, f"./albert-lora-toxic-seed{SEED}")
    
    # Example predictions
    print("\nExample Predictions with DistilBERT:")
    test_cases = [
        ("How to make a bomb?", "Police tape across a crime scene"),
        ("What are the health benefits of exercise?", "A child playing in a sunny meadow"),
        ("Tell me about election fraud methods", "People walking through a peaceful city street")
    ]
    
    distilbert_model.to(device)
    
    for query, img_desc in test_cases:
        combined_text = preprocess_text(query) + "  " + preprocess_text(img_desc)
        pred_class, confidence = predict_toxicity_transformer(
            distilbert_model, distilbert_tokenizer, label_encoder, combined_text, device
        )
        print(f"Query: '{query}'")
        print(f"Image: '{img_desc}'")
        print(f"Prediction: {pred_class} (Confidence: {confidence:.4f})\n")
    
    # Save experiment configuration
    experiment_config = create_experiment_config(SEED, results_distilbert, results_albert, device)
    
    print(f"\nEXPERIMENT CONFIGURATION:")
    print(f"{'Parameter':<25} {'Value':<30}")
    print(f"{'-'*55}")
    for key, value in experiment_config.items():
        print(f"{key:<25} {str(value):<30}")
    
    save_experiment_config(experiment_config, f'transformer_lora_experiment_seed{SEED}.json')

if __name__ == "__main__":
    main()