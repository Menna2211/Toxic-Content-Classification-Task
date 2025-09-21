# utils_transformer.py
import json
import torch
from data_preprocessing_transformer import preprocess_text

def predict_toxicity_transformer(model, tokenizer, label_encoder, text, device='cpu'):
    """Predict toxicity using transformer model"""
    model.eval()
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(predictions, dim=1)
    
    return label_encoder.classes_[predicted_class.item()], confidence.item()

def save_model(model, model_path):
    """Save trained model"""
    model.save_pretrained(model_path)
    print(f"Model saved to: {model_path}")

def save_experiment_config(config, file_path):
    """Save experiment configuration to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to '{file_path}'")

def create_experiment_config(seed, results_distilbert, results_albert, device):
    """Create experiment configuration dictionary"""
    return {
        'seed': seed,
        'models_trained': ['DistilBERT-LoRA', 'ALBERT-LoRA'],
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'learning_rate': 2e-4,
        'batch_size': 16,
        'epochs': 10,
        'max_length': 256,
        'distilbert_test_accuracy': results_distilbert['eval_accuracy'],
        'distilbert_test_f1_macro': results_distilbert['eval_f1_macro'],
        'distilbert_test_f1_weighted': results_distilbert['eval_f1_weighted'],
        'albert_test_accuracy': results_albert['eval_accuracy'],
        'albert_test_f1_macro': results_albert['eval_f1_macro'],
        'albert_test_f1_weighted': results_albert['eval_f1_weighted'],
        'device': str(device)
    }