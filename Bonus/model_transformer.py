# model_transformer.py
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import f1_score

def get_lora_config(model_type="distilbert"):
    """Get LoRA configuration for different model types"""
    if model_type == "distilbert":
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin", "k_lin", "out_lin"]
        )
    elif model_type == "albert":
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value", "key", "output"]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def tokenize_datasets(dataset_dict, tokenizer, max_length=256):
    """Tokenize datasets using the provided tokenizer"""
    def tokenize_function(examples):
        return tokenizer(
            examples['combined_text'],
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
    
    tokenized_datasets = {
        split: dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['combined_text']
        )
        for split, dataset in dataset_dict.items()
    }
    
    return tokenized_datasets

def setup_tokenizer(model_name):
    """Setup tokenizer with proper padding token"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return tokenizer

def setup_model(model_name, num_classes, lora_config):
    """Setup model with LoRA configuration"""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        if model.get_input_embeddings().num_embeddings < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    accuracy = np.mean(predictions == labels)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def train_transformer_model(model_name, dataset_dict, num_classes, lora_config, 
                          learning_rate=2e-4, batch_size=16, epochs=10, 
                          output_dir="./output", seed=42):
    """Train a transformer model with LoRA"""
    print(f"\n{'='*60}")
    print(f"Training {model_name} with LoRA")
    print(f"{'='*60}")
    
    # Setup tokenizer and model
    tokenizer = setup_tokenizer(model_name)
    tokenized_datasets = tokenize_datasets(dataset_dict, tokenizer)
    model = setup_model(model_name, num_classes, lora_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir='./logs',
        logging_steps=10,
        report_to=None,
        seed=seed,
        fp16=torch.cuda.is_available(),
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    test_results = trainer.evaluate(tokenized_datasets['test'])
    test_predictions = trainer.predict(tokenized_datasets['test'])
    y_true = test_predictions.label_ids
    y_pred = np.argmax(test_predictions.predictions, axis=1)
    
    return model, trainer, test_results, y_true, y_pred, tokenizer