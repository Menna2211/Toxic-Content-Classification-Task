import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

def evaluate_model(model, X_train, X_val, X_test, y_train_classes, y_val_classes, y_test_classes, label_encoder):
    """Evaluate model on all datasets"""
    # Make predictions
    y_train_pred = model.predict(X_train, verbose=0)
    y_train_pred_classes = np.argmax(y_train_pred, axis=1)
    
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    
    y_test_pred = model.predict(X_test, verbose=0)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    
    # Calculate F1 scores
    train_f1_macro = f1_score(y_train_classes, y_train_pred_classes, average='macro')
    train_f1_weighted = f1_score(y_train_classes, y_train_pred_classes, average='weighted')
    
    val_f1_macro = f1_score(y_val_classes, y_val_pred_classes, average='macro')
    val_f1_weighted = f1_score(y_val_classes, y_val_pred_classes, average='weighted')
    
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
    print(classification_report(y_test_classes, y_test_pred_classes, 
                              target_names=label_encoder.classes_))
    
    return (y_train_pred_classes, y_val_pred_classes, y_test_pred_classes, 
            train_f1_macro, train_f1_weighted, val_f1_macro, val_f1_weighted, 
            test_f1_macro, test_f1_weighted)

def plot_results(history, y_test_classes, y_test_pred_classes, 
                train_f1_macro, val_f1_macro, test_f1_macro,
                train_f1_weighted, val_f1_weighted, test_f1_weighted,
                X_train, X_val, X_test, label_encoder):
    """Plot training history and evaluation metrics"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('BiLSTM Model Training & Evaluation Results', fontsize=16, fontweight='bold')
    
    # Training history - Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].axvline(x=len(history.history['loss'])-1, color='red', linestyle='--', alpha=0.7, label='Early Stop')
    axes[0, 0].set_title('Model Loss Over Time (Early Stopping)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training history - Accuracy
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
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[1, 0])
    axes[1, 0].set_title('Test Set Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # F1 Scores by class (Test Set)
    f1_per_class = f1_score(y_test_classes, y_test_pred_classes, average=None)
    axes[1, 1].bar(label_encoder.classes_, f1_per_class, color='skyblue', alpha=0.7)
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