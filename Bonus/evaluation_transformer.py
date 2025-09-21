# evaluation_transformer.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def compare_models(results_distilbert, results_albert, y_true_distilbert, y_pred_distilbert, 
                  y_true_albert, y_pred_albert, label_encoder):
    """Compare results of two models"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    
    print(f"\nDistilBERT with LoRA Results:")
    print(f"Test Accuracy: {results_distilbert['eval_accuracy']:.4f}")
    print(f"Test F1 Macro: {results_distilbert['eval_f1_macro']:.4f}")
    print(f"Test F1 Weighted: {results_distilbert['eval_f1_weighted']:.4f}")
    
    print(f"\nALBERT with LoRA Results:")
    print(f"Test Accuracy: {results_albert['eval_accuracy']:.4f}")
    print(f"Test F1 Macro: {results_albert['eval_f1_macro']:.4f}")
    print(f"Test F1 Weighted: {results_albert['eval_f1_weighted']:.4f}")
    
    print(f"\nDistilBERT Classification Report:")
    print(classification_report(y_true_distilbert, y_pred_distilbert, target_names=label_encoder.classes_))
    
    print(f"\nALBERT Classification Report:")
    print(classification_report(y_true_albert, y_pred_albert, target_names=label_encoder.classes_))
    
    return results_distilbert, results_albert

def plot_comparison(results_distilbert, results_albert, y_true_distilbert, y_pred_distilbert,
                   y_true_albert, y_pred_albert, label_encoder):
    """Plot comparison of model results"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Transformer Models with LoRA - Performance Comparison', fontsize=16, fontweight='bold')
    
    # F1 Score comparison
    models = ['DistilBERT', 'ALBERT']
    f1_macro_scores = [results_distilbert['eval_f1_macro'], results_albert['eval_f1_macro']]
    f1_weighted_scores = [results_distilbert['eval_f1_weighted'], results_albert['eval_f1_weighted']]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, f1_macro_scores, width, label='Macro F1', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + width/2, f1_weighted_scores, width, label='Weighted F1', alpha=0.8, color='lightcoral')
    axes[0, 0].set_title('F1 Scores Comparison')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    accuracy_scores = [results_distilbert['eval_accuracy'], results_albert['eval_accuracy']]
    axes[0, 1].bar(models, accuracy_scores, alpha=0.8, color='lightgreen')
    axes[0, 1].set_title('Accuracy Comparison')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix for DistilBERT
    cm_distilbert = confusion_matrix(y_true_distilbert, y_pred_distilbert)
    sns.heatmap(cm_distilbert, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[1, 0])
    axes[1, 0].set_title('DistilBERT Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Confusion Matrix for ALBERT
    cm_albert = confusion_matrix(y_true_albert, y_pred_albert)
    sns.heatmap(cm_albert, annot=True, fmt='d', cmap='Reds', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[1, 1])
    axes[1, 1].set_title('ALBERT Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()