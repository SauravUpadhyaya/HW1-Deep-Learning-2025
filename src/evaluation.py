import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import os

# Try to import seaborn, fallback to matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .model import TCRAntigenClassifier

class ModelEvaluator:
    """Comprehensive model evaluation for TCR-antigen interaction prediction."""
    
    def __init__(self, model: TCRAntigenClassifier, device: torch.device):
        self.model = model.to(device)
        self.device = device
    
    def evaluate(self, data_loader: DataLoader, dataset_name: str = "test") -> Dict[str, float]:
        """
        Comprehensive evaluation of the model.
        
        Args:
            data_loader: DataLoader for the dataset to evaluate
            dataset_name: Name of the dataset (for logging)
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating on {dataset_name} set...")
        
        # Get predictions and true labels
        y_true, y_pred, y_prob = self._get_predictions(data_loader)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        
        # Print results
        self._print_results(metrics, dataset_name)
        
        return metrics
    
    def _get_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions for the entire dataset."""
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Getting predictions"):
                # Get batch data
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Create attention mask
                attention_mask = (input_ids == 0)  # True for padding tokens
                
                # Forward pass
                logits = self.model(input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(logits, dim=-1)
                predictions = logits.argmax(dim=-1)
                
                # Store results
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        return y_true, y_pred, y_prob
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic accuracy
        accuracy = (y_true == y_pred).mean()
        
        # AUC score (main metric for this task)
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Balanced accuracy
        sensitivity = recall  # Same as recall
        balanced_accuracy = (sensitivity + specificity) / 2
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def _print_results(self, metrics: Dict[str, float], dataset_name: str):
        """Print formatted evaluation results."""
        print(f"\n{dataset_name.upper()} SET RESULTS:")
        print("=" * 50)
        print(f"AUC Score:           {metrics['auc_score']:.4f}")
        print(f"Accuracy:            {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
        print(f"Precision:           {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"Specificity:         {metrics['specificity']:.4f}")
        print(f"F1 Score:            {metrics['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(f"True Positives:  {metrics['true_positives']:4d}")
        print(f"True Negatives:  {metrics['true_negatives']:4d}")
        print(f"False Positives: {metrics['false_positives']:4d}")
        print(f"False Negatives: {metrics['false_negatives']:4d}")
    
    def plot_confusion_matrix(self, data_loader: DataLoader, save_path: str = None):
        """Plot confusion matrix."""
        y_true, y_pred, _ = self._get_predictions(data_loader)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['No Interaction', 'Interaction'],
                        yticklabels=['No Interaction', 'Interaction'])
        else:
            # Fallback to matplotlib
            plt.imshow(cm, cmap='Blues', interpolation='nearest')
            plt.colorbar()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center')
            plt.xticks([0, 1], ['No Interaction', 'Interaction'])
            plt.yticks([0, 1], ['No Interaction', 'Interaction'])
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, data_loader: DataLoader, save_path: str = None, model_name: str = "Model") -> Tuple[float, float, float]:
        """
        Plot ROC curve and return AUC score with confidence intervals.
        
        Args:
            data_loader: DataLoader for the dataset
            save_path: Path to save the ROC curve plot
            model_name: Name of the model (for plot title)
        
        Returns:
            Tuple of (AUC score, False Positive Rate array, True Positive Rate array)
        """
        # Get predictions
        y_true, y_pred, y_prob = self._get_predictions(data_loader)
        
        # Calculate ROC curve (use positive class probabilities)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Add some key points
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
                label=f'Optimal Point (threshold={optimal_threshold:.3f})')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        plt.show()
        
        return auc_score, fpr, tpr
    
    def analyze_predictions(self, data_loader: DataLoader, save_path: str = None) -> pd.DataFrame:
        """Analyze model predictions with sequence information."""
        self.model.eval()
        
        results = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Analyzing predictions"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                antigens = batch['antigen']
                tcrs = batch['tcr']
                
                attention_mask = (input_ids == 0)
                logits = self.model(input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(logits, dim=-1)
                predictions = logits.argmax(dim=-1)
                
                # Store detailed results
                for i in range(len(labels)):
                    results.append({
                        'antigen': antigens[i],
                        'tcr': tcrs[i],
                        'true_label': labels[i].item(),
                        'predicted_label': predictions[i].item(),
                        'interaction_probability': probabilities[i, 1].item(),
                        'no_interaction_probability': probabilities[i, 0].item(),
                        'correct_prediction': labels[i].item() == predictions[i].item()
                    })
        
        df = pd.DataFrame(results)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Detailed predictions saved to: {save_path}")
        
        return df

def compare_models(baseline_results: Dict[str, float], 
                  pretrained_results: Dict[str, float],
                  save_path: str = None) -> Dict[str, float]:
    """
    Compare baseline and pretrained model results.
    
    Args:
        baseline_results: Results from baseline model
        pretrained_results: Results from pretrained model
        save_path: Path to save comparison results
    
    Returns:
        Dictionary containing improvement metrics
    """
    
    print("\nMODEL COMPARISON:")
    print("=" * 60)
    
    improvements = {}
    
    for metric in ['auc_score', 'accuracy', 'precision', 'recall', 'f1_score']:
        baseline_val = baseline_results.get(metric, 0.0)
        pretrained_val = pretrained_results.get(metric, 0.0)
        improvement = pretrained_val - baseline_val
        improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0.0
        
        improvements[f'{metric}_improvement'] = improvement
        improvements[f'{metric}_improvement_pct'] = improvement_pct
        
        print(f"{metric.upper()}:")
        print(f"  Baseline:    {baseline_val:.4f}")
        print(f"  Pretrained:  {pretrained_val:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)")
        print()
    
    # Summary
    auc_improvement = improvements['auc_score_improvement']
    auc_improvement_pct = improvements['auc_score_improvement_pct']
    
    print("SUMMARY:")
    print(f"AUC Score Improvement: {auc_improvement:.4f} ({auc_improvement_pct:+.2f}%)")
    
    if auc_improvement > 0:
        print("✓ Pretraining provides performance gain!")
    else:
        print("✗ Pretraining does not improve performance.")
    
    # Save results
    if save_path:
        comparison_data = {
            'baseline_results': baseline_results,
            'pretrained_results': pretrained_results,
            'improvements': improvements
        }
        
        # Save as text file
        with open(save_path, 'w') as f:
            f.write("MODEL COMPARISON RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            for metric in ['auc_score', 'accuracy', 'precision', 'recall', 'f1_score']:
                baseline_val = baseline_results.get(metric, 0.0)
                pretrained_val = pretrained_results.get(metric, 0.0)
                improvement = improvements[f'{metric}_improvement']
                improvement_pct = improvements[f'{metric}_improvement_pct']
                
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Baseline:    {baseline_val:.4f}\n")
                f.write(f"  Pretrained:  {pretrained_val:.4f}\n")
                f.write(f"  Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"AUC Score Improvement: {auc_improvement:.4f} ({auc_improvement_pct:+.2f}%)\n")
            
            if auc_improvement > 0:
                f.write("SUCCESS: Pretraining provides performance gain!\n")
            else:
                f.write("FAILURE: Pretraining does not improve performance.\n")
        
        print(f"Comparison results saved to: {save_path}")
    
    return improvements

def plot_roc_comparison(baseline_evaluator: ModelEvaluator, 
                       pretrained_evaluator: ModelEvaluator,
                       data_loader: DataLoader,
                       save_path: str = None) -> Dict[str, float]:
    """
    Compare ROC curves between baseline and pretrained models.
    
    Args:
        baseline_evaluator: ModelEvaluator for baseline model
        pretrained_evaluator: ModelEvaluator for pretrained model
        data_loader: DataLoader for evaluation
        save_path: Path to save the comparison plot
    
    Returns:
        Dictionary containing AUC scores for both models
    """
    
    # Get predictions for both models
    print("Getting baseline model predictions...")
    baseline_y_true, _, baseline_y_prob = baseline_evaluator._get_predictions(data_loader)
    
    print("Getting pretrained model predictions...")
    pretrained_y_true, _, pretrained_y_prob = pretrained_evaluator._get_predictions(data_loader)
    
    # Calculate ROC curves (use positive class probabilities)
    baseline_fpr, baseline_tpr, _ = roc_curve(baseline_y_true, baseline_y_prob[:, 1])
    baseline_auc = roc_auc_score(baseline_y_true, baseline_y_prob[:, 1])
    
    pretrained_fpr, pretrained_tpr, _ = roc_curve(pretrained_y_true, pretrained_y_prob[:, 1])
    pretrained_auc = roc_auc_score(pretrained_y_true, pretrained_y_prob[:, 1])
    
    # Plot comparison
    plt.figure(figsize=(10, 8))
    
    plt.plot(baseline_fpr, baseline_tpr, linewidth=2, 
             label=f'Baseline Model (AUC = {baseline_auc:.4f})', color='blue')
    plt.plot(pretrained_fpr, pretrained_tpr, linewidth=2, 
             label=f'Pretrained Model (AUC = {pretrained_auc:.4f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison: Baseline vs Pretrained Model')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotation
    auc_improvement = pretrained_auc - baseline_auc
    improvement_pct = (auc_improvement / baseline_auc * 100) if baseline_auc > 0 else 0.0
    
    plt.text(0.6, 0.2, f'AUC Improvement: {auc_improvement:.4f} ({improvement_pct:+.2f}%)', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC comparison plot saved to: {save_path}")
    
    plt.show()
    
    return {
        'baseline_auc': baseline_auc,
        'pretrained_auc': pretrained_auc,
        'auc_improvement': auc_improvement,
        'auc_improvement_pct': improvement_pct
    }

def plot_training_curves(baseline_stats: Dict[str, List], 
                        pretrained_stats: Dict[str, List],
                        save_path: str = None):
    """Plot training curves for comparison."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    ax1.plot(baseline_stats['train_losses'], label='Baseline', color='blue')
    ax1.plot(pretrained_stats['train_losses'], label='Pretrained', color='red')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training accuracy
    ax2.plot(baseline_stats['train_accuracies'], label='Baseline', color='blue')
    ax2.plot(pretrained_stats['train_accuracies'], label='Pretrained', color='red')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Validation loss
    ax3.plot(baseline_stats['val_losses'], label='Baseline', color='blue')
    ax3.plot(pretrained_stats['val_losses'], label='Pretrained', color='red')
    ax3.set_title('Validation Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    # Validation accuracy
    ax4.plot(baseline_stats['val_accuracies'], label='Baseline', color='blue')
    ax4.plot(pretrained_stats['val_accuracies'], label='Pretrained', color='red')
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()
