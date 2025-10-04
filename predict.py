#!/usr/bin/env python3
"""
Inference script for TCR-Antigen interaction prediction using trained models.

This script allows you to make predictions using either:
1. Baseline model (trained without pretraining)
2. Pretrained model (trained with novel pretraining strategy)

Usage:
    python predict.py --model baseline --data test.csv
    python predict.py --model pretrained --data test.csv
    python predict.py --model both --data test.csv
"""

import os
import torch
import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

from src.data_loader import TCRAntigenDataset, create_data_loaders
from src.model import TCRAntigenClassifier
from src.evaluation import ModelEvaluator

def load_model(model_path: str, vocab_size: int, device: torch.device) -> TCRAntigenClassifier:
    """Load a trained model from checkpoint."""
    
    # Create model architecture
    model = TCRAntigenClassifier(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        max_length=128
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from: {model_path}")
    return model

def make_predictions(model: TCRAntigenClassifier, 
                    data_loader, 
                    device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions using the loaded model."""
    
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = (input_ids == 0)
            
            logits = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

def create_prediction_report(data_path: str,
                           predictions: np.ndarray,
                           probabilities: np.ndarray,
                           model_name: str,
                           save_path: str = None) -> pd.DataFrame:
    """Create a detailed prediction report."""
    
    # Load original data
    df = pd.read_csv(data_path)
    
    # Create report
    report_df = pd.DataFrame({
        'antigen': df['antigen'].values,
        'TCR': df['TCR'].values,
        'true_interaction': df['interaction'].values if 'interaction' in df.columns else None,
        'predicted_interaction': predictions,
        'no_interaction_prob': probabilities[:, 0],
        'interaction_prob': probabilities[:, 1],
        # 'confidence_score': [prob[pred] for pred, prob in zip(predictions, probabilities)],  # Add confidence score
        'model_used': model_name
    })
    
    # Add confidence level
    confidence_levels = []
    for pred, prob in zip(predictions, probabilities):
        pred_prob = prob[pred]  # Use the probability of the predicted class
        if pred_prob >= 0.7:
            confidence_levels.append('High')
        elif pred_prob >= 0.4:
            confidence_levels.append('Medium')
        else:
            confidence_levels.append('Low')
    report_df['confidence'] = confidence_levels
    
    if save_path:
        report_df.to_csv(save_path, index=False)
        print(f"✓ Prediction report saved to: {save_path}")
    
    return report_df

def evaluate_predictions(true_labels: np.ndarray,
                        predictions: np.ndarray,
                        probabilities: np.ndarray,
                        model_name: str) -> Dict:
    """Evaluate predictions if true labels are available."""
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
        'f1_score': f1_score(true_labels, predictions),
        'auc_score': roc_auc_score(true_labels, probabilities[:, 1])
    }
    
    print(f"\n{model_name.upper()} MODEL PERFORMANCE:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='TCR-Antigen Interaction Prediction')
    parser.add_argument('--model', choices=['baseline', 'pretrained', 'both'], 
                       default='both', help='Model to use for prediction')
    parser.add_argument('--data', required=True, help='Path to CSV file with data')
    parser.add_argument('--output', help='Output path for predictions (optional)')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Evaluate predictions if true labels available')
    parser.add_argument('--plot_roc', action='store_true',
                       help='Generate ROC curves (requires true labels and evaluate flag)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent.absolute()
    
    # Check if data file exists
    data_path = args.data
    if not os.path.exists(data_path):
        # Try relative to script directory
        data_path = script_dir / args.data
        if not os.path.exists(data_path):
            print(f"Error: Data file not found: {args.data}")
            return
    
    print(f"Loading data from: {data_path}")
    
    # Create tokenizer and dataset
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<SEP>', '<MASK>']
    vocab = special_tokens + amino_acids
    tokenizer = {token: idx for idx, token in enumerate(vocab)}
    vocab_size = len(tokenizer)
    
    # Create dataset and dataloader
    dataset = TCRAntigenDataset(str(data_path), tokenizer, max_length=128)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Model paths
    models_dir = script_dir / 'models'
    baseline_path = models_dir / 'baseline_model_best.pt'
    pretrained_path = models_dir / 'finetuned_model_best.pt'
    
    results = {}
    
    # Run predictions based on arguments
    if args.model in ['baseline', 'both']:
        if baseline_path.exists():
            print(f"\n🤖 Running BASELINE model predictions...")
            baseline_model = load_model(str(baseline_path), vocab_size, device)
            baseline_preds, baseline_probs = make_predictions(baseline_model, data_loader, device)
            
            # Create report
            output_path = args.output or 'baseline_predictions.csv'
            if args.model == 'both':
                output_path = output_path.replace('.csv', '_baseline.csv')
            
            baseline_report = create_prediction_report(
                str(data_path), baseline_preds, baseline_probs, 'Baseline', output_path
            )
            results['baseline'] = (baseline_preds, baseline_probs, baseline_report)
            
            # Evaluate if true labels available and requested
            if args.evaluate and 'interaction' in baseline_report.columns:
                true_labels = baseline_report['true_interaction'].values
                evaluate_predictions(true_labels, baseline_preds, baseline_probs, 'Baseline')
                
                # Generate ROC curve if requested
                if args.plot_roc:
                    from src.evaluation import ModelEvaluator
                    evaluator = ModelEvaluator(baseline_model, device)
                    evaluator.plot_roc_curve(data_loader, save_path='baseline_roc.png', 
                                          model_name='Baseline Model')
        else:
            print(f"❌ Baseline model not found: {baseline_path}")
    
    if args.model in ['pretrained', 'both']:
        if pretrained_path.exists():
            print(f"\n🧠 Running PRETRAINED model predictions...")
            pretrained_model = load_model(str(pretrained_path), vocab_size, device)
            pretrained_preds, pretrained_probs = make_predictions(pretrained_model, data_loader, device)
            
            # Create report
            output_path = args.output or 'pretrained_predictions.csv'
            if args.model == 'both':
                output_path = output_path.replace('.csv', '_pretrained.csv')
            
            pretrained_report = create_prediction_report(
                str(data_path), pretrained_preds, pretrained_probs, 'Pretrained', output_path
            )
            results['pretrained'] = (pretrained_preds, pretrained_probs, pretrained_report)
            
            # Evaluate if true labels available and requested
            if args.evaluate and 'interaction' in pretrained_report.columns:
                true_labels = pretrained_report['true_interaction'].values
                evaluate_predictions(true_labels, pretrained_preds, pretrained_probs, 'Pretrained')
                
                # Generate ROC curve if requested
                if args.plot_roc:
                    from src.evaluation import ModelEvaluator
                    evaluator = ModelEvaluator(pretrained_model, device)
                    evaluator.plot_roc_curve(data_loader, save_path='pretrained_roc.png', 
                                          model_name='Pretrained Model')
        else:
            print(f"❌ Pretrained model not found: {pretrained_path}")
    
    # Compare models if both were run
    if len(results) == 2 and args.evaluate:
        print(f"\n📊 MODEL COMPARISON:")
        print("=" * 50)
        baseline_preds, baseline_probs, _ = results['baseline']
        pretrained_preds, pretrained_probs, _ = results['pretrained']
        
        # Load true labels
        df = pd.read_csv(data_path)
        if 'interaction' in df.columns:
            true_labels = df['interaction'].values
            
            from sklearn.metrics import roc_auc_score
            baseline_auc = roc_auc_score(true_labels, baseline_probs[:, 1])
            pretrained_auc = roc_auc_score(true_labels, pretrained_probs[:, 1])
            improvement = pretrained_auc - baseline_auc
            improvement_pct = (improvement / baseline_auc) * 100 if baseline_auc > 0 else 0
            
            print(f"Baseline AUC:    {baseline_auc:.4f}")
            print(f"Pretrained AUC:  {pretrained_auc:.4f}")
            print(f"Improvement:     {improvement:.4f} ({improvement_pct:+.2f}%)")
            
            if improvement > 0:
                print("✅ Pretrained model performs better!")
            else:
                print("❌ Baseline model performs better.")
            
            # Generate ROC comparison if requested
            if args.plot_roc:
                print(f"\n📈 Generating ROC curve comparison...")
                from src.evaluation import ModelEvaluator, plot_roc_comparison
                
                # Load models for comparison
                baseline_model = load_model(str(baseline_path), vocab_size, device)
                pretrained_model = load_model(str(pretrained_path), vocab_size, device)
                
                baseline_evaluator = ModelEvaluator(baseline_model, device)
                pretrained_evaluator = ModelEvaluator(pretrained_model, device)
                
                roc_results = plot_roc_comparison(
                    baseline_evaluator, pretrained_evaluator, data_loader,
                    save_path='roc_comparison.png'
                )
                print(f"✓ ROC comparison saved as: roc_comparison.png")
    
    print(f"\n✅ Prediction complete!")

if __name__ == "__main__":
    main()
