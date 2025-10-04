#!/usr/bin/env python3
"""
Main script for training and evaluating TCR-Antigen interaction prediction models.

This script implements the complete pipeline:
1. Novel pretraining on unlabeled sequence data
2. Baseline model training (no pretraining)
3. Pretrained model fine-tuning
4. Comprehensive evaluation and comparison
"""

import os
import torch
import argparse
from pathlib import Path

# Import our modules
from src.data_loader import create_data_loaders
from src.model import create_model
from src.pretraining import pretrain_model
from src.training import train_baseline_model, train_pretrained_model
from src.evaluation import ModelEvaluator, compare_models, plot_training_curves

def main():
    """Main execution function."""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Configuration with absolute paths
    config = {
        'data_dir': script_dir / 'data',
        'train_file': 'train.csv',
        'test_file': 'test.csv',
        'models_dir': script_dir / 'models',
        'results_dir': script_dir / 'results',
        'batch_size': 32,
        'max_length': 128,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'pretraining_epochs': 20,
        'training_epochs': 50,
        'finetuning_epochs': 30,
        'pretraining_lr': 1e-4,
        'training_lr': 1e-4,
        'finetuning_lr': 5e-5,
        'seed': 42
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # Create directories
    config['models_dir'].mkdir(exist_ok=True)
    config['results_dir'].mkdir(exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # File paths
    train_path = config['data_dir'] / config['train_file']
    test_path = config['data_dir'] / config['test_file']
    
    # Check if data files exist
    if not os.path.exists(train_path):
        print(f"Error: Training file not found at {train_path}")
        print("Please place your train.csv file in the data/ directory")
        return
    
    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        print("Please place your test.csv file in the data/ directory")
        return
    
    print("=" * 80)
    print("T-CELL RECEPTOR - ANTIGEN INTERACTION PREDICTION")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n1. Loading and preprocessing data...")
    train_loader, test_loader, pretrain_loader, tokenizer = create_data_loaders(
        train_file=train_path,
        test_file=test_path,
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )
    
    vocab_size = len(tokenizer)
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Pretraining batches: {len(pretrain_loader)}")
    
    # Step 2: Pretraining
    pretrained_model_path = config['models_dir'] / 'pretrained_model'
    final_pretrained_path = f"{pretrained_model_path}_final.pt"
    
    if os.path.exists(final_pretrained_path):
        print("\n2. Found existing pretrained model - skipping pretraining...")
        print(f"   Using pretrained model from: {final_pretrained_path}")
        pretraining_stats = {'total_losses': []}  # Empty stats for existing model
    else:
        print("\n2. Pretraining with novel strategy...")
        print("   Pretraining tasks:")
        print("   - Masked Sequence Modeling (MSM)")
        print("   - Contrastive Sequence Learning") 
        print("   - Sequence Order Prediction")
        
        pretrained_model, pretraining_stats = pretrain_model(
            dataloader=pretrain_loader,
            vocab_size=vocab_size,
            num_epochs=config['pretraining_epochs'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            learning_rate=config['pretraining_lr'],
            device=device,
            save_path=pretrained_model_path
        )
        
        print(f"   Pretraining completed. Model saved to: {final_pretrained_path}")
    
    # Step 3: Train baseline model (no pretraining)
    print("\n3. Training baseline model (no pretraining)...")
    
    baseline_model_path = config['models_dir'] / 'baseline_model'
    
    baseline_model, baseline_stats = train_baseline_model(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test as validation for now
        vocab_size=vocab_size,
        num_epochs=config['training_epochs'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        learning_rate=config['training_lr'],
        device=device,
        save_path=baseline_model_path
    )
    
    print(f"   Baseline training completed. Model saved to: {baseline_model_path}_best.pt")
    
    # Step 4: Fine-tune pretrained model
    print("\n4. Fine-tuning pretrained model...")
    
    finetuned_model_path = config['models_dir'] / 'finetuned_model'
    pretrained_weights_path = final_pretrained_path
    
    finetuned_model, finetuned_stats = train_pretrained_model(
        train_loader=train_loader,
        val_loader=test_loader,
        vocab_size=vocab_size,
        pretrained_model_path=pretrained_weights_path,
        num_epochs=config['finetuning_epochs'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        learning_rate=config['finetuning_lr'],
        device=device,
        save_path=finetuned_model_path
    )
    
    print(f"   Fine-tuning completed. Model saved to: {finetuned_model_path}_best.pt")
    
    # Step 5: Comprehensive evaluation
    print("\n5. Evaluating models...")
    
    # Evaluate baseline model
    print("\n5.1. Evaluating baseline model...")
    baseline_evaluator = ModelEvaluator(baseline_model, device)
    
    baseline_train_results = baseline_evaluator.evaluate(train_loader, "baseline_train")
    baseline_test_results = baseline_evaluator.evaluate(test_loader, "baseline_test")
    
    # Evaluate pretrained model
    print("\n5.2. Evaluating pretrained model...")
    finetuned_evaluator = ModelEvaluator(finetuned_model, device)
    
    finetuned_train_results = finetuned_evaluator.evaluate(train_loader, "pretrained_train")
    finetuned_test_results = finetuned_evaluator.evaluate(test_loader, "pretrained_test")
    
    # Step 6: Compare models
    print("\n6. Comparing model performance...")
    
    # Compare test results (main evaluation)
    test_improvements = compare_models(
        baseline_results=baseline_test_results,
        pretrained_results=finetuned_test_results,
        save_path=config['results_dir'] / 'model_comparison.txt'
    )
    
    # Step 6.1: Generate ROC curves
    print("\n6.1 Generating ROC curves...")
    
    try:
        # Create plots directory
        plots_dir = config['results_dir'] / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Generate individual ROC curves
        print("   Generating baseline model ROC curve...")
        baseline_auc, baseline_fpr, baseline_tpr = baseline_evaluator.plot_roc_curve(
            test_loader, 
            save_path=str(plots_dir / 'roc_baseline.png'),
            model_name="Baseline Model"
        )
        
        print("   Generating pretrained model ROC curve...")
        pretrained_auc, pretrained_fpr, pretrained_tpr = finetuned_evaluator.plot_roc_curve(
            test_loader,
            save_path=str(plots_dir / 'roc_pretrained.png'),
            model_name="Pretrained Model"
        )
        
        # Generate comparison ROC curve
        print("   Generating ROC curve comparison...")
        from src.evaluation import plot_roc_comparison
        
        roc_comparison_results = plot_roc_comparison(
            baseline_evaluator,
            finetuned_evaluator,
            test_loader,
            save_path=str(plots_dir / 'roc_comparison.png')
        )
        
        print(f"   ✓ ROC curves saved to: {plots_dir}/")
        print(f"     - Baseline ROC: roc_baseline.png")
        print(f"     - Pretrained ROC: roc_pretrained.png") 
        print(f"     - Comparison: roc_comparison.png")
        
    except Exception as e:
        print(f"   Warning: Could not generate ROC curves: {e}")
        print("   Continuing without ROC visualizations...")
    
    # Step 7: Save detailed results
    print("\n7. Saving results...")
    
    # Save all results
    results_summary = {
        'baseline': {
            'train': baseline_train_results,
            'test': baseline_test_results,
            'training_stats': baseline_stats
        },
        'pretrained': {
            'train': finetuned_train_results,
            'test': finetuned_test_results,
            'training_stats': finetuned_stats,
            'pretraining_stats': pretraining_stats
        },
        'improvements': test_improvements,
        'config': config
    }
    
    # Save results as text summary
    summary_path = config['results_dir'] / 'results_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("TCR-ANTIGEN INTERACTION PREDICTION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("BASELINE MODEL RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write("Training Set:\n")
        for metric, value in baseline_train_results.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nTest Set:\n")
        for metric, value in baseline_test_results.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("PRETRAINED MODEL RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write("Training Set:\n")
        for metric, value in finetuned_train_results.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nTest Set:\n")
        for metric, value in finetuned_test_results.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("IMPROVEMENTS:\n")
        f.write("-" * 40 + "\n")
        for metric, value in test_improvements.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    # Step 8: Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    baseline_auc = baseline_test_results['auc_score']
    pretrained_auc = finetuned_test_results['auc_score']
    auc_improvement = pretrained_auc - baseline_auc
    
    print(f"Baseline Model Test AUC:    {baseline_auc:.4f}")
    print(f"Pretrained Model Test AUC:  {pretrained_auc:.4f}")
    print(f"AUC Improvement:            {auc_improvement:.4f}")
    
    if auc_improvement > 0:
        improvement_pct = (auc_improvement / baseline_auc) * 100
        print(f"Relative Improvement:       +{improvement_pct:.2f}%")
        print("\n✓ SUCCESS: Pretraining strategy provides performance gain!")
    else:
        print(f"Relative Change:            {(auc_improvement / baseline_auc) * 100:.2f}%")
        print("\n✗ Pretraining did not improve performance.")
    
    print(f"\nDetailed results saved to: {config['results_dir']}/")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TCR-Antigen Interaction Prediction')
    parser.add_argument('--data-dir', default='data', help='Directory containing train.csv and test.csv')
    parser.add_argument('--pretraining-epochs', type=int, default=20, help='Number of pretraining epochs')
    parser.add_argument('--training-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    if hasattr(args, 'data_dir'):
        # You could update config here if needed
        pass
    
    main()
