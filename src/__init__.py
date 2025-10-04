"""
TCR-Antigen Interaction Prediction Package

This package implements a transformer-based model with novel pretraining strategies
for predicting T-cell receptor (TCR) and antigen interactions.

Modules:
- data_loader: Data preprocessing and loading utilities
- model: Transformer model implementations
- pretraining: Novel pretraining strategies implementation
- training: Model training and fine-tuning utilities
- evaluation: Model evaluation and comparison tools
"""

__version__ = "1.0.0"
__author__ = "Saurav Upadhyaya, 2025"

# Import main classes for easy access
from .data_loader import TCRAntigenDataset, PretrainingDataset, create_data_loaders
from .model import TCRAntigenClassifier, PretrainingModel, create_model
from .pretraining import PretrainingTrainer, pretrain_model
from .training import ClassificationTrainer, train_baseline_model, train_pretrained_model
from .evaluation import ModelEvaluator, compare_models

__all__ = [
    'TCRAntigenDataset',
    'PretrainingDataset', 
    'create_data_loaders',
    'TCRAntigenClassifier',
    'PretrainingModel',
    'create_model',
    'PretrainingTrainer',
    'pretrain_model',
    'ClassificationTrainer',
    'train_baseline_model',
    'train_pretrained_model',
    'ModelEvaluator',
    'compare_models'
]
