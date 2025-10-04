import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Tuple, Optional
import numpy as np

from .model import TCRAntigenClassifier, load_pretrained_encoder

class ClassificationTrainer:
    """Trainer for TCR-antigen interaction classification."""
    
    def __init__(self, 
                 model: TCRAntigenClassifier,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 class_weights: Optional[torch.Tensor] = None):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
        
        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training statistics
        self.training_stats = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
        
        self.best_val_accuracy = 0.0
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Get batch data
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Create attention mask (True for padding tokens)
            attention_mask = (input_ids == 0)  # Assuming 0 is PAD token
            
            # Forward pass
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': f'{current_acc:.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get batch data
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Create attention mask
                attention_mask = (input_ids == 0)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              num_epochs: int = 50,
              save_path: str = None,
              early_stopping_patience: int = 10) -> Dict[str, list]:
        """Train the model for multiple epochs with validation."""
        
        print(f"Starting training for {num_epochs} epochs...")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Update statistics
            self.training_stats['train_losses'].append(train_loss)
            self.training_stats['train_accuracies'].append(train_acc)
            self.training_stats['val_losses'].append(val_loss)
            self.training_stats['val_accuracies'].append(val_acc)
            
            # Print statistics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if save_path:
                    best_path = f"{save_path}_best.pt"
                    self.save_model(best_path)
                    print(f"Saved best model: {best_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
            
            # Save regular checkpoint
            if save_path and (epoch + 1) % 10 == 0:
                checkpoint_path = f"{save_path}_epoch_{epoch + 1}.pt"
                self.save_model(checkpoint_path)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation accuracy: {self.best_val_accuracy:.4f}")
        
        return self.training_stats
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on a dataset."""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = (input_ids == 0)
                
                logits = self.model(input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(logits, dim=-1)
                predictions = logits.argmax(dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def save_model(self, path: str):
        """Save model state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats,
            'best_val_accuracy': self.best_val_accuracy
        }, path)
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)

def train_baseline_model(train_loader: DataLoader,
                        val_loader: DataLoader,
                        vocab_size: int,
                        num_epochs: int = 50,
                        d_model: int = 256,
                        nhead: int = 8,
                        num_layers: int = 6,
                        learning_rate: float = 1e-4,
                        device: torch.device = None,
                        save_path: str = None) -> Tuple[TCRAntigenClassifier, Dict[str, list]]:
    """
    Train a baseline model without pretraining.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab_size: Size of vocabulary
        num_epochs: Number of training epochs
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        learning_rate: Learning rate
        device: Training device
        save_path: Path to save the model
    
    Returns:
        Tuple of (trained model, training statistics)
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training baseline model on device: {device}")
    
    # Create model
    model = TCRAntigenClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    # Calculate class weights for balanced training
    class_counts = torch.zeros(2)
    for batch in train_loader:
        labels = batch['labels']
        for label in labels:
            class_counts[label] += 1
    
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 2  # Normalize
    
    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # Train model
    training_stats = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    return model, training_stats

def train_pretrained_model(train_loader: DataLoader,
                          val_loader: DataLoader,
                          vocab_size: int,
                          pretrained_model_path: str,
                          num_epochs: int = 30,
                          d_model: int = 256,
                          nhead: int = 8,
                          num_layers: int = 6,
                          learning_rate: float = 5e-5,
                          device: torch.device = None,
                          save_path: str = None) -> Tuple[TCRAntigenClassifier, Dict[str, list]]:
    """
    Train a model initialized with pretrained weights.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab_size: Size of vocabulary
        pretrained_model_path: Path to pretrained model
        num_epochs: Number of training epochs
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        learning_rate: Learning rate (lower for fine-tuning)
        device: Training device
        save_path: Path to save the model
    
    Returns:
        Tuple of (trained model, training statistics)
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training pretrained model on device: {device}")
    
    # Create model
    model = TCRAntigenClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    # Load pretrained weights
    model = load_pretrained_encoder(model, pretrained_model_path, device)
    
    # Calculate class weights
    class_counts = torch.zeros(2)
    for batch in train_loader:
        labels = batch['labels']
        for label in labels:
            class_counts[label] += 1
    
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 2
    
    # Create trainer with lower learning rate for fine-tuning
    trainer = ClassificationTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # Train model
    training_stats = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    return model, training_stats
