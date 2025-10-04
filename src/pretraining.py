import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Tuple
import numpy as np

from .model import PretrainingModel, ContrastiveLoss

class PretrainingTrainer:
    """Trainer for novel pretraining strategies."""
    
    def __init__(self, 
                 model: PretrainingModel,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Loss functions for different tasks
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)
        self.order_loss = nn.CrossEntropyLoss()
        
        # Training statistics
        self.training_stats = {
            'mlm_losses': [],
            'contrastive_losses': [],
            'order_losses': [],
            'total_losses': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch using multiple pretraining tasks."""
        self.model.train()
        
        epoch_stats = {
            'mlm_loss': 0.0,
            'contrastive_loss': 0.0,
            'order_loss': 0.0,
            'total_loss': 0.0,
            'mlm_count': 0,
            'contrastive_count': 0,
            'order_count': 0
        }
        
        progress_bar = tqdm(dataloader, desc="Pretraining")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            task_type = batch['task_type']
            
            if task_type == 'msm':
                loss = self._masked_sequence_modeling_step(batch)
                epoch_stats['mlm_loss'] += loss.item()
                epoch_stats['mlm_count'] += 1
                
            elif task_type == 'contrastive':
                loss = self._contrastive_learning_step(batch)
                epoch_stats['contrastive_loss'] += loss.item()
                epoch_stats['contrastive_count'] += 1
                
            elif task_type == 'order':
                loss = self._order_prediction_step(batch)
                epoch_stats['order_loss'] += loss.item()
                epoch_stats['order_count'] += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_stats['total_loss'] += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'task': task_type})
        
        # Calculate averages
        total_batches = len(dataloader)
        return {
            'avg_mlm_loss': epoch_stats['mlm_loss'] / max(epoch_stats['mlm_count'], 1),
            'avg_contrastive_loss': epoch_stats['contrastive_loss'] / max(epoch_stats['contrastive_count'], 1),
            'avg_order_loss': epoch_stats['order_loss'] / max(epoch_stats['order_count'], 1),
            'avg_total_loss': epoch_stats['total_loss'] / total_batches
        }
    
    def _masked_sequence_modeling_step(self, batch: Dict) -> torch.Tensor:
        """Perform masked sequence modeling step."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids, task_type='mlm')
        
        # Compute loss only for masked tokens
        loss = self.mlm_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss
    
    def _contrastive_learning_step(self, batch: Dict) -> torch.Tensor:
        """Perform contrastive learning step."""
        pos_input_ids = batch['pos_input_ids'].to(self.device)
        neg_input_ids = batch['neg_input_ids'].to(self.device)
        
        # Get embeddings for positive and negative pairs
        pos_embeddings = self.model(pos_input_ids, task_type='contrastive')
        neg_embeddings = self.model(neg_input_ids, task_type='contrastive')
        
        # Compute contrastive loss
        loss = self.contrastive_loss(pos_embeddings, neg_embeddings)
        
        return loss
    
    def _order_prediction_step(self, batch: Dict) -> torch.Tensor:
        """Perform sequence order prediction step."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids, task_type='order')
        
        # Compute loss
        loss = self.order_loss(logits, labels)
        
        return loss
    
    def train(self, 
              dataloader: DataLoader, 
              num_epochs: int = 10,
              save_path: str = None) -> Dict[str, list]:
        """Train the model for multiple epochs."""
        
        print(f"Starting pretraining for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train for one epoch
            epoch_stats = self.train_epoch(dataloader)
            
            # Update training statistics
            self.training_stats['mlm_losses'].append(epoch_stats['avg_mlm_loss'])
            self.training_stats['contrastive_losses'].append(epoch_stats['avg_contrastive_loss'])
            self.training_stats['order_losses'].append(epoch_stats['avg_order_loss'])
            self.training_stats['total_losses'].append(epoch_stats['avg_total_loss'])
            
            # Print epoch statistics
            print(f"Avg MLM Loss: {epoch_stats['avg_mlm_loss']:.4f}")
            print(f"Avg Contrastive Loss: {epoch_stats['avg_contrastive_loss']:.4f}")
            print(f"Avg Order Loss: {epoch_stats['avg_order_loss']:.4f}")
            print(f"Avg Total Loss: {epoch_stats['avg_total_loss']:.4f}")
            
            # Save checkpoint
            if save_path and (epoch + 1) % 5 == 0:
                checkpoint_path = f"{save_path}_epoch_{epoch + 1}.pt"
                self.save_model(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        if save_path:
            final_path = f"{save_path}_final.pt"
            self.save_model(final_path)
            print(f"Saved final model: {final_path}")
        
        return self.training_stats
    
    def save_model(self, path: str):
        """Save model state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.model.get_encoder_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)

def pretrain_model(dataloader: DataLoader,
                   vocab_size: int,
                   num_epochs: int = 20,
                   d_model: int = 256,
                   nhead: int = 8,
                   num_layers: int = 6,
                   learning_rate: float = 1e-4,
                   device: torch.device = None,
                   save_path: str = None) -> Tuple[PretrainingModel, Dict[str, list]]:
    """
    Pretrain a transformer model using novel pretraining strategies.
    
    Args:
        dataloader: DataLoader for pretraining data
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
    
    print(f"Using device: {device}")
    
    # Create model
    model = PretrainingModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    # Create trainer
    trainer = PretrainingTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Train model
    training_stats = trainer.train(
        dataloader=dataloader,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    return model, training_stats

def evaluate_pretraining_tasks(model: PretrainingModel,
                              dataloader: DataLoader,
                              device: torch.device) -> Dict[str, float]:
    """Evaluate pretraining tasks."""
    model.eval()
    
    task_accuracies = {
        'mlm_accuracy': [],
        'order_accuracy': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating pretraining"):
            task_type = batch['task_type'][0] if isinstance(batch['task_type'], list) else batch['task_type']
            
            if task_type == 'msm':
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, task_type='mlm')
                predictions = logits.argmax(dim=-1)
                
                # Calculate accuracy only for masked tokens
                mask = labels != -100
                if mask.sum() > 0:
                    correct = (predictions[mask] == labels[mask]).float().mean()
                    task_accuracies['mlm_accuracy'].append(correct.item())
            
            elif task_type == 'order':
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, task_type='order')
                predictions = logits.argmax(dim=-1)
                
                correct = (predictions == labels).float().mean()
                task_accuracies['order_accuracy'].append(correct.item())
    
    # Calculate average accuracies
    results = {}
    for task, accuracies in task_accuracies.items():
        if accuracies:
            results[task] = np.mean(accuracies)
        else:
            results[task] = 0.0
    
    return results
