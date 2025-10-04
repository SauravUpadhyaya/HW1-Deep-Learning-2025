import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import random

class TCRAntigenDataset(Dataset):
    """Dataset for T-cell receptor and antigen interaction data."""
    
    def __init__(self, csv_file: str, tokenizer=None, max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV file containing antigen, TCR, and interaction data
            tokenizer: Tokenizer for converting sequences to tokens
            max_length: Maximum sequence length for padding/truncation
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer if tokenizer else self._create_tokenizer()
        self.max_length = max_length
        
    def _create_tokenizer(self):
        """Create a simple tokenizer for amino acid sequences."""
        # 20 standard amino acids + special tokens
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<SEP>', '<MASK>']
        
        vocab = special_tokens + amino_acids
        return {token: idx for idx, token in enumerate(vocab)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        antigen = row['antigen']
        tcr = row['TCR']
        interaction = row['interaction']
        
        # Create combined sequence: <SOS>antigen<SEP>tcr
        combined_seq = f"<SOS>{antigen}<SEP>{tcr}"
        
        # Tokenize
        tokens = self._tokenize_sequence(combined_seq)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(interaction, dtype=torch.long),
            'antigen': antigen,
            'tcr': tcr
        }
    
    def _tokenize_sequence(self, sequence: str) -> List[int]:
        """Convert sequence to token IDs."""
        tokens = []
        i = 0
        while i < len(sequence):
            if sequence[i:i+5] == '<SOS>':
                tokens.append(self.tokenizer['<SOS>'])
                i += 5
            elif sequence[i:i+5] == '<SEP>':
                tokens.append(self.tokenizer['<SEP>'])
                i += 5
            elif sequence[i:i+6] == '<MASK>':
                tokens.append(self.tokenizer['<MASK>'])
                i += 6
            elif sequence[i] in self.tokenizer:
                tokens.append(self.tokenizer[sequence[i]])
                i += 1
            else:
                tokens.append(self.tokenizer['<UNK>'])
                i += 1
        
        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([self.tokenizer['<PAD>']] * (self.max_length - len(tokens)))
        
        return tokens

class PretrainingDataset(Dataset):
    """Dataset for pretraining tasks."""
    
    def __init__(self, csv_file: str, tokenizer: Dict, max_length: int = 128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data) * 3  # Multiple pretraining tasks per sample
    
    def __getitem__(self, idx):
        # Cycle through different pretraining tasks
        task_type = idx % 3
        data_idx = idx // 3
        
        if data_idx >= len(self.data):
            data_idx = data_idx % len(self.data)
            
        row = self.data.iloc[data_idx]
        antigen = row['antigen']
        tcr = row['TCR']
        
        if task_type == 0:
            return self._masked_sequence_modeling(antigen, tcr)
        elif task_type == 1:
            return self._contrastive_learning(antigen, tcr, data_idx)
        else:
            return self._sequence_order_prediction(antigen, tcr)
    
    def _masked_sequence_modeling(self, antigen: str, tcr: str):
        """Masked sequence modeling pretraining task."""
        combined_seq = f"<SOS>{antigen}<SEP>{tcr}"
        original_tokens = self._tokenize_sequence(combined_seq)
        
        # Randomly mask 15% of amino acid tokens
        masked_tokens = original_tokens.copy()
        labels = [-100] * len(original_tokens)  # -100 for ignored tokens
        
        for i, token in enumerate(original_tokens):
            if token not in [self.tokenizer['<SOS>'], self.tokenizer['<SEP>'], 
                           self.tokenizer['<PAD>']] and random.random() < 0.15:
                labels[i] = token
                if random.random() < 0.8:
                    masked_tokens[i] = self.tokenizer['<MASK>']
                elif random.random() < 0.5:
                    # Random amino acid
                    aa_tokens = [self.tokenizer[aa] for aa in 'ARNDCQEGHILKMFPSTWYV']
                    masked_tokens[i] = random.choice(aa_tokens)
                # else keep original token
        
        return {
            'input_ids': torch.tensor(masked_tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'task_type': 'msm'
        }
    
    def _contrastive_learning(self, antigen: str, tcr: str, data_idx: int):
        """Contrastive learning pretraining task."""
        # Positive pair: original antigen-TCR
        pos_seq = f"<SOS>{antigen}<SEP>{tcr}"
        pos_tokens = self._tokenize_sequence(pos_seq)
        
        # Negative pair: random TCR with same antigen
        neg_idx = random.randint(0, len(self.data) - 1)
        while neg_idx == data_idx:
            neg_idx = random.randint(0, len(self.data) - 1)
        
        neg_tcr = self.data.iloc[neg_idx]['TCR']
        neg_seq = f"<SOS>{antigen}<SEP>{neg_tcr}"
        neg_tokens = self._tokenize_sequence(neg_seq)
        
        return {
            'pos_input_ids': torch.tensor(pos_tokens, dtype=torch.long),
            'neg_input_ids': torch.tensor(neg_tokens, dtype=torch.long),
            'labels': torch.tensor(1, dtype=torch.long),  # 1 for positive pair
            'task_type': 'contrastive'
        }
    
    def _sequence_order_prediction(self, antigen: str, tcr: str):
        """Sequence order prediction pretraining task."""
        if random.random() < 0.5:
            # Correct order
            seq = f"<SOS>{antigen}<SEP>{tcr}"
            label = 1
        else:
            # Swapped order
            seq = f"<SOS>{tcr}<SEP>{antigen}"
            label = 0
        
        tokens = self._tokenize_sequence(seq)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'task_type': 'order'
        }
    
    def _tokenize_sequence(self, sequence: str) -> List[int]:
        """Convert sequence to token IDs with padding/truncation."""
        tokens = []
        i = 0
        while i < len(sequence):
            if sequence[i:i+5] == '<SOS>':
                tokens.append(self.tokenizer['<SOS>'])
                i += 5
            elif sequence[i:i+5] == '<SEP>':
                tokens.append(self.tokenizer['<SEP>'])
                i += 5
            elif sequence[i:i+6] == '<MASK>':
                tokens.append(self.tokenizer['<MASK>'])
                i += 6
            elif sequence[i] in self.tokenizer:
                tokens.append(self.tokenizer[sequence[i]])
                i += 1
            else:
                tokens.append(self.tokenizer['<UNK>'])
                i += 1
        
        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([self.tokenizer['<PAD>']] * (self.max_length - len(tokens)))
        
        return tokens

def pretraining_collate_fn(batch):
    """Custom collate function for pretraining batches with mixed task types."""
    # Group by task type
    msm_items = []
    contrastive_items = []
    order_items = []
    
    for item in batch:
        task_type = item['task_type']
        if task_type == 'msm':
            msm_items.append(item)
        elif task_type == 'contrastive':
            contrastive_items.append(item)
        elif task_type == 'order':
            order_items.append(item)
    
    # Return the largest group to maintain batch size
    if len(msm_items) >= len(contrastive_items) and len(msm_items) >= len(order_items):
        # Process MSM batch
        return {
            'input_ids': torch.stack([item['input_ids'] for item in msm_items]),
            'labels': torch.stack([item['labels'] for item in msm_items]),
            'task_type': 'msm'
        }
    elif len(contrastive_items) >= len(order_items):
        # Process contrastive batch
        return {
            'pos_input_ids': torch.stack([item['pos_input_ids'] for item in contrastive_items]),
            'neg_input_ids': torch.stack([item['neg_input_ids'] for item in contrastive_items]),
            'labels': torch.stack([item['labels'] for item in contrastive_items]),
            'task_type': 'contrastive'
        }
    else:
        # Process order batch
        return {
            'input_ids': torch.stack([item['input_ids'] for item in order_items]),
            'labels': torch.stack([item['labels'] for item in order_items]),
            'task_type': 'order'
        }

def create_data_loaders(train_file: str, test_file: str, batch_size: int = 32, max_length: int = 128):
    """Create data loaders for training and testing."""
    # Create tokenizer
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<SEP>', '<MASK>']
    vocab = special_tokens + amino_acids
    tokenizer = {token: idx for idx, token in enumerate(vocab)}
    
    # Create datasets
    train_dataset = TCRAntigenDataset(train_file, tokenizer, max_length)
    test_dataset = TCRAntigenDataset(test_file, tokenizer, max_length)
    pretrain_dataset = PretrainingDataset(train_file, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=pretraining_collate_fn)
    
    return train_loader, test_loader, pretrain_loader, tokenizer
