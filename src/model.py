import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0).expand(x.size(0), -1, -1)
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """Multi-layer transformer encoder."""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_length: int = 128):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Scale embedding by sqrt(d_model)
        self.embedding_scale = math.sqrt(d_model)
        
    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        # src shape: (batch_size, seq_len)
        src = self.embedding(src) * self.embedding_scale  # (batch_size, seq_len, d_model)
        src = self.pos_encoding(src)  # Apply positional encoding
        
        # Create attention mask for padding tokens
        if src_key_padding_mask is None:
            src_key_padding_mask = (src.sum(dim=-1) == 0)  # True for padding tokens
        
        output = self.transformer(src, src_key_padding_mask=src_key_padding_mask)
        return output

class TCRAntigenClassifier(nn.Module):
    """Transformer-based classifier for TCR-antigen interaction prediction."""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_length: int = 128,
                 num_classes: int = 2):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Get encoder output
        encoder_output = self.encoder(input_ids, src_key_padding_mask=attention_mask)
        
        # Use [SOS] token representation (first token) for classification
        cls_representation = encoder_output[:, 0, :]  # (batch_size, d_model)
        cls_representation = self.dropout(cls_representation)
        
        # Classification
        logits = self.classifier(cls_representation)
        return logits

class PretrainingModel(nn.Module):
    """Model for pretraining tasks."""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_length: int = 128):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length
        )
        
        # Heads for different pretraining tasks
        self.mlm_head = nn.Linear(d_model, vocab_size)  # Masked language modeling
        self.contrastive_head = nn.Linear(d_model, d_model)  # Contrastive learning
        self.order_head = nn.Sequential(  # Sequence order prediction
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, task_type: str = 'mlm', **kwargs):
        encoder_output = self.encoder(input_ids)
        
        if task_type == 'mlm':
            return self.mlm_head(encoder_output)
        elif task_type == 'contrastive':
            cls_repr = encoder_output[:, 0, :]  # Use [SOS] token
            return self.contrastive_head(cls_repr)
        elif task_type == 'order':
            cls_repr = encoder_output[:, 0, :]  # Use [SOS] token
            cls_repr = self.dropout(cls_repr)
            return self.order_head(cls_repr)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_encoder_state_dict(self):
        """Get encoder state dict for transfer learning."""
        return self.encoder.state_dict()

class ContrastiveLoss(nn.Module):
    """Contrastive loss for pretraining."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pos_embeddings: torch.Tensor, neg_embeddings: torch.Tensor):
        # pos_embeddings: (batch_size, d_model)
        # neg_embeddings: (batch_size, d_model)
        
        # Normalize embeddings
        pos_embeddings = F.normalize(pos_embeddings, p=2, dim=1)
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=1)
        
        # Compute similarities
        pos_sim = torch.sum(pos_embeddings * pos_embeddings, dim=1) / self.temperature
        neg_sim = torch.sum(pos_embeddings * neg_embeddings, dim=1) / self.temperature
        
        # Contrastive loss
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)

def create_model(vocab_size: int, 
                d_model: int = 256,
                nhead: int = 8,
                num_layers: int = 6,
                dropout: float = 0.1,
                max_length: int = 128,
                model_type: str = 'classifier'):
    """Create model based on type."""
    
    if model_type == 'classifier':
        return TCRAntigenClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            max_length=max_length
        )
    elif model_type == 'pretraining':
        return PretrainingModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            max_length=max_length
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_pretrained_encoder(classifier_model: TCRAntigenClassifier, 
                           pretrained_model_path: str,
                           device: torch.device):
    """Load pretrained encoder weights into classifier."""
    pretrained_state = torch.load(pretrained_model_path, map_location=device)
    
    # Extract encoder weights
    if 'encoder' in pretrained_state:
        encoder_state = pretrained_state['encoder']
    else:
        # Filter encoder weights from full model state
        encoder_state = {k.replace('encoder.', ''): v 
                        for k, v in pretrained_state.items() 
                        if k.startswith('encoder.')}
    
    # Load encoder weights
    classifier_model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded pretrained encoder from {pretrained_model_path}")
    
    return classifier_model
