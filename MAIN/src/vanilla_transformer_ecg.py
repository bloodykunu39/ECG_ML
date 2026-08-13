import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class VanillaTransformerECG(nn.Module):
    """
    Vanilla Transformer for 12-lead ECG Classification
    
    Args:
        num_leads: Number of ECG leads (default: 12)
        signal_length: Length of each signal (default: 5000)
        num_classes: Number of disease classes (default: 3)
        d_model: Dimension of model (default: 128)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer encoder layers (default: 4)
        dim_feedforward: Dimension of feedforward network (default: 512)
        dropout: Dropout rate (default: 0.1)
        patch_size: Size of each patch for tokenization (default: 50)
    """
    def __init__(self, 
                 num_leads=12, 
                 signal_length=5000, 
                 num_classes=3,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 patch_size=50):
        super(VanillaTransformerECG, self).__init__()
        
        self.num_leads = num_leads
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Calculate number of patches
        self.num_patches = signal_length // patch_size
        
        # Patch embedding: convert each patch to d_model dimension
        # Input: (batch, num_leads, signal_length)
        # After patching: (batch, num_leads, num_patches, patch_size)
        self.patch_embedding = nn.Linear(patch_size * num_leads, d_model)
        
        # Class token (optional, for better classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches + 1)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input shape: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, num_leads, signal_length)
        
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape into patches
        # (batch, num_leads, signal_length) -> (batch, num_patches, patch_size * num_leads)
        x = x.view(batch_size, self.num_leads, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # (batch, num_patches, num_leads, patch_size)
        x = x.reshape(batch_size, self.num_patches, -1)  # (batch, num_patches, patch_size * num_leads)
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch, num_patches, d_model)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, num_patches + 1, d_model)
        
        # Use class token for classification
        cls_output = x[:, 0]  # (batch, d_model)
        
        # Classification
        output = self.classifier(cls_output)  # (batch, num_classes)
        
        return output


class VanillaTransformerECG_Alternative(nn.Module):
    """
    Alternative approach: Treat each lead separately and aggregate
    
    This version processes each lead independently and then combines them
    """
    def __init__(self, 
                 num_leads=12, 
                 signal_length=5000, 
                 num_classes=3,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 patch_size=50):
        super(VanillaTransformerECG_Alternative, self).__init__()
        
        self.num_leads = num_leads
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Calculate number of patches per lead
        self.num_patches = signal_length // patch_size
        
        # Lead embedding to distinguish different leads
        self.lead_embedding = nn.Embedding(num_leads, d_model)
        
        # Patch embedding for each patch
        self.patch_embedding = nn.Linear(patch_size, d_model)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches * num_leads + 1)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, num_leads, signal_length)
        
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Reshape into patches
        # (batch, num_leads, signal_length) -> (batch, num_leads, num_patches, patch_size)
        x = x.view(batch_size, self.num_leads, self.num_patches, self.patch_size)
        
        # Flatten leads and patches
        x = x.view(batch_size, self.num_leads * self.num_patches, self.patch_size)
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch, num_leads * num_patches, d_model)
        
        # Add lead embeddings
        lead_ids = torch.arange(self.num_leads, device=x.device).repeat_interleave(self.num_patches)
        lead_embeds = self.lead_embedding(lead_ids)  # (num_leads * num_patches, d_model)
        x = x + lead_embeds.unsqueeze(0)  # Add lead information
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Use class token for classification
        cls_output = x[:, 0]
        
        # Classification
        output = self.classifier(cls_output)
        
        return output


# Example usage and testing
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    batch_size = 16
    num_leads = 12
    signal_length = 5000
    num_classes = 3
    
    # Create dummy data
    x = torch.randn(batch_size, num_leads, signal_length).to(device)
    y = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Test Model 1: Main approach
    print("\n" + "="*50)
    print("Testing VanillaTransformerECG (Main Approach)")
    print("="*50)
    model1 = VanillaTransformerECG(
        num_leads=num_leads,
        signal_length=signal_length,
        num_classes=num_classes,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        patch_size=50
    ).to(device)
    
    # Forward pass
    output1 = model1(x)
    print(f"Output shape: {output1.shape}")
    
    # Count parameters
    total_params1 = sum(p.numel() for p in model1.parameters())
    trainable_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params1:,}")
    print(f"Trainable parameters: {trainable_params1:,}")
    
    # Test Model 2: Alternative approach
    print("\n" + "="*50)
    print("Testing VanillaTransformerECG_Alternative")
    print("="*50)
    model2 = VanillaTransformerECG_Alternative(
        num_leads=num_leads,
        signal_length=signal_length,
        num_classes=num_classes,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        patch_size=50
    ).to(device)
    
    # Forward pass
    output2 = model2(x)
    print(f"Output shape: {output2.shape}")
    
    # Count parameters
    total_params2 = sum(p.numel() for p in model2.parameters())
    trainable_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params2:,}")
    print(f"Trainable parameters: {trainable_params2:,}")
    
    print("\n" + "="*50)
    print("Models created successfully!")
    print("="*50)
