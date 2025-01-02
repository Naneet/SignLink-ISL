import torch
import torch.nn as nn
import torch.nn.functional as F

class BUMBLEBEE(nn.Module):
    def __init__(self, num_frames, keypoint_dim, num_classes, d_model=128, num_heads=4, num_layers=2, dropout=0.1, name='bumblebee'):
        super(BUMBLEBEE, self).__init__()

        # model_name for save_checkpoint
        self.name = name

        # Embedding layer to project input keypoints to d_model dimensions
        self.embedding = nn.Linear(2 * keypoint_dim, d_model)  # Input is (x, y) for each keypoint
        
        # Positional encoding for temporal sequence
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_frames, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_frames, 2, keypoint_dim)
        """
        batch_size, num_frames, _, keypoint_dim = x.shape

        # Flatten (x, y) coordinates per frame: (batch_size, num_frames, 2, keypoint_dim) -> (batch_size, num_frames, 2*keypoint_dim)
        x = x.view(batch_size, num_frames, -1)

        # Embedding and positional encoding
        x = self.embedding(x)  # (batch_size, num_frames, d_model)
        x = x + self.positional_encoding[:, :num_frames, :]

        # Transformer encoder
        x = self.transformer(x)  # (batch_size, num_frames, d_model)

        # Global average pooling over frames
        x = torch.mean(x, dim=1)  # (batch_size, d_model)

        # Classification head
        x = self.fc(x)  # (batch_size, num_classes)
        return x
