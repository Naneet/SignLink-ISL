import torch
import torch.nn as nn
from torchvision import models

# CNN + GRU Model with Dropouts
class SignLanguageClassifier(nn.Module):
    def __init__(self, num_classes, dropout):
        super(SignLanguageClassifier, self).__init__()

        # Pre-trained CNN (e.g., ResNet)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Remove the final classification layer

        # Dropout after CNN features
        self.cnn_dropout = nn.Dropout(p=dropout)

        # GRU for temporal modeling
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)

        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)

        # Dropout before final FC layer
        self.fc_dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
        - x: Input tensor with shape [Batch, Sequence, Channels, Height, Width]

        Returns:
        - output: Model predictions with shape [Batch, Num_Classes]
        """
        batch_size, seq_len, c, h, w = x.size()  # x: [Batch, Sequence, Channels, Height, Width]

        # Reshape input for CNN processing
        x = x.view(batch_size * seq_len, c, h, w)  # Combine Batch and Sequence dimensions: [Batch * Sequence, Channels, Height, Width]

        # Extract features using CNN
        features = self.cnn(x)  # [Batch * Sequence, 512, 1, 1] (ResNet-18 output)
        features = features.view(features.size(0), -1)  # Flatten: [Batch * Sequence, 512]

        # Apply CNN dropout
        features = self.cnn_dropout(features)

        # Reshape features back to sequence format for GRU
        features = features.view(batch_size, seq_len, -1)  # [Batch, Sequence, Features]

        # Pass the sequence of features through the GRU
        gru_out, _ = self.gru(features)  # [Batch, Sequence, Hidden_Size]

        # Take the output of the last time step
        final_output = gru_out[:, -1, :]  # [Batch, Hidden_Size]

        # Apply dropout before the fully connected layer
        final_output = self.fc_dropout(final_output)

        # Pass through the fully connected layer
        output = self.fc(final_output)  # [Batch, Num_Classes]
        return output