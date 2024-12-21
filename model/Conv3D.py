import torch
import torch.nn as nn
from torchvision import models

# CNN Model
class SignLanguageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageClassifier, self).__init__()

        # CNN (ResNet - 3D)
        self.cnn = models.video.r3d_18(pretrained=True)  # Pretrained 3D ResNet
        self.cnn.fc = nn.Identity()  # Remove final classification layer

        # Fully connected layer for classification
        self.fc = nn.Linear(512, num_classes)  # r3d_18 final layer outputs 512 features

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
        - x: Input tensor with shape [Batch, Sequence, Channels, Height, Width]

        Returns:
        - output: Model predictions with shape [Batch, Num_Classes]
        """
        batch_size, seq_len, c, h, w = x.size()  # x: [Batch, Sequence, Channels, Height, Width]

        # Reshape sequence as depth for 3D CNN
        x = x.permute(0, 2, 1, 3, 4)  # [Batch, Channels, Depth (Seq_Len), Height, Width]

        # Extract features using CNN
        features = self.cnn(x)  # [Batch, 512, 1, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten: [Batch, 512]

        # Pass through the fully connected layer
        output = self.fc(features)  # [Batch, Num_Classes]
        return output
