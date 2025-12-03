# src/model_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeClassifierCNN(nn.Module):
    """
    A custom CNN designed to classify cropped image patches of a glass slide edge,
    focusing on preserving spatial details for detecting subtle discrepancies.
    """
    def __init__(self, input_channels=3, num_classes=2, input_size=96):
        """
        Initializes the CNN layers.

        Args:
            input_channels (int): Number of channels in the input image (e.g., 3 for RGB, 1 for Grayscale).
            num_classes (int): Number of output classes (e.g., 2 for Success/Failure).
            input_size (int): Assumed square size of the input image patch (e.g., 64, 96, 128).
                               This is needed to calculate the flattened size before dense layers.
        """
        super(EdgeClassifierCNN, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # --- Convolutional Blocks ---
        # We use stride 1 convolutions and limited pooling to maintain resolution

        # Block 1: Capture initial features
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1) # padding=1 keeps size same for 3x3 kernel
        self.bn1 = nn.BatchNorm2d(32)
        # No pooling yet

        # Block 2: More features, first downsampling
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Size: input_size / 2

        # Block 3: Increase feature depth
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # No pooling

        # Block 4: More features, second downsampling
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Size: input_size / 4

        # Block 5: Further feature extraction
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        # No pooling here to keep more spatial info before flattening

        # --- Calculate Flattened Size ---
        # After two 2x2 pooling layers, the spatial dimensions are input_size / 4
        # The number of feature maps is 128 (from conv5)
        flattened_size = 128 * (input_size // 4) * (input_size // 4)
        if flattened_size <= 0:
             raise ValueError(f"Calculated flattened size is zero or negative ({flattened_size}). "
                              f"Check input_size ({input_size}) and pooling layers.")

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(flattened_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128) # BatchNorm for dense layers
        self.dropout = nn.Dropout(0.5) # Dropout for regularization

        self.fc2 = nn.Linear(128, num_classes) # Output layer

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) containing raw logits.
        """
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        # Conv Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        # Conv Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        # Conv Block 5
        x = F.relu(self.bn5(self.conv5(x)))

        # Flatten
        x = torch.flatten(x, 1) # Flatten all dimensions except batch

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) # Raw logits output

        # Note: Softmax is typically applied *outside* the model, often within the loss function
        # (like nn.CrossEntropyLoss) or separately during inference.
        return x

# --- Example Usage ---
if __name__ == '__main__':
    # Example: Create a model instance for 96x96 RGB images and 2 classes
    input_channels = 3
    num_classes = 2
    input_size = 96

    model = EdgeClassifierCNN(input_channels=input_channels, num_classes=num_classes, input_size=input_size)
    print("Model Architecture:")
    print(model)

    # Create a dummy input tensor (batch_size=4, channels=3, height=96, width=96)
    dummy_input = torch.randn(4, input_channels, input_size, input_size)

    # Pass the dummy input through the model
    try:
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape (logits): {output.shape}") # Should be (4, 2)
        # Verify output shape matches (batch_size, num_classes)
        assert output.shape == (4, num_classes)
        print("\nModel forward pass successful!")
    except Exception as e:
        print(f"\nError during model forward pass: {e}")

