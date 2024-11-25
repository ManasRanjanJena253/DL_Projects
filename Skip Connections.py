# Importing dependencies

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining our CNN model with skip connections
class SkipConnections(nn.Module):
    def __init__(self):
        super().__init__()   # Initialising the base class.
        # Defining convolutional layers :

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)
        # Kernel_size is the dimension of the filter used for convolutions.
        # Padding adds 1 pixel padding or outer layer to the original image to ensure output dimensions are preserved.
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)

        # Defining batch normalisation ::
        # Batch normalisation ensures that the output of a layer has a mean of 0 and a standard deviation of 1 before passing it to a new layer .
        # Prevents large changes in activations as they propagate through the network, reducing the risk of exploding and vanishing gradients.
        # It improves generalisation by reducing overfitting by introducing noise due to batch-wise computation.
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

        # Defining the fully connected layer
        self.fc = nn.Linear(in_features = 64, out_features = 10)  # Here 10 is the output classes which are setting for this example model class.

    # Defining the forward pass
    def forward(self, x):
        # First convolution
        x1 = F.relu(self.bn1(self.conv1(x)))
        # Self.conv1 applies the first convolution and the batch normalisation is added to the output and then ReLu activation function is applied to add non-linearity.

        # Second convolution with skip connections
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.conv3(x2) # Third convolution

        # Adding the skip connection ( Element-wise addition)
        x2 += x1  # Adds the input x1 ( from the previous layer) to the output x2 of the deeper layer. This is the skip connection. It allows gradients to flow directly through the network, improving training for deeper models.

        # Global average pooling(GAP) before the final fully connected layer
        # GAP helps to reduce spatial dimensions (height and width) by converting them into single number (channel). This results in a low dimension dataset.
        x2 = F.adaptive_avg_pool2d(x2, (1,1))

        # Feeding the data into the final fully connected layer
        x2 = torch.flatten(x2, start_dim = 1)  # Flattens the tensor from 4D (batch, height, width, colour_channels) to 2D tensor(batch, features).It is an important step as the fully connected layer doesn't take a 4D tensor as an input and throws an error. It also helps in dimensionality reduction.
        output = self.fc(x2)

        return output

# Using the model on a random tensor
model = SkipConnections()
print(model)

dummy_input = torch.randn(1, 3, 32, 32) # Creates a random image having (batch_size, colour_channels, height, width)

print(f"Shape of the output given by the model :: {model(dummy_input).shape}")

