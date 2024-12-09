from functools import partial
import torch.nn as nn
import torch
from torch.nn import functional as F


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    Constructs a sequential block of convolutional and pooling layers.

    Args:
        cfg (list): Configuration specifying the number of output channels or 'M' for MaxPool.
        in_channels (int): Number of input channels for the first layer.
        batch_norm (bool): Whether to include BatchNorm layers.
        dilation (bool): Whether to apply dilation in convolutional layers.

    Returns:
        nn.Sequential: A sequential container of layers.
    """
    d_rate = 2 if dilation else 1  # Set dilation rate
    layers = []
    for v in cfg:
        if v == 'M':
            # Add MaxPool layer
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Add Conv2D and optionally BatchNorm and ReLU
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class LayerNorm(nn.Module):
    """
    Layer normalization for inputs with shape (batch_size, channels, height, width).
    Performs mean and variance normalization over the channel dimension.

    Args:
        normalized_shape (int): Number of channels to normalize.
        eps (float): Small value to prevent division by zero.
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """
        Forward pass for layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Compute mean and variance along the channel dimension
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        # Normalize and apply learned scale and bias
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_norm(norm, out_channels):
    """
    Retrieve the appropriate normalization layer.

    Args:
        norm (str or callable): Specifies normalization ('LN' for LayerNorm).
        out_channels (int): Number of channels for the normalization layer.

    Returns:
        nn.Module or None: The normalization layer or None if not specified.
    """
    if norm is None or (isinstance(norm, str) and len(norm) == 0):
        return None
    if isinstance(norm, str):
        norm = {
            "LN": lambda channels: LayerNorm(channels),  # Use LayerNorm for 'LN'
        }[norm]
    return norm(out_channels)


class Conv2d(nn.Conv2d):
    """
    A wrapper around torch.nn.Conv2d to add support for optional normalization and activation.

    Args:
        norm (nn.Module, optional): Normalization layer to apply after convolution.
        activation (callable, optional): Activation function to apply after normalization.
    """

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        """
        Forward pass with optional normalization and activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        # Apply convolution
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        # Apply normalization if defined
        if self.norm is not None:
            x = self.norm(x)
        # Apply activation if defined
        if self.activation is not None:
            x = self.activation(x)
        return x
