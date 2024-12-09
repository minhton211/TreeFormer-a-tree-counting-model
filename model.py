from functools import partial
import torch.nn as nn
import torch

from detectron2.modeling import ViT
from detectron2.checkpoint import DetectionCheckpointer

from model_utils import *


class TreeVision(nn.Module):
    """
    TreeVision is a deep learning model combining a ViT (Vision Transformer) frontend 
    with a dilated convolutional backend for generating density maps.
    """

    def __init__(self, load_weights=False):
        """
        Initialize the TreeVision model.

        Args:
            load_weights (bool): Flag to indicate whether to load pretrained weights for the frontend.
        """
        super(TreeVision, self).__init__()

        # Vision Transformer (ViT) as the feature extractor (frontend)
        embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
        self.frontend = ViT(
            img_size=1024,              # Input image size
            patch_size=16,              # Patch size for ViT
            embed_dim=embed_dim,        # Embedding dimension
            depth=depth,                # Number of transformer blocks
            num_heads=num_heads,        # Number of attention heads
            drop_path_rate=dp,          # Stochastic depth rate
            window_size=14,             # Window size for local attention
            mlp_ratio=4,                # Ratio of MLP hidden dim to embedding dim
            qkv_bias=True,              # Allow bias in QKV projections
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # Specify indexes for local attention blocks
                0, 1, 3, 4, 6, 7, 9, 10,
            ],
            residual_block_indexes=[],  # No residual blocks
            use_rel_pos=True,           # Use relative positional encoding
            out_feature="last_feat",    # Output feature name
        )

        # Load pretrained weights if specified
        if load_weights:
            DetectionCheckpointer(self.frontend).load(
                "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
            )

        # Backend: Convolutional layers for refining features
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)

        # Output layer: Convolution to produce final prediction
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Convolutional block for region proposal processing
        self.conv = self._get_rpn_conv(in_channels=768, out_channels=512, norm="LN")

    def _get_rpn_conv(self, in_channels, out_channels, norm):
        """
        Create the region proposal convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            norm (str): Type of normalization layer to use.

        Returns:
            nn.Sequential: Convolutional block for feature transformation.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),  # Upsampling
            Conv2d(
                in_channels // 2,
                out_channels,
                kernel_size=1,
                bias=True,
                norm=get_norm(norm, out_channels),
            ),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                norm=get_norm(norm, out_channels),
            ),
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        x = self.frontend(x)["last_feat"]  # Extract features from ViT
        x = self.conv(x)                   # Apply RPN convolution
        x = self.backend(x)                # Process features through the backend
        x = self.output_layer(x)           # Generate final output
        return x

    def _initialize_weights(self):
        """
        Initialize weights for convolutional and linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
