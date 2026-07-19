"""
Attention U-Net Architecture for Breast Ultrasound Tumor Segmentation.

This implements the Attention U-Net with Attention Gates on skip connections
to suppress irrelevant background tissue and highlight tumor margins.

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv2d → BN → ReLU → Conv2d → BN → ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.

    Learns to suppress irrelevant features from the encoder (skip connection)
    and highlight salient features relevant to the decoder output.

    AG(x_skip, g_decoder) = σ(ψ(ReLU(W_x · x + W_g · g + b_g))) ⊙ x_skip
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        # Transform gating signal (from decoder)
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        # Transform skip connection (from encoder)
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Gating signal from decoder path (lower resolution)
            skip: Skip connection from encoder path (higher resolution)

        Returns:
            Attention-weighted skip connection features
        """
        # Align spatial dimensions — upsample gate to match skip
        g = self.W_gate(gate)
        x = self.W_skip(skip)

        # If spatial sizes don't match, interpolate gate signal
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)

        # Compute attention: σ(ψ(ReLU(W_g·g + W_x·x)))
        attention = self.relu(g + x)
        attention = self.psi(attention)

        # Apply attention to skip connection
        return skip * attention


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock → MaxPool2d"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block: Upsample → AttentionGate → Concat → ConvBlock"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.attention_gate = AttentionGate(
            gate_channels=in_channels // 2,
            skip_channels=skip_channels,
            inter_channels=skip_channels // 2,
        )
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample decoder features
        x = self.upsample(x)

        # Pad if sizes don't match exactly (due to odd dimensions)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        # Apply attention gate to skip connection
        skip_attended = self.attention_gate(gate=x, skip=skip)

        # Concatenate and convolve
        x = torch.cat([x, skip_attended], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary tumor segmentation.

    Architecture:
        Encoder: 4 downsampling blocks (1→64→128→256→512)
        Bottleneck: 512→1024
        Decoder: 4 upsampling blocks with Attention Gates (1024→512→256→128→64)
        Output: 1×1 conv → Sigmoid (binary mask)

    Input:  (B, 1, 256, 256) — grayscale ultrasound crop
    Output: (B, 1, 256, 256) — binary segmentation mask
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: list = None):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        # Encoder path
        self.encoder1 = EncoderBlock(in_channels, features[0])  # 1 → 64
        self.encoder2 = EncoderBlock(features[0], features[1])  # 64 → 128
        self.encoder3 = EncoderBlock(features[1], features[2])  # 128 → 256
        self.encoder4 = EncoderBlock(features[2], features[3])  # 256 → 512

        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)  # 512 → 1024

        # Decoder path with Attention Gates
        self.decoder4 = DecoderBlock(features[3] * 2, features[3], features[3])  # 1024 → 512
        self.decoder3 = DecoderBlock(features[3], features[2], features[2])      # 512 → 256
        self.decoder2 = DecoderBlock(features[2], features[1], features[1])      # 256 → 128
        self.decoder1 = DecoderBlock(features[1], features[0], features[0])      # 128 → 64

        # Final 1x1 convolution for binary segmentation
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Attention U-Net.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Sigmoid-activated binary mask of shape (B, 1, H, W)
        """
        # Encoder
        skip1, x = self.encoder1(x)  # skip1: (B, 64, H, W)
        skip2, x = self.encoder2(x)  # skip2: (B, 128, H/2, W/2)
        skip3, x = self.encoder3(x)  # skip3: (B, 256, H/4, W/4)
        skip4, x = self.encoder4(x)  # skip4: (B, 512, H/8, W/8)

        # Bottleneck
        x = self.bottleneck(x)  # (B, 1024, H/16, W/16)

        # Decoder with attention-gated skip connections
        x = self.decoder4(x, skip4)  # (B, 512, H/8, W/8)
        x = self.decoder3(x, skip3)  # (B, 256, H/4, W/4)
        x = self.decoder2(x, skip2)  # (B, 128, H/2, W/2)
        x = self.decoder1(x, skip1)  # (B, 64, H, W)

        # Final segmentation map
        return torch.sigmoid(self.final_conv(x))


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross-Entropy Loss.

    This combination provides:
    - Dice Loss: Handles class imbalance (tumor pixels << background pixels)
    - BCE Loss: Provides stable per-pixel gradients

    Final Loss = α × DiceLoss + β × BCELoss
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model output after sigmoid, shape (B, 1, H, W)
            targets: Ground truth binary mask, shape (B, 1, H, W)

        Returns:
            Combined loss scalar
        """
        # Dice Loss
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)

        intersection = (predictions_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice

        # BCE Loss
        bce_loss = self.bce(predictions, targets)

        # Combined
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def build_attention_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    pretrained_path: str = None,
) -> AttentionUNet:
    """
    Factory function to build an Attention U-Net.

    Args:
        in_channels: Number of input channels (1 for grayscale ultrasound)
        out_channels: Number of output channels (1 for binary segmentation)
        pretrained_path: Path to pretrained weights (.pth file)

    Returns:
        Configured AttentionUNet model
    """
    model = AttentionUNet(in_channels=in_channels, out_channels=out_channels)

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[AttentionUNet] Loaded weights from: {pretrained_path}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[AttentionUNet] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model


if __name__ == "__main__":
    # Quick validation
    model = build_attention_unet(in_channels=1, out_channels=1)
    dummy_input = torch.randn(2, 1, 256, 256)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
