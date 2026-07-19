"""
Classification Model Heads for Breast Cancer Detection.

Implements:
1. ResNet50 classifier with custom head
2. EfficientNetV2-B0 classifier with custom head
3. Soft Voting Ensemble combining both models
4. Trust Score computation for clinical confidence

Both classifiers use ImageNet-1K pretrained backbones with frozen early layers
and custom classification heads designed for binary (Benign vs Malignant) diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional


class ResNet50Classifier(nn.Module):
    """
    ResNet50 backbone with custom classification head for breast cancer detection.

    Architecture:
        ResNet50 (ImageNet pretrained) → AdaptiveAvgPool2d
        → Dropout(0.5) → Linear(2048, 512) → ReLU → BatchNorm
        → Dropout(0.3) → Linear(512, num_classes)

    The first 6 layers of ResNet50 are frozen to leverage pretrained features
    while allowing fine-tuning of deeper layers for medical domain adaptation.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5, freeze_layers: int = 6):
        super().__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze early layers for transfer learning
        layers = list(self.backbone.children())
        for i, layer in enumerate(layers[:freeze_layers]):
            for param in layer.parameters():
                param.requires_grad = False

        # Extract feature dimension from backbone
        in_features = self.backbone.fc.in_features  # 2048

        # Replace the classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)

    def get_feature_extractor(self) -> nn.Module:
        """Returns the backbone without the classification head (for Grad-CAM)."""
        return nn.Sequential(*list(self.backbone.children())[:-1])

    def get_last_conv_layer(self) -> nn.Module:
        """Returns the last convolutional layer for Grad-CAM extraction."""
        return self.backbone.layer4[-1]


class EfficientNetV2B0Classifier(nn.Module):
    """
    EfficientNetV2-B0 backbone with custom classification head.

    Architecture:
        EfficientNetV2-B0 (ImageNet pretrained) → AdaptiveAvgPool2d
        → Dropout(0.5) → Linear(1280, 512) → ReLU → BatchNorm
        → Dropout(0.3) → Linear(512, num_classes)

    EfficientNetV2 uses progressive learning and compound scaling for
    excellent accuracy-efficiency tradeoff. Ideal for medical imaging
    where computational resources may be limited.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        # Load pretrained EfficientNetV2-B0
        self.backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )

        # Freeze feature extractor layers (keep classifier trainable)
        for param in self.backbone.features[:5].parameters():
            param.requires_grad = False

        # Extract feature dimension
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)

    def get_last_conv_layer(self) -> nn.Module:
        """Returns the last convolutional layer for Grad-CAM extraction."""
        return self.backbone.features[-1]


class EnsembleClassifier(nn.Module):
    """
    Soft Voting Ensemble combining ResNet50 and EfficientNetV2-B0.

    The ensemble combines logits from both models using learned weights:
        Final = w_resnet × logits_resnet + w_efficient × logits_efficient

    The weights are initialized based on expected performance but can be
    dynamically adjusted from validation results.

    Additionally computes a Trust Score:
        - HIGH: Both models agree AND confidence > 90%
        - MEDIUM: Models agree but confidence 70-90%
        - LOW: Models disagree OR confidence < 70% → flag for human review
    """

    def __init__(
        self,
        num_classes: int = 2,
        resnet_weight: float = 0.0,
        efficient_weight: float = 1.0,
        resnet_path: Optional[str] = None,
        efficient_path: Optional[str] = None,
    ):
        super().__init__()

        self.resnet = ResNet50Classifier(num_classes=num_classes)
        self.efficient = EfficientNetV2B0Classifier(num_classes=num_classes)

        # Ensemble weights (can be learned or set from validation)
        self.resnet_weight = resnet_weight
        self.efficient_weight = efficient_weight

        # Load pretrained weights if provided
        if resnet_path:
            state_dict = torch.load(resnet_path, map_location="cpu", weights_only=True)
            self.resnet.load_state_dict(state_dict)
            print(f"[Ensemble] ResNet50 weights loaded from: {resnet_path}")

        if efficient_path:
            state_dict = torch.load(efficient_path, map_location="cpu", weights_only=True)
            self.efficient.load_state_dict(state_dict)
            print(f"[Ensemble] EfficientNetV2 weights loaded from: {efficient_path}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both models with soft voting ensemble.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Dictionary containing:
                - 'logits': Final ensemble logits (B, num_classes)
                - 'probabilities': Softmax probabilities (B, num_classes)
                - 'prediction': Predicted class index (B,)
                - 'confidence': Confidence of prediction (B,)
                - 'resnet_logits': Raw ResNet50 logits
                - 'efficient_logits': Raw EfficientNet logits
                - 'trust_score': Trust level string per sample
        """
        # Get individual model outputs
        resnet_logits = self.resnet(x)
        efficient_logits = self.efficient(x)

        # Weighted soft voting
        ensemble_logits = (
            self.resnet_weight * resnet_logits
            + self.efficient_weight * efficient_logits
        )

        # Probabilities and predictions
        probabilities = F.softmax(ensemble_logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)

        # Individual model predictions for trust scoring
        resnet_pred = resnet_logits.argmax(dim=1)
        efficient_pred = efficient_logits.argmax(dim=1)

        # Compute trust scores
        trust_scores = self._compute_trust_scores(
            resnet_pred, efficient_pred, confidence
        )

        return {
            "logits": ensemble_logits,
            "probabilities": probabilities,
            "prediction": prediction,
            "confidence": confidence,
            "resnet_logits": resnet_logits,
            "efficient_logits": efficient_logits,
            "trust_score": trust_scores,
        }

    def _compute_trust_scores(
        self,
        resnet_pred: torch.Tensor,
        efficient_pred: torch.Tensor,
        confidence: torch.Tensor,
    ) -> list:
        """
        Compute trust scores based on model agreement and confidence.

        Trust Levels:
            HIGH:   Both models agree AND confidence > 90%
            MEDIUM: Both models agree AND confidence 70-90%
            LOW:    Models disagree OR confidence < 70%

        Args:
            resnet_pred: ResNet50 predictions (B,)
            efficient_pred: EfficientNet predictions (B,)
            confidence: Ensemble confidence scores (B,)

        Returns:
            List of trust level strings
        """
        trust_scores = []
        models_agree = (resnet_pred == efficient_pred)

        for i in range(len(confidence)):
            conf = confidence[i].item()
            agree = models_agree[i].item()

            if agree and conf >= 0.90:
                trust_scores.append("HIGH")
            elif agree and conf >= 0.70:
                trust_scores.append("MEDIUM")
            else:
                trust_scores.append("LOW")

        return trust_scores

    def update_weights(self, resnet_accuracy: float, efficient_accuracy: float):
        """
        Dynamically update ensemble weights based on validation performance.

        Args:
            resnet_accuracy: ResNet50 validation accuracy
            efficient_accuracy: EfficientNet validation accuracy
        """
        total = resnet_accuracy + efficient_accuracy
        self.resnet_weight = resnet_accuracy / total
        self.efficient_weight = efficient_accuracy / total
        print(
            f"[Ensemble] Updated weights — "
            f"ResNet: {self.resnet_weight:.3f}, "
            f"EfficientNet: {self.efficient_weight:.3f}"
        )


def build_ensemble(
    num_classes: int = 2,
    resnet_path: Optional[str] = None,
    efficient_path: Optional[str] = None,
) -> EnsembleClassifier:
    """
    Factory function to build the ensemble classifier.

    Args:
        num_classes: Number of output classes (2 for Benign/Malignant)
        resnet_path: Path to pretrained ResNet50 weights
        efficient_path: Path to pretrained EfficientNetV2 weights

    Returns:
        Configured EnsembleClassifier
    """
    model = EnsembleClassifier(
        num_classes=num_classes,
        resnet_path=resnet_path,
        efficient_path=efficient_path,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Ensemble] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model


if __name__ == "__main__":
    # Quick validation
    model = build_ensemble(num_classes=2)
    dummy_input = torch.randn(4, 3, 224, 224)
    results = model(dummy_input)

    print(f"\nInput shape:      {dummy_input.shape}")
    print(f"Logits shape:     {results['logits'].shape}")
    print(f"Probabilities:    {results['probabilities']}")
    print(f"Predictions:      {results['prediction']}")
    print(f"Confidence:       {results['confidence']}")
    print(f"Trust scores:     {results['trust_score']}")
