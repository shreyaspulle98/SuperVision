"""
dino_pipeline/model.py

Vision Transformer model using DINOv3/v3 for Tc prediction.

This module implements:
- Pre-trained DINO model loading
- Feature extraction from vision transformer
- Regression head for Tc prediction
- Fine-tuning strategies (full, partial, linear probe)

Architecture:
1. Pre-trained DINO backbone (frozen or fine-tuned)
2. Feature extraction from [CLS] token or global pooling
3. MLP regression head
4. Optional dropout for regularization

Supported DINOv3 variants:
- dinov3_vits14: Small model, 16x16 patches
- dinov3_vitb14: Base model, 16x16 patches
- dinov3_vitl14: Large model, 16x16 patches
- dinov3_vitg14: Huge+ model, 16x16 patches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lora import apply_lora_to_model


class DINORegressor(nn.Module):
    """
    Vision Transformer with DINO pre-training for Tc regression.

    Uses a pre-trained DINO model as feature extractor and adds
    a regression head for predicting critical temperatures.
    """

    def __init__(
        self,
        model_name="dinov3_vitb14",
        pretrained=True,
        freeze_backbone=False,
        dropout=0.3,
        hidden_dim=512,
        use_lora=False,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1
    ):
        """
        Initialize the DINO regressor.

        Args:
            model_name (str): DINO model variant
            pretrained (bool): Use pre-trained weights
            freeze_backbone (bool): Freeze backbone during training (ignored if use_lora=True)
            dropout (float): Dropout rate in regression head
            hidden_dim (int): Hidden dimension in regression head
            use_lora (bool): Use LoRA for parameter-efficient fine-tuning
            lora_rank (int): Rank of LoRA decomposition (default: 16)
            lora_alpha (int): LoRA scaling parameter (default: 32)
            lora_dropout (float): Dropout rate for LoRA layers
        """
        super(DINORegressor, self).__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        # Load pre-trained DINO model
        self.backbone = self._load_dino_model(model_name, pretrained)

        # Get feature dimension
        self.feature_dim = self._get_feature_dim()

        # Apply LoRA or freeze backbone
        if use_lora:
            # First, freeze the entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Froze all backbone parameters")

            # Apply custom LoRA to Vision Transformer
            # Target modules: attention QKV projections in timm models
            print(f"\nApplying LoRA (rank={lora_rank}, alpha={lora_alpha})...")

            self.backbone, trainable_params, lora_params = apply_lora_to_model(
                self.backbone,
                target_module_names=["attn.qkv"],  # Target attention QKV in timm ViT
                r=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )

            # Print parameter statistics
            total_params = sum(p.numel() for p in self.backbone.parameters())
            print(f"\nLoRA Statistics:")
            print(f"  Rank: {lora_rank}")
            print(f"  Alpha: {lora_alpha}")
            print(f"  LoRA parameters: {lora_params:,}")
            print(f"  Total backbone trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

            self.freeze_backbone = False  # LoRA manages this
        elif freeze_backbone:
            # Traditional linear probing: freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Backbone frozen. Only training regression head (linear probing).")
        else:
            # Full fine-tuning: train all parameters
            print(f"Full fine-tuning enabled (training all parameters).")

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _load_dino_model(self, model_name, pretrained):
        """
        Loads pre-trained Vision Transformer model from timm.

        Args:
            model_name (str): Model variant name
            pretrained (bool): Use pre-trained weights

        Returns:
            nn.Module: ViT backbone model
        """
        try:
            import timm

            # Map our model names to timm model names (DINOv3 models)
            timm_model_map = {
                "dinov3_vits14": "vit_small_patch16_dinov3",
                "dinov3_vitb14": "vit_base_patch16_dinov3",
                "dinov3_vitl14": "vit_large_patch16_dinov3",
                "dinov3_vitg14": "vit_huge_plus_patch16_dinov3"
            }

            timm_name = timm_model_map.get(model_name, "vit_base_patch16_dinov3")

            # Load model with timm (removes classification head automatically)
            model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head, keep features only
                img_size=224
            )

            print(f"Loaded {timm_name} from timm library")
            return model

        except Exception as e:
            print(f"Warning: Could not load DINOv3 from timm: {e}")
            print("Falling back to untrained ViT-B/16")
            import timm
            model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=0,
                img_size=224
            )
            return model

    def _get_feature_dim(self):
        """
        Gets the feature dimension from the backbone.

        Returns:
            int: Feature dimension
        """
        # Feature dimensions for different DINO variants
        feature_dims = {
            "dinov3_vits14": 384,
            "dinov3_vitb14": 768,
            "dinov3_vitl14": 1024,
            "dinov3_vitg14": 1536
        }

        return feature_dims.get(self.model_name, 768)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Predicted Tc values [batch_size, 1]
        """
        # Extract features from ViT backbone
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.backbone(x)

        # timm models with num_classes=0 return features directly [batch_size, embed_dim]
        # No need to extract from dict

        # Regression head
        output = self.regressor(features)

        return output

    def unfreeze_backbone(self, num_layers=None):
        """
        Unfreezes the backbone for fine-tuning.

        Args:
            num_layers (int): Number of layers to unfreeze from the end.
                             If None, unfreezes all layers.
        """
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Unfroze entire backbone")
        else:
            # Unfreeze last N layers
            # TODO: Implement layer-wise unfreezing
            print(f"Layer-wise unfreezing not implemented yet")

        self.freeze_backbone = False


def create_dino_model(config=None):
    """
    Factory function to create DINO model with configuration.

    Args:
        config (dict): Model configuration parameters

    Returns:
        DINORegressor: Initialized model
    """
    if config is None:
        config = {
            "model_name": "dinov3_vitb14",
            "pretrained": True,
            "freeze_backbone": False,
            "dropout": 0.3,
            "hidden_dim": 512,
            "use_lora": False,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1
        }

    model = DINORegressor(
        model_name=config.get("model_name", "dinov3_vitb14"),
        pretrained=config.get("pretrained", True),
        freeze_backbone=config.get("freeze_backbone", False),
        dropout=config.get("dropout", 0.3),
        hidden_dim=config.get("hidden_dim", 512),
        use_lora=config.get("use_lora", False),
        lora_rank=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1)
    )

    return model


def get_model_size(model):
    """
    Computes model size and parameter counts.

    Args:
        model: PyTorch model

    Returns:
        dict: Model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "size_mb": total_params * 4 / (1024 ** 2)  # Assuming float32
    }

    return stats


if __name__ == "__main__":
    # Test model creation
    print("Testing DINO model with different fine-tuning strategies...")
    print("=" * 70)

    # 1. Linear probing (frozen backbone)
    print("\n1. Linear Probing (Frozen Backbone):")
    print("-" * 70)
    model_frozen = create_dino_model({
        "model_name": "dinov3_vitb14",
        "freeze_backbone": True,
        "use_lora": False
    })

    stats = get_model_size(model_frozen)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Trainable parameters: {stats['trainable_params']:,}")
    print(f"Trainable %: {100*stats['trainable_params']/stats['total_params']:.3f}%")
    print(f"Model size: {stats['size_mb']:.2f} MB")

    # 2. LoRA fine-tuning (parameter-efficient)
    print("\n2. LoRA Fine-Tuning (Rank 16):")
    print("-" * 70)
    model_lora = create_dino_model({
        "model_name": "dinov3_vitb14",
        "use_lora": True,
        "lora_rank": 16,
        "lora_alpha": 32
    })

    stats = get_model_size(model_lora)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Trainable parameters: {stats['trainable_params']:,}")
    print(f"Trainable %: {100*stats['trainable_params']/stats['total_params']:.3f}%")
    print(f"Model size: {stats['size_mb']:.2f} MB")

    # 3. Full fine-tuning
    print("\n3. Full Fine-Tuning (All Parameters):")
    print("-" * 70)
    model_full = create_dino_model({
        "model_name": "dinov3_vitb14",
        "freeze_backbone": False,
        "use_lora": False
    })

    stats = get_model_size(model_full)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Trainable parameters: {stats['trainable_params']:,}")
    print(f"Trainable %: {100*stats['trainable_params']/stats['total_params']:.3f}%")
    print(f"Model size: {stats['size_mb']:.2f} MB")

    # Test forward pass
    print("\n4. Testing Forward Pass:")
    print("-" * 70)
    batch = torch.randn(2, 3, 224, 224)
    model_lora.eval()

    with torch.no_grad():
        output = model_lora(batch)

    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output.squeeze()}")

    print("\n" + "=" * 70)
    print("DINO model module ready!")
    print("LoRA is recommended for best parameter efficiency and performance.")
    print("=" * 70)
