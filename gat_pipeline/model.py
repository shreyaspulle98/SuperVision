"""
gat_pipeline/model.py

Pre-trained ALIGNN model for superconductor Tc prediction via transfer learning.

This module implements:
- Loading pre-trained ALIGNN from Materials Project formation energy task
- Replacing final prediction head for Tc regression
- Different learning rates for backbone (1e-5) vs head (1e-3)
- Parameter-efficient fine-tuning on superconductor data

ALIGNN (Atomistic Line Graph Neural Network):
- Pre-trained on 100k+ materials from Materials Project
- Learns both atomic and bond representations
- Uses line graphs to capture angular information
- State-of-the-art performance on formation energy prediction

Transfer Learning Strategy:
1. Load ALIGNN pre-trained on formation energy (mp_e_form)
2. Freeze or slow down backbone updates (lr=1e-5)
3. Replace final FC layer for Tc prediction
4. Fine-tune head faster (lr=1e-3) on superconductor data
"""

import torch
import torch.nn as nn
from pathlib import Path


class PretrainedALIGNN(nn.Module):
    """
    ALIGNN model pre-trained on Materials Project, fine-tuned for Tc prediction.

    Uses transfer learning from formation energy prediction task to
    superconductor critical temperature prediction.
    """

    def __init__(
        self,
        pretrained_model_name="mp_e_form",
        freeze_backbone=False,
        hidden_dim=256
    ):
        """
        Initialize pre-trained ALIGNN model.

        Args:
            pretrained_model_name (str): Name of pre-trained model from ALIGNN
                Available models:
                - "mp_e_form": Formation energy (recommended for materials)
                - "mp_gappbe": Band gap
                - "qm9_U0": QM9 energy
            freeze_backbone (bool): If True, freeze all layers except final FC
            hidden_dim (int): Hidden dimension for new prediction head
        """
        super(PretrainedALIGNN, self).__init__()

        self.pretrained_model_name = pretrained_model_name
        self.freeze_backbone = freeze_backbone

        # Load pre-trained ALIGNN model
        print(f"Loading pre-trained ALIGNN model: {pretrained_model_name}")
        try:
            from alignn.pretrained import get_figshare_model
            from alignn.models.alignn import ALIGNN
            import json

            # Download and load pre-trained model (returns just model, not tuple)
            self.backbone = get_figshare_model(pretrained_model_name)
            print(f"✓ Loaded pre-trained ALIGNN from figshare")

            # Extract embedding dimension from model config
            # ALIGNN uses hidden_features as the embedding dimension before final layer
            if hasattr(self.backbone, 'config'):
                self.embedding_dim = self.backbone.config.hidden_features
            else:
                self.embedding_dim = 256  # Default ALIGNN hidden features

            print(f"✓ Using embedding dimension: {self.embedding_dim}")

        except Exception as e:
            print(f"Error loading ALIGNN: {e}")
            print("\nTo install ALIGNN, run:")
            print("  pip install alignn")
            print("\nFalling back to random initialization (not recommended!)")

            # Fallback: create random ALIGNN (not pre-trained)
            from alignn.models.alignn import ALIGNN, ALIGNNConfig

            self.alignn_config = ALIGNNConfig(
                name="alignn",
                alignn_layers=4,
                gcn_layers=4,
                atom_input_features=92,
                edge_input_features=80,
                hidden_features=256,
                output_features=1
            )
            self.backbone = ALIGNN(self.alignn_config)
            self.embedding_dim = 1024  # 4 layers * 256 features
            print("⚠ Using randomly initialized ALIGNN (NOT pre-trained)")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✓ Froze all backbone parameters (linear probe mode)")

        # Replace final prediction layer for Tc prediction
        # ALIGNN's original fc layer predicts formation energy (scalar)
        # We replace it with our own Tc prediction head
        # The backbone will output pooled features (h) before fc layer

        # Add new Tc prediction head (replaces the backbone's fc layer)
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Replace backbone's fc layer with our new head
        # This way we can use backbone.forward() directly and it will use our fc
        self.backbone.fc = self.fc

        print(f"✓ Created new Tc prediction head ({self.embedding_dim} -> {hidden_dim} -> 1)")

        # Print parameter statistics
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.fc.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = backbone_params + head_params

        print(f"\nModel Statistics:")
        print(f"  Backbone parameters: {backbone_params:,}")
        print(f"  Head parameters: {head_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    def forward(self, g, lg=None):
        """
        Forward pass through ALIGNN backbone and Tc prediction head.

        Args:
            g: DGL graph (atom graph)
            lg: DGL line graph (bond graph) - optional, will be computed if not provided

        Returns:
            torch.Tensor: Predicted Tc values [batch_size, 1]
        """
        # ALIGNN expects either:
        # - A tuple (g, lg, lattice) for full ALIGNN
        # - Just g for GCN-only mode
        # Since we have both g and lg, pass them as expected by ALIGNN

        if lg is not None:
            # ALIGNN forward expects (g, lg, lattice) tuple
            # The lattice is None for our use case
            input_graphs = (g, lg, None)
        else:
            input_graphs = g

        # Forward through backbone (which now has our custom fc head)
        # This will run through all ALIGNN layers, pool with readout, then apply fc
        tc_pred = self.backbone(input_graphs)

        return tc_pred

    def get_optimizer_params(self, backbone_lr=1e-5, head_lr=1e-3):
        """
        Get parameter groups with different learning rates for fine-tuning.

        This enables transfer learning where the pre-trained backbone is
        updated slowly while the new Tc prediction head learns faster.

        Args:
            backbone_lr (float): Learning rate for pre-trained backbone (default: 1e-5)
            head_lr (float): Learning rate for Tc prediction head (default: 1e-3)

        Returns:
            list: List of parameter dicts for optimizer
                [
                    {'params': backbone_params, 'lr': 1e-5},
                    {'params': head_params, 'lr': 1e-3}
                ]
        """
        # Separate backbone and head parameters
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if 'fc' in name or 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []

        if len(backbone_params) > 0:
            param_groups.append({
                'params': backbone_params,
                'lr': backbone_lr,
                'name': 'backbone'
            })
            print(f"  Backbone: {sum(p.numel() for p in backbone_params):,} params @ lr={backbone_lr}")

        if len(head_params) > 0:
            param_groups.append({
                'params': head_params,
                'lr': head_lr,
                'name': 'head'
            })
            print(f"  Head: {sum(p.numel() for p in head_params):,} params @ lr={head_lr}")

        return param_groups

    def unfreeze_backbone(self, num_layers=None):
        """
        Unfreeze the backbone for fine-tuning.

        Args:
            num_layers (int): Number of layers to unfreeze from the end.
                             If None, unfreezes all layers.
        """
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Unfroze entire ALIGNN backbone")
        else:
            # TODO: Implement layer-wise unfreezing for ALIGNN
            print(f"Layer-wise unfreezing not yet implemented for ALIGNN")

        self.freeze_backbone = False


def create_alignn_model(config=None):
    """
    Factory function to create pre-trained ALIGNN model.

    Args:
        config (dict): Model configuration parameters

    Returns:
        PretrainedALIGNN: Initialized model ready for fine-tuning
    """
    if config is None:
        config = {
            "pretrained_model_name": "mp_e_form",
            "freeze_backbone": False,
            "hidden_dim": 256
        }

    model = PretrainedALIGNN(
        pretrained_model_name=config.get("pretrained_model_name", "mp_e_form"),
        freeze_backbone=config.get("freeze_backbone", False),
        hidden_dim=config.get("hidden_dim", 256)
    )

    return model


if __name__ == "__main__":
    # Test model creation
    print("=" * 70)
    print("Testing Pre-trained ALIGNN Model")
    print("=" * 70)

    print("\n1. Creating model with pre-trained weights...")
    print("-" * 70)

    try:
        model = create_alignn_model({
            "pretrained_model_name": "mp_e_form",
            "freeze_backbone": False,
            "hidden_dim": 256
        })

        print("\n2. Testing parameter groups for fine-tuning...")
        print("-" * 70)
        param_groups = model.get_optimizer_params(backbone_lr=1e-5, head_lr=1e-3)

        print(f"\n✓ Model ready for fine-tuning on superconductor Tc prediction!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure ALIGNN is installed:")
        print("  pip install alignn")
        print("=" * 70)
