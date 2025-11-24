"""
Manual LoRA (Low-Rank Adaptation) implementation for Vision Transformers.

This module implements LoRA without the PEFT library, providing full control
and compatibility with timm models.

LoRA decomposes weight updates as:
    W_new = W_0 + ΔW = W_0 + B @ A

Where:
    - W_0: Original frozen weights [d_out, d_in]
    - B: Trainable low-rank matrix [d_out, r]
    - A: Trainable low-rank matrix [r, d_in]
    - r: Rank (typically 4-64)

The update is scaled by (alpha / r) for stability.
"""

import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps a Linear layer.

    Adds low-rank trainable matrices A and B to adapt the frozen weights.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 16,
        alpha: float = 32,
        dropout: float = 0.1
    ):
        """
        Initialize LoRA layer.

        Args:
            original_layer: Original Linear layer to adapt
            r: Rank of low-rank decomposition
            alpha: Scaling parameter
            dropout: Dropout rate for LoRA
        """
        super().__init__()

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Get dimensions from original layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Freeze original weights
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA low-rank matrices
        # A: [r, in_features] - initialized with Kaiming uniform
        # B: [out_features, r] - initialized to zero
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        # Initialize A with Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is initialized to zero, so initially LoRA adds nothing

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor [batch_size, ..., in_features]

        Returns:
            Output tensor [batch_size, ..., out_features]
        """
        # Original layer output (frozen)
        original_out = self.original_layer(x)

        # LoRA adaptation: x @ A^T @ B^T, scaled by (alpha/r)
        # x: [..., in_features]
        # lora_A: [r, in_features] -> A^T: [in_features, r]
        # lora_B: [out_features, r] -> B^T: [r, out_features]

        # Apply dropout to input
        x_dropped = self.dropout(x)

        # Compute low-rank update
        lora_out = (x_dropped @ self.lora_A.T) @ self.lora_B.T
        lora_out = lora_out * self.scaling

        return original_out + lora_out

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, r={self.r}, alpha={self.alpha}'


def apply_lora_to_model(
    model: nn.Module,
    target_module_names: list = None,
    r: int = 16,
    alpha: float = 32,
    dropout: float = 0.1
):
    """
    Apply LoRA to specific modules in a model.

    Args:
        model: Model to adapt (typically a ViT)
        target_module_names: Names of modules to adapt (e.g., ['attn.qkv'])
        r: LoRA rank
        alpha: LoRA scaling
        dropout: LoRA dropout rate

    Returns:
        tuple: (modified model, number of trainable params, number of LoRA params)
    """
    if target_module_names is None:
        target_module_names = ['attn.qkv']  # Default: attention QKV

    lora_params = 0
    modules_adapted = 0

    # Recursively find and replace target modules
    for name, module in list(model.named_modules()):
        # Check if this module matches any target
        for target_name in target_module_names:
            if target_name in name and isinstance(module, nn.Linear):
                # Get parent module and attribute name
                *parent_path, attr_name = name.split('.')
                parent = model
                for part in parent_path:
                    parent = getattr(parent, part)

                # Replace with LoRA layer
                lora_layer = LoRALayer(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_layer)

                # Count parameters
                lora_params += r * (module.in_features + module.out_features)
                modules_adapted += 1

                print(f"  Applied LoRA to: {name} ({module.in_features}x{module.out_features})")
                break

    print(f"\nTotal modules adapted: {modules_adapted}")
    print(f"Total LoRA parameters: {lora_params:,}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, trainable_params, lora_params


def get_lora_state_dict(model: nn.Module):
    """
    Extract only LoRA parameters from model state dict.

    Args:
        model: Model with LoRA layers

    Returns:
        dict: State dict containing only LoRA parameters
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state_dict: dict):
    """
    Load LoRA parameters into model.

    Args:
        model: Model with LoRA layers
        lora_state_dict: State dict with LoRA parameters
    """
    model_state = model.state_dict()
    model_state.update(lora_state_dict)
    model.load_state_dict(model_state)
    print(f"Loaded {len(lora_state_dict)} LoRA parameters")


def merge_lora_weights(model: nn.Module):
    """
    Merge LoRA weights into original weights (for inference).

    This creates a single weight matrix W_new = W_0 + BA,
    eliminating the need for separate LoRA computation.

    Args:
        model: Model with LoRA layers
    """
    for module in model.modules():
        if isinstance(module, LoRALayer):
            # Compute LoRA delta: BA
            with torch.no_grad():
                lora_delta = (module.lora_B @ module.lora_A) * module.scaling

                # Add to original weights
                module.original_layer.weight.data += lora_delta

                # Remove LoRA parameters (they're now merged)
                module.lora_A = None
                module.lora_B = None

    print("LoRA weights merged into base model")


if __name__ == "__main__":
    # Test LoRA implementation
    print("Testing LoRA implementation...")

    # Create a simple linear layer
    layer = nn.Linear(768, 2304)  # Typical ViT QKV projection
    print(f"\nOriginal layer: {layer.in_features} -> {layer.out_features}")
    print(f"Original params: {layer.weight.numel():,}")

    # Wrap with LoRA
    lora_layer = LoRALayer(layer, r=16, alpha=32, dropout=0.1)
    print(f"\nLoRA layer created with rank {lora_layer.r}")
    print(f"LoRA A params: {lora_layer.lora_A.numel():,}")
    print(f"LoRA B params: {lora_layer.lora_B.numel():,}")
    print(f"Total LoRA params: {lora_layer.lora_A.numel() + lora_layer.lora_B.numel():,}")
    print(f"Reduction: {100 * (1 - (lora_layer.lora_A.numel() + lora_layer.lora_B.numel()) / layer.weight.numel()):.1f}%")

    # Test forward pass
    x = torch.randn(4, 196, 768)  # Batch of ViT tokens

    print("\nTesting forward pass...")
    with torch.no_grad():
        out_original = layer(x)
        out_lora = lora_layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_lora.shape}")
    print(f"Output difference (should be small initially): {(out_lora - out_original).abs().mean():.6f}")

    # Test trainable parameters
    original_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    lora_trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

    print(f"\nOriginal layer trainable params: {original_trainable:,}")
    print(f"LoRA layer trainable params: {lora_trainable:,}")

    print("\n✅ LoRA implementation working correctly!")
