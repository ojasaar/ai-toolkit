from typing import TYPE_CHECKING, Union
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from safetensors.torch import load_file
import os

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.models.base_model import BaseModel


def load_assistant_lora_from_path(adapter_path: str, sd: Union['StableDiffusion', 'BaseModel']) -> LoRASpecialNetwork:
    """
    Load an assistant LoRA adapter from a path.
    Supports Flux and Qwen models.

    Args:
        adapter_path: Path to the safetensors file (local path or HuggingFace URL)
        sd: The model instance (StableDiffusion or BaseModel subclass)
    """
    print(f"Loading assistant adapter from {adapter_path}")

    # Handle HuggingFace URLs by downloading first
    if adapter_path.startswith("http://") or adapter_path.startswith("https://"):
        from huggingface_hub import hf_hub_download
        # Parse HuggingFace URL: https://huggingface.co/repo/resolve/main/path/file.safetensors
        if "huggingface.co" in adapter_path and "/resolve/" in adapter_path:
            parts = adapter_path.split("huggingface.co/")[1]
            repo_and_rest = parts.split("/resolve/")
            repo_id = repo_and_rest[0]
            # Remove branch (main, etc) and get filename
            rest = repo_and_rest[1].split("/", 1)[1]  # Skip 'main'
            filename = rest
            print(f"Downloading from HuggingFace: repo={repo_id}, file={filename}")
            adapter_path = hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            raise ValueError(f"Unsupported URL format: {adapter_path}")

    lora_state_dict = load_file(adapter_path)

    # Detect model type from the sd instance
    is_flux = getattr(sd, 'is_flux', False)
    arch = getattr(sd, 'arch', '')
    is_qwen = arch.startswith('qwen_image')

    if is_flux:
        return _load_flux_assistant_lora(sd, lora_state_dict)
    elif is_qwen:
        return _load_qwen_assistant_lora(sd, lora_state_dict)
    else:
        raise ValueError(f"Unsupported model architecture for assistant adapters: {arch}")


def _load_flux_assistant_lora(sd, lora_state_dict) -> LoRASpecialNetwork:
    """Load assistant LoRA for Flux models."""
    pipe = sd.pipeline

    linear_dim = int(lora_state_dict['transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight'].shape[0])
    linear_alpha = linear_dim
    transformer_only = 'transformer.proj_out.alpha' not in lora_state_dict

    network_config = NetworkConfig(
        linear=linear_dim,
        linear_alpha=linear_alpha,
        transformer_only=transformer_only,
    )

    network = LoRASpecialNetwork(
        text_encoder=pipe.text_encoder,
        unet=pipe.transformer,
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        is_flux=True,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_assistant_adapter=True
    )
    network.apply_to(
        pipe.text_encoder,
        pipe.transformer,
        apply_text_encoder=False,
        apply_unet=True
    )
    network.force_to(sd.device_torch, dtype=sd.torch_dtype)
    network.eval()
    network._update_torch_multiplier()
    network.load_weights(lora_state_dict)
    network.is_active = True

    return network


def _load_qwen_assistant_lora(sd, lora_state_dict) -> LoRASpecialNetwork:
    """Load assistant LoRA for Qwen Image models."""
    # Find the linear dimension and alpha from the LoRA weights
    linear_dim = None
    linear_alpha = None

    for key in lora_state_dict.keys():
        # Get dimension from lora_down weight shape
        if linear_dim is None and ('lora_A.weight' in key or 'lora_down.weight' in key):
            linear_dim = int(lora_state_dict[key].shape[0])
        # Get alpha from .alpha keys (critical for correct scaling!)
        if linear_alpha is None and key.endswith('.alpha'):
            linear_alpha = float(lora_state_dict[key].item())
        # Break early if we have both
        if linear_dim is not None and linear_alpha is not None:
            break

    if linear_dim is None:
        raise ValueError("Could not determine LoRA dimension from state dict")

    # Default alpha to dim if not found (standard behavior)
    if linear_alpha is None:
        linear_alpha = linear_dim

    # Calculate the correct multiplier to compensate for PEFT format forcing alpha=dim
    # PEFT format uses scale=1.0, but Lightning LoRA expects scale=alpha/dim
    lora_multiplier = linear_alpha / linear_dim
    print(f"Lightning LoRA: dim={linear_dim}, alpha={linear_alpha}, multiplier={lora_multiplier}")

    network_config = NetworkConfig(
        linear=linear_dim,
        linear_alpha=linear_alpha,
        transformer_only=True,  # Qwen LoRAs are transformer-only
    )

    # Qwen models use text_encoder as a list
    text_encoder = sd.text_encoder
    if isinstance(text_encoder, list):
        text_encoder = text_encoder[0] if len(text_encoder) > 0 else None

    network = LoRASpecialNetwork(
        text_encoder=text_encoder,
        unet=sd.transformer,
        lora_dim=network_config.linear,
        multiplier=1.0,  # Set to 1.0 initially, will update after modules are created
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        is_flux=False,
        is_transformer=True,  # Required for PEFT format key conversion (lora_A/lora_B → lora_down/lora_up)
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        target_lin_modules=["QwenImageTransformer2DModel"],
        is_assistant_adapter=True,
        base_model=sd,  # Required for convert_lora_weights_before_load to be called
    )
    network.apply_to(
        text_encoder,
        sd.transformer,
        apply_text_encoder=False,
        apply_unet=True
    )
    network.force_to(sd.device_torch, dtype=sd.torch_dtype)
    network.eval()

    # Set the correct multiplier AFTER modules are created
    # This compensates for PEFT format forcing alpha=dim
    network.multiplier = lora_multiplier

    network.load_weights(lora_state_dict)
    network.is_active = True

    return network
