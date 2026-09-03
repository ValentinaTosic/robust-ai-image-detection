"""Reusable spatial explanation utilities for image classifiers."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def gradcam(model: nn.Module, layer: nn.Module, inputs: torch.Tensor, predicted_class: str) -> tuple[np.ndarray, float]:
    """Compute Grad CAM for the model's predicted binary class."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(inputs)[0]
        probability = torch.sigmoid(logit).item()
        (logit if predicted_class == "ai" else -logit).backward()
        weights = captured["gradient"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * captured["activation"]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.detach().cpu().numpy(), probability
    finally:
        handle.remove()


def attention_rollout(model: nn.Module, inputs: torch.Tensor) -> tuple[np.ndarray, float]:
    """Compute mean head attention rollout from CLS to image patches."""
    attentions: list[torch.Tensor] = []
    handles = []

    def make_hook(attention_module):
        def hook(_module, _inputs, output):
            batch, tokens, combined = output.shape
            heads = attention_module.num_heads
            dim = combined // (3 * heads)
            qkv = output.reshape(batch, tokens, 3, heads, dim).permute(2, 0, 3, 1, 4)
            query = attention_module.q_norm(qkv[0])
            key = attention_module.k_norm(qkv[1])
            attentions.append(((query * attention_module.scale) @ key.transpose(-2, -1)).softmax(dim=-1).detach())
        return hook

    for block in model.blocks:
        handles.append(block.attn.qkv.register_forward_hook(make_hook(block.attn)))
    try:
        with torch.inference_mode():
            probability = torch.sigmoid(model(inputs)[0]).item()
    finally:
        for handle in handles:
            handle.remove()
    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens, device=attentions[0].device).unsqueeze(0)
    rollout = identity
    for attention in attentions:
        fused = attention.mean(dim=1) + identity
        rollout = (fused / fused.sum(dim=-1, keepdim=True)) @ rollout
    patches = rollout[0, 0, getattr(model, "num_prefix_tokens", 1):]
    height, width = model.patch_embed.grid_size
    result = patches.reshape(height, width)
    result -= result.min()
    if result.max() > 0:
        result /= result.max()
    return result.cpu().numpy(), probability


def resize_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a normalized explanation map to an image height and width."""
    return F.interpolate(torch.from_numpy(values)[None, None], size=size, mode="bilinear", align_corners=False)[0, 0].numpy()
