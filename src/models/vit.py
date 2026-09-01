"""ImageNet-pretrained Vision Transformer adapted for binary classification."""

import timm
from torch import nn
from src.models.heads import BinaryHead, set_backbone_trainable


def build_vit(model_name: str, freeze_backbone: bool = False, drop_path_rate: float = 0.0) -> nn.Module:
    """Build an ImageNet-pretrained Vision Transformer with a binary head.

    The pretrained classification head is replaced with a single-logit output
    so the model can use the shared BCEWithLogitsLoss training and evaluation
    pipeline.

    Args:
        model_name: timm model identifier, e.g. ``"vit_small_patch16_224"``.
        freeze_backbone: If True, freeze all backbone parameters while keeping
            the classification head trainable. Used during the warmup phase.
        drop_path_rate: Stochastic depth rate applied inside the transformer
            blocks. A positive value provides additional regularization during
            fine-tuning.

    Returns:
        A Vision Transformer returning one logit per image with shape
        ``(batch,)``.
    """
    model: nn.Module = timm.create_model(model_name, pretrained=True, num_classes=1, drop_path_rate=drop_path_rate)
    in_features: int = model.head.in_features
    model.head = BinaryHead(in_features)

    if freeze_backbone:
        set_backbone_trainable(model, head_attr="head", trainable=False)

    return model


def unfreeze_vit_blocks(model: nn.Module, num_blocks: int, head_attr: str = "head") -> None:
    """Unfreeze the head and the last transformer blocks of a ViT.

    Earlier transformer blocks remain frozen to reduce the number of
    trainable parameters and limit overfitting on the relatively small
    training set. The final normalization layer is also unfrozen because it
    directly follows the transformer blocks before classification.

    Args:
        model: timm Vision Transformer containing a blocks attribute.
        num_blocks: Number of final transformer blocks to unfreeze. Must be
            between 1 and the total number of transformer blocks.
        head_attr: Name of the classification head attribute. Defaults to "head".

    Returns:
        None.

    Raises:
        AttributeError: If the model does not have a blocks attribute.
        ValueError: If num_blocks is outside the valid range.
    """
    if not hasattr(model, "blocks"):
        raise AttributeError("Model has no 'blocks' attribute; unfreeze_vit_blocks expects a timm ViT.")
    total_blocks: int = len(model.blocks)
    if not 1 <= num_blocks <= total_blocks:
        raise ValueError(f"num_blocks must be between 1 and {total_blocks}, got {num_blocks}.")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in getattr(model, head_attr).parameters():
        parameter.requires_grad = True
    for block in model.blocks[-num_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
            
    # Fine-tune the final normalization layer together with the last blocks.
    for norm_attr in ("norm", "fc_norm"):
        norm_module: nn.Module | None = getattr(model, norm_attr, None)
        if norm_module is not None:
            for parameter in norm_module.parameters():
                parameter.requires_grad = True
