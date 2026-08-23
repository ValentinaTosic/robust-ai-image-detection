"""A small convolutional baseline for binary real-vs-AI classification."""

from torch import Tensor, nn


class BaselineCNN(nn.Module):
    """Four-stage CNN trained from scratch for binary image classification.

    The network returns one raw logit per image. A sigmoid is intentionally not
    part of ``forward`` because training uses ``BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ) -> None:
        """Build the baseline network.

        Args:
            channels: Output channels for consecutive convolutional blocks.
            dropout: Dropout probability before the binary classifier.

        Raises:
            ValueError: If the channel configuration or dropout is invalid.
        """
        super().__init__()

        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(channels[-1], 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize convolutional, normalization, and classifier weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: Tensor) -> Tensor:
        """Return one binary-classification logit for each input image."""
        features = self.features(images)
        pooled = self.pool(features).flatten(start_dim=1)
        return self.classifier(self.dropout(pooled)).squeeze(dim=1)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated during training."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
