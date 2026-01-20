import torch
from torch import nn


class Model(nn.Module):
    """VGG16 style CNN for audio spectrograms."""

    def __init__(self, cfg):
        super().__init__()

        in_channels = int(cfg.model.in_channels)
        out_channels = int(cfg.model.out_channels)
        num_classes = int(cfg.model.num_classes)
        dropout = float(cfg.model.dropout)
        kernel_size = int(cfg.model.kernel_size)
        stride = int(cfg.model.stride)
        padding = int(cfg.model.padding)
        max_pool_kernel = int(cfg.model.max_pool_kernel)
        max_pool_stride = int(cfg.model.max_pool_stride)
        channels = list(cfg.model.channels)

        if not 0.0 <= dropout <= 1.0:
            msg = f"dropout must be in [0, 1], got {dropout}"
            raise ValueError(msg)

        self.features = nn.Sequential(
            *self._make_block(in_channels, out_channels, 2, kernel_size, stride, padding, max_pool_kernel, max_pool_stride),
            *self._make_block(channels[0], channels[1], 2, kernel_size, stride, padding, max_pool_kernel, max_pool_stride),
            *self._make_block(channels[1], channels[2], 3, kernel_size, stride, padding, max_pool_kernel, max_pool_stride),
            *self._make_block(channels[2], channels[3], 3, kernel_size, stride, padding, max_pool_kernel, max_pool_stride),
            *self._make_block(channels[3], channels[4], 3, kernel_size, stride, padding, max_pool_kernel, max_pool_stride),
        )

        self.adapt = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[4] * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    @staticmethod
    def _make_block(
        in_channels: int,
        out_channels: int,
        conv_layers: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_kernel: int,
        pool_stride: int,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []
        for layer_idx in range(conv_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels if layer_idx == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(pool_kernel, pool_stride))
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adapt(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
