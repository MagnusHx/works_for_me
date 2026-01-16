from torch import nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    """
    VGG16-style CNN til audio spectrograms.
    Input:  (B, C=1, F, T)
    Output: (B, num_classes)
    """

    def __init__(self, cfg):
        super().__init__()

        # Læser hyperparametre fra config.yaml
        in_channels = int(cfg.model.in_channels)
        out_channels = int(cfg.model.out_channels)
        num_classes = int(cfg.model.num_classes)  # Antal klasser (forskellige emotions)
        dropout = float(cfg.model.dropout)
        kernel_size = int(cfg.model.kernel_size)
        stride = int(cfg.model.stride)
        padding = int(cfg.model.padding)
        max_pool_kernel = int(cfg.model.max_pool_kernel)
        max_pool_stride = int(cfg.model.max_pool_stride)
        channels = list(cfg.model.channels)

        # Sikkerhedscheck
        assert 0.0 <= dropout <= 1.0

        # ------------ Block 1 ------------
        # (C=1 -> 64), bevarer F,T
        self.conv1_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.pool1 = nn.MaxPool2d(max_pool_kernel, max_pool_stride)  # halverer F,T

        # ------------ Block 2 ------------
        # (64 -> 128)
        self.conv2_1 = nn.Conv2d(
            in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=channels[1], out_channels=channels[1], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.pool2 = nn.MaxPool2d(max_pool_kernel, max_pool_stride)

        # ------------ Block 3 ------------
        # (128 -> 256)
        self.conv3_1 = nn.Conv2d(
            in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv3_2 = nn.Conv2d(
            in_channels=channels[2], out_channels=channels[2], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv3_3 = nn.Conv2d(
            in_channels=channels[2], out_channels=channels[2], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.pool3 = nn.MaxPool2d(max_pool_kernel, max_pool_stride)

        # ------------ Block 4 ------------
        # (256 -> 512)
        self.conv4_1 = nn.Conv2d(
            in_channels=channels[2], out_channels=channels[3], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv4_2 = nn.Conv2d(
            in_channels=channels[3], out_channels=channels[3], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv4_3 = nn.Conv2d(
            in_channels=channels[3], out_channels=channels[3], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.pool4 = nn.MaxPool2d(max_pool_kernel, max_pool_stride)

        # ------------ Block 5 ------------
        # (512 -> 512)
        self.conv5_1 = nn.Conv2d(
            in_channels=channels[3], out_channels=channels[4], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv5_2 = nn.Conv2d(
            in_channels=channels[4], out_channels=channels[4], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.conv5_3 = nn.Conv2d(
            in_channels=channels[4], out_channels=channels[4], kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.pool5 = nn.MaxPool2d(max_pool_kernel, max_pool_stride)

        # ----------- Adapter -----------
        # Gør output fast (7x7), uanset input-dimensioner
        # Det originale VGG 16 brugte 224 x 224 x 3 input dim. Så for at kører den samme arkitektur
        # skal vi sørge for at vores output matcher, så vi ikke får mismatch error
        self.adapt = nn.AdaptiveAvgPool2d((7, 7))

        # ------------ Classifier VGG16 stil ------------
        self.fc1 = nn.Linear(channels[4] * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ------------ Block 1 ------------
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        # ------------ Block 2 ------------
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        # ------------ Block 3 ------------
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        # ------------ Block 4 ------------
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        # ------------ Block 5 ------------
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)

        # Fast spatial størrelse før FC
        x = self.adapt(x)

        # Flatten: (B, 512, 7, 7) -> (B, 512*7*7)
        x = x.view(x.size(0), -1)

        # ------------ Classifier ------------
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)  # logits (ingen softmax her)

        return x


# Dummy model kan bruges som sanity check - Den erstatter config.yaml og indsætter hardcoded værdier
# if __name__ == "__main__":
#     class DummyCfg:
#         class model:
#             in_channels = 1
#             out_channels = 64
#             num_classes = 7
#             dropout = 0.3
#             kernel_size = 3
#             stride = 1
#             padding = 1
#             max_pool_kernel = 2
#             max_pool_stride = 2
#             channels = [64, 128, 256, 512, 512]

#     model = Model(DummyCfg())
#     x = torch.randn(2, 1, 128, 256)  # (B, C, F, T)
#     y = model(x)

#     print("Output shape:", y.shape)  # Burde gerne være (2, 7)
