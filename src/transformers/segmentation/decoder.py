from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch import Tensor

from torchsummary import summary


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.n_cls = n_cls
        self.conv = nn.Conv1d(d_encoder, d_encoder, kernel_size=(13,), stride=(1,), padding=6)
        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x):
        x = F.relu(self.conv(x.transpose(-1, -2)))
        x = self.head(x.transpose(-1, -2))
        # x = self.head(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        x = x.transpose(-1, -2)
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        norm = norm.transpose(-1, -2)
        return norm


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 3, 1, 1),
            # LayerNorm?
            Norm(output_channels),
            nn.GELU(),
            nn.Conv1d(output_channels, output_channels, 3, 1, 1),
            # LayerNorm?
            Norm(output_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class DownSampling(nn.Module):
    """
    1D-UNET DownSampling block.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
    ):
        super().__init__()
        self.conv_layers = DoubleConv(input_channels, output_channels)

        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.activation_fn = nn.GELU()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.avg_pool(x)
        x = self.conv_layers(x)
        return x


class UpSampling(nn.Module):
    """
    1D-UNET UpSampling block.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
    ):
        super().__init__()
        self.up = nn.ConvTranspose1d(input_channels, output_channels, 2, 2)
        self.conv_layers = DoubleConv(input_channels, output_channels)

    def forward(self, x: Tensor, x2: Tensor) -> Tensor:
        x = self.up(x)
        diff = x.size()[2] - x2.size()[2]
        x2 = F.pad(x2, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x], dim=1)
        return self.conv_layers(x)


class FinalConv(nn.Module):
    """
    Final output block of the 1D-UNET.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
    ):
        super().__init__()

        self.conv_layers = nn.Conv1d(input_channels, output_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        return x


class StartConv(nn.Module):
    """
    Final output block of the 1D-UNET.
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
    ):
        super().__init__()

        self.conv_layers = DoubleConv(input_channels, output_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        return x


class UNETHead(nn.Module):
    """
    1D-UNET based head to be plugged on top of a pretrained model to perform
    semantic segmentation.
    """

    def __init__(
            self,
            num_classes: int,
            input_channels=768
    ):
        super().__init__()
        self.input_conv = StartConv(input_channels, input_channels)

        self.down1 = DownSampling(input_channels, input_channels * 2)
        self.down2 = DownSampling(input_channels * 2, input_channels * 4)
        self.down3 = DownSampling(input_channels * 4, input_channels * 8)

        self.up1 = UpSampling(input_channels * 8, input_channels * 4)
        self.up2 = UpSampling(input_channels * 4, input_channels * 2)
        self.up3 = UpSampling(input_channels * 2, input_channels)

        self.output_conv = FinalConv(input_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        inputs = self.input_conv(x)

        d1 = self.down1(inputs)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        u1 = self.up1(d3, d2)
        u2 = self.up2(u1, d1)
        u3 = self.up3(u2, inputs)

        x = self.output_conv(u3)
        return x


class UNet(nn.Module):
    """
    Returns a probability between 0 and 1 over a target feature presence
    for each nucleotide in the input sequence. Assumes the sequence has been tokenized
    with non-overlapping 6-mers.
    """

    def __init__(
            self,
            num_features: int,
            embed_dimension: int = 768,
    ):
        super().__init__()
        self._num_features = num_features
        self.unet = UNETHead(
            num_classes=num_features,
            input_channels=embed_dimension
        )

    def forward(
            self, x: Tensor
    ) -> Dict[str, Tensor]:
        """
        Input shape: (batch_size, sequence_length + 1, embed_dim)
        Output_shape: (batch_size, 6 * sequence_length, 2)
        """
        logits = self.unet(x)  # remove CLS token
        return {"logits": logits}


unet = UNet(num_features=2)
summary(unet, (768, 135), device='cpu')
