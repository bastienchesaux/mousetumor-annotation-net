from monai.networks.nets import BasicUNetPlusPlus, BasicUNet, UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution


def unetpp_default():
    return BasicUNetPlusPlus(
        spatial_dims=3, in_channels=1, out_channels=1, deep_supervision=False
    )


def unetpp_dropout():
    return BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        deep_supervision=False,
        dropout=0.1,
    )


def unetpp_half():
    return BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(16, 16, 32, 64, 128, 16),
        dropout=0.1,
        deep_supervision=True,
    )


def unet_default():
    return BasicUNet(spatial_dims=3, in_channels=1, out_channels=1)


def unet_half():
    return BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(16, 16, 32, 64, 128, 16),
        dropout=0.1,
    )


def unet_half_no_norm():
    return BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(16, 16, 32, 64, 128, 16),
        dropout=0.1,
        norm=None,
    )


# attempts to make shallower UNet with monai UNet implementation, gave bad results
def unet_S5D2W16_monai():  # similar form as basic_unet_half
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 16, 32, 64, 128, 128),
        strides=(1, 2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
        act=("leakyrelu", {"negative_slope": 0.1}),
    )


# def unet_S5D2W16_bis():  # similar form as basic_unet_half
#     return UNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=1,
#         channels=(16, 16, 32, 64, 128),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#         dropout=0.1,
#         act=("leakyrelu", {"negative_slope": 0.1}),
#     )


def unet_S4D2W16():  # similar form as basic_unet_half
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 16, 32, 64, 64),
        strides=(1, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
        act=("leakyrelu", {"negative_slope": 0.1}),
    )


class ConvBlock(nn.Module):
    """N convolutions at the same resolution."""

    def __init__(
        self, spatial_dims, in_channels, out_channels, n_convs, act, norm, dropout
    ):
        super().__init__()
        for i in range(n_convs):
            self.add_module(
                f"conv_{i}",
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                ),
            )

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class Down(nn.Module):
    """MaxPool followed by a ConvBlock."""

    def __init__(
        self, spatial_dims, in_channels, out_channels, n_convs, act, norm, dropout
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(2) if spatial_dims == 3 else nn.MaxPool2d(2)
        self.convs = ConvBlock(
            spatial_dims, in_channels, out_channels, n_convs, act, norm, dropout
        )

    def forward(self, x):
        return self.convs(self.pool(x))


class UpCat(nn.Module):
    """Transposed conv upsample, concat with skip, then ConvBlock."""

    def __init__(
        self, spatial_dims, in_channels, out_channels, n_convs, act, norm, dropout
    ):
        super().__init__()
        self.upsample = (
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            if spatial_dims == 3
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )
        self.convs = ConvBlock(
            spatial_dims, out_channels * 2, out_channels, n_convs, act, norm, dropout
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.convs(x)


class FlexibleUNet(nn.Module):
    """
    Same architecture as MONAI's BasicUNet with added:
    - Configurable number of resolution stages
    - Configurable number of convolutions per stage
    - Optional deep supervision (auxiliary heads at each decoder stage)

    Applied to this project -> spatial dims = 3, in channels = 1, out channels = 1

    Args:
        spatial_dims:   number of spatial dimensions (2 or 3)
        in_channels:    number of input channels
        out_channels:   number of output channels
        features:       tuple of feature channels, one per resolution stage
                        e.g. (16, 32, 64, 128) gives 4 stages
        n_convs:        number of convolutions per resolution stage
        act:            activation function spec
        norm:           normalisation spec
        dropout:        dropout probability
        deep_supervision: if True, return auxiliary outputs from each decoder stage
                          during training (all upsampled to input resolution)
    """

    def __init__(
        self,
        features: tuple = (16, 32, 64, 128),
        n_convs: int = 2,
        act: tuple = ("leakyrelu", {"inplace": True, "negative_slope": 0.1}),
        norm: tuple = ("instance", {"affine": True}),
        dropout: float = 0.1,
        deep_supervision: bool = False,
    ):
        super().__init__()

        assert len(features) >= 2, "Need at least 2 resolution stages"

        self.spatial_dims = 3
        self.in_channels = 1
        self.out_channels = 1
        self.deep_supervision = deep_supervision
        self.n_stages = len(features)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.conv_0 = ConvBlock(
            self.spatial_dims,
            self.in_channels,
            features[0],
            n_convs,
            act,
            norm,
            dropout,
        )

        for i in range(1, len(features)):
            self.add_module(
                f"down_{i}",
                Down(
                    self.spatial_dims,
                    features[i - 1],
                    features[i],
                    n_convs,
                    act,
                    norm,
                    dropout,
                ),
            )

        # ── Decoder ──────────────────────────────────────────────────────────
        for i in range(len(features) - 1, 0, -1):
            self.add_module(
                f"upcat_{i}",
                UpCat(
                    self.spatial_dims,
                    features[i],
                    features[i - 1],
                    n_convs,
                    act,
                    norm,
                    dropout,
                ),
            )

        # ── Output heads ─────────────────────────────────────────────────────
        self.final_conv = nn.Conv3d(features[0], self.out_channels, kernel_size=1)

        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                [
                    nn.Conv3d(features[i], self.out_channels, kernel_size=1)
                    for i in range(len(features) - 2, 0, -1)
                ]
            )
        else:
            self.aux_heads = nn.ModuleList(
                [nn.Identity() for i in range(len(features) - 2)]
            )

    def forward(self, x):
        input_shape = x.shape[2:]

        # ── Encoder forward ──────────────────────────────────────────────────
        skips = []
        x = self.conv_0(x)
        skips.append(x)
        for i in range(1, self.n_stages):
            x = getattr(self, f"down_{i}")(x)
            skips.append(x)

        # ── Decoder forward ──────────────────────────────────────────────────
        x = skips[-1]
        skip_connections = skips[:-1][::-1]

        aux_outputs = []
        for i in range(self.n_stages - 1):
            upcat_idx = self.n_stages - 1 - i
            x = getattr(self, f"upcat_{upcat_idx}")(x, skip_connections[i])

            if self.deep_supervision and self.training and i < len(self.aux_heads):
                aux_out = self.aux_heads[i](x)
                aux_out = F.interpolate(
                    aux_out, size=input_shape, mode="trilinear", align_corners=False
                )
                aux_outputs.append(aux_out)

        # ── Final output ─────────────────────────────────────────────────────
        final = self.final_conv(x)

        if self.deep_supervision and self.training:
            return aux_outputs + [final]

        return [final]


def unet_S5D2W16(deep_supervision=False):
    return FlexibleUNet(
        features=(16, 16, 32, 64, 128),
        n_convs=2,
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        dropout=0.1,
        deep_supervision=deep_supervision,
    )


def unet_S4D3W16(deep_supervision=False):
    return FlexibleUNet(
        features=(16, 16, 32, 64),
        n_convs=3,
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        dropout=0.1,
        deep_supervision=deep_supervision,
    )


def unet_S5D1W16(deep_supervision=False):
    return FlexibleUNet(
        features=(16, 16, 32, 64, 128),
        n_convs=1,
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        dropout=0.1,
        deep_supervision=deep_supervision,
    )


def unet_S6D2W16(deep_supervision=False):
    return FlexibleUNet(
        features=(16, 16, 32, 64, 128, 256),
        n_convs=2,
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        dropout=0.1,
        deep_supervision=deep_supervision,
    )
