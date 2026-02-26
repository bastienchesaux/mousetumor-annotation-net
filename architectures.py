from monai.networks.nets import BasicUNetPlusPlus, BasicUNet


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
    )


def unet_default():
    return BasicUNet(spatial_dims=3, in_channels=1, out_channels=1)
