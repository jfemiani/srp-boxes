"""
Build the network architecture



>>> arch = srp.model.arch.Architecture(
...              rgb_shape=(3, 64, 64),
...              lidar_shape=(6, 64, 64),
...              fusion='early',
...              obb_parametrization='vector_and_width',
...              synthetic='no_pretrain',
...              channel_dropout='cdrop',
...              class_loss = 'xent_loss',
...              regression_loss = 'smooth_L1',
... )

>>> arch.fusion
'early'
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg

# pylint:disable=too-few-public-methods


class FusionOptions:
    """FusionOptions"""
    EARLY = 'early'
    LATE_ADD = 'late_add'
    LATE_CAT = 'late_cat'


class ObbOptions:
    """ObbOptions"""
    VECTOR_AND_WIDTH = 'vector_and_width'
    TWO_VECTORS = 'two_vectors'
    FOUR_POINTS = 'four_points'


class SyntheticOptions:
    """SyntheticOptions"""
    PRETRAIN = 'pretrain'
    NO_PRETRAIN = 'no_pretrain'


class ChannelDropoutOptions:
    """ChannelDropoutOptions"""
    CDROP = 'cdrop'
    NO_CDROP = 'no_cdrop'


class ClassLossOptions:
    """ClassLossOptions"""
    HING_LOSS = 'hing_loss'
    XENT_LOSS = 'xent_loss'


class RegressionLossOptions:
    """RegressionLossOptions"""
    SMOOTH_L1 = 'smooth_L1'
    L2 = 'L2'


def _change_num_channels(layer, num_in):
    """Changes the number of in_channels for a coinvolution in-place.

    Since the semanics of each input presumably change, this sets them all to their mean
    """
    layer.in_channels = num_in
    weights = layer.weight
    weights = weights.mean(1)
    weights = weights[:, None, :, :]
    weights = weights.expand(-1, num_in, -1, -1)
    weights = weights.contiguous()
    layer.weight = nn.Parameter(weights)


class Architecture(nn.Module):
    #  pylint:disable=too-many-instance-attributes
    """Architecture"""

    def __init__(self, **kwargs):
        """
        The network architecture.


        :param rgb_shape:
        :param lidar_shape:
        :param fusion:


        :param channel_dropout:

        :param class_loss:

        :param obb_parametrization:
        :param regression_loss:


        """
        super().__init__()

        self.fusion = kwargs.pop('fusion', FusionOptions.EARLY)
        self.obb_parametrization = kwargs.pop('obb_parametrization',
                                              ObbOptions.VECTOR_AND_WIDTH)
        self.channel_dropout = kwargs.pop('channel_dropout',
                                          ChannelDropoutOptions.CDROP)
        self.synthetic = kwargs.pop('synthetic', SyntheticOptions.NO_PRETRAIN)
        self.class_loss = kwargs.pop('class_loss', ClassLossOptions.XENT_LOSS)
        self.regression_loss = kwargs.pop('regression_loss',
                                          RegressionLossOptions.SMOOTH_L1)

        self.rgb_shape = kwargs.pop('rgb_shape', (3, 64, 64))
        self.lidar_shape = kwargs.pop('lidar_shape', (6, 64, 64))

        assert self.lidar_shape[1:] == self.rgb_shape[
            1:], "Lidar and RGB must have the same number of rows and columns"

        base = vgg.vgg13_bn(pretrained=True)

        # Initialize some of the optional layers to None

        self.rgb_features = None
        self.lidar_features = None
        self.combined_features = None
        self.fusion_layer = None

        # Create the layers of this net based on settings.

        if self.fusion == FusionOptions.EARLY:
            self.combined_features = deepcopy(base.features)
            _change_num_channels(self.combined_features[0],
                                 self.rgb_shape[0] + self.lidar_shape[0])
        else:
            self.rgb_features = deepcopy(base.features)
            self.lidar_features = deepcopy(base.features)
            _change_num_channels(self.lidar_features[0], self.lidar_shape[0])

            if self.rgb_shape[0] != 3:
                _change_num_channels(self.rgb_features[0], self.rgb_shape[0])

        if self.fusion == FusionOptions.LATE_CAT:
            # Get the number of output features by processing a dummy (random) input.
            dummy = torch.rand((1, ) + self.rgb_shape)
            out_shape = self.rgb_features.forward(dummy).shape
            num_features = out_shape[1]

            self.fusion_layer = nn.Linear(
                2 * num_features, num_features, bias=True)

        # TODO: Finish bulding the module

    def forward(self, x):
        # pylint:disable=arguments-differ
        """Forward pass through then net

        :param x: batch sized input; shape=(bs, num_rgb_channels +
                  num_lidar_channels, height, width)

        """
        batch_size = x.shape[0]
        # num_channels = x.shape[1]
        # num_rows = x.shape[2]
        # num_cols = x.shape[3]

        num_rgb_channels = self.rgb_shape[0]
        # num_lidar_channels = self.lidar_shape[0]

        rgb = x[:, :num_rgb_channels]
        lidar = x[:, num_rgb_channels:]

        if self.fusion == FusionOptions.EARLY:
            fused = self.combined_features.forward(x)
            fused = F.adaptive_avg_pool2d(fused,
                                          (batch_size, fused.shape[1], 1, 1))
        else:
            # Late fusion -- process each source separately

            rgb_f = self.rgb_features.forward(rgb)
            lidar_f = self.lidar_features.forward(lidar)

            num_features = rgb_f.shape[1]
            assert num_features == lidar_f.shape[
                1], "Lidar and RGB should produce the same no. of features"

            # Make sure the outputs are
            rgb_f = F.adaptive_max_pool2d(rgb_f,
                                          (batch_size, num_features, 1, 1))
            lidar_f = F.adaptive_max_pool2d(lidar_f,
                                            (batch_size, num_features, 1, 1))

            if self.fusion == FusionOptions.LATE_ADD:
                fused = rgb_f + lidar_f
            elif self.fusion == FusionOptions.LATE_CAT:
                cat = torch.cat([rgb_f, lidar_f], dim=1)
                fused = self.fusion_layer.forward(cat)

        # Features have been processed
