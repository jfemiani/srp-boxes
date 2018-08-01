"""
Build the network architecture



>>> arch = Architecture(
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

import shutil
# pylint:disable=too-few-public-methods
import sys
from copy import deepcopy

import numpy as np
import torch
import tqdm as tq
from torch import nn, optim
from torch.nn import functional as F

from torchvision.models import vgg


def clear_tqdm():
    """clear_tqdm
    Get rid of any lingering progress bar that may remain in tqdm.
    """
    inst = getattr(tq.tqdm, '_instances', None)
    if not inst:
        return
    try:
        for _ in range(len(inst)):
            inst.pop().close()
    except Exception:  # pylint:disable=broad-except
        pass


def tqdm(*args, **kwargs):
    """tqdm that prints to stdout instead of stderr

    :param *args:
    :param **kwargs:
    """
    clear_tqdm()
    return tq.tqdm(*args, file=sys.stdout, **kwargs)


def trange(*args, **kwargs):
    """trange that prints to stdout instead of stderr

    :param *args:
    :param **kwargs:
    """
    clear_tqdm()
    return tq.trange(*args, file=sys.stdout, **kwargs)


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


class EarlyFusion(nn.Module):
    """
    Early Fusion Features
    ---------------------

    This is the portion of the net that transforms images into features.

    """

    def __init__(self, lidar_channels, rgb_channels, proto):
        """Early Fusion Feature Layes

        This is the portion of the net that transforms images into features.

        :param lidar_channels: The number of channels in the lidar input
        :param rgb_channels: The number of RGB channels (i.e. 3)
        :param proto: A prototype net to base feature extraction on;
                      e.g.`torchvision.models.vgg.vgg13_bn(True).features`
        """
        super().__init__()

        self.lidar_channels = lidar_channels
        self.rgb_channels = rgb_channels
        self.net = deepcopy(proto)

        # Since the lidar data is fundamentally different than RGB, this
        # sets all weights to the mean.
        _change_num_channels(proto[0], self.rgb_channel + self.lidar_channels)

        # The first 3 channels are RGB; those weights are okay to use as is
        proto[0].weights[:self.rgb_channels] = proto[0].weights[:self.rgb_channels]

    def forward(self, x):  # pylint:disable=arguments-differ
        """forward

        :param x:
        """
        return self.net(x)


class LateFusion(nn.Module):
    """
    Late Fusion Features (abstract class)
    -------------------------------------

    The portion of the net that transforms images into features
    using late fusion.

    Late fusion refers to processing each input source seperately
    with networks trained to detect features from that source of data.

    After the inputs are processed, the output features are combined
    or _fused_ to form a single output vector.

    This class provides an abstract `fuse` method to do the data
    fusion;

    """

    def __init__(self, lidar_channels, rgb_channels, proto):
        """
        Create the portion of a late fusion net that processes the
        two input sources separately. This should only be called
        by derived classes that provide a `fuse` method to complete
        the data fusion process.

        :param lidar_channels: The number of channels in the lidar input
        :param rgb_channels: The number of RGB channels (i.e. 3)
        :param proto: A prototype net to base feature extraction on;
                      e.g.`torchvision.models.vgg.vgg13_bn(True).features`
        """
        super().__init__()

        self.lidar_channels = lidar_channels
        self.rgb_channels = rgb_channels

        self.rgb_net = deepcopy(proto)
        self.lidar_net = deepcopy(proto)
        _change_num_channels(self.lidar_net[0], self.lidar_channels)
        if self.rgb_channels != 3:
            _change_num_channels(self.rgb_net[0], self.rgb_channels)

    def forward(self, x):  # pylint:disable=arguments-differ
        """Do forward inference.

        This splits the input vector (x) into two slices corresponding
        to different input sources. Then each input source is processed
        by its own feature extraction net. A `fuse` operation is provided
        by derived classes in order to combine the features.

        :param x: The (concatenated) input features.
        """
        rgb = x[:, :self.rgb_channels]
        lidar = x[:, self.rgb_channels:]

        rgb_features = self.rgb_net(rgb)
        lidar_features = self.lidar_net(lidar)

        # Make sure the outputs are all the same shape even if we change the input size
        # NOTE: I am not sure that this is necessary -- without it we could make a
        #       fully convolutional net.
        rgb_features = F.adaptive_max_pool2d(rgb_features, output_size=(1, 1))
        lidar_features = F.adaptive_max_pool2d(lidar_features, output_size=(1, 1))

        fused = self.fuse(rgb_features, lidar_features)
        return fused

    def fuse(self, rgb_features, lidar_features):
        # pylint:disable=unused-argument
        """
        Abstract method to fuse RGB and LiDAR features.
        """
        raise NotImplementedError("Subclasses should implement a `fuse` operation")


class LateFusionAdd(LateFusion):
    """LateFusionAdd"""

    def fuse(self, rgb_features, lidar_features):
        """fuse rgb and lidar features by adding them (element by element)

        :param rgb: Features derived from RGB data
        :param lidar: Features derived from LiDAR data
        """
        return rgb_features + lidar_features


class LateFusionCat(LateFusion):
    """LateFusionCat"""

    def __init__(self, lidar_channels, rgb_channels, num_features, proto):
        """
        process the input sources seperately by and then combine
        features by concatenating them.

        The concatenated features are combined using a linear layer
        to consolidate them into `num_output` activations.

        :param lidar_channels: The number of channels in the lidar input
        :param rgb_channels: The number of RGB channels (i.e. 3)
        :param proto: A prototype net to base feature extraction on;
                      e.g.`torchvision.models.vgg.vgg13_bn(True).features`
        """
        super().__init__(lidar_channels, rgb_channels, proto)
        self.fusion_layer = nn.Linear(2 * num_features, num_features, bias=True)

    def fuse(self, rgb_features, lidar_features):
        """fuse rgb and lidar features by adding them (element by element)

        :param rgb: Features derived from RGB data
        :param lidar: Features derived from LiDAR data
        """
        # pylint:disable=no-member
        cat = torch.cat([rgb_features, lidar_features], dim=1)
        fused = self.fusion_layer(cat)
        return fused


class Architecture(nn.Module):
    #  pylint:disable=too-many-instance-attributes
    """
    Architecture
    -------------
    Build the network architecture for box detections.
    """

    def __init__(self, shape=(64, 64), lidar_channels=6, rgb_channels=3, num_features=512, **kwargs):
        """
        The network architecture.


        :param shape:
            The shape of the input windows (rows and columns)

        :param lidar_channels:
            The number of channels of LiDAR data (the volume depth)

        :param rgb_channels:
            The number of channels of RGB data (e.g. 3)

        :param fusion:
            How to do fusion; one of:

            - 'early' to concatenate the input channels before the 'features' net,

            - 'late_cat', to process inputs separately and concatenate them. The
               concatenated vector is reduced using a linear layer.

            - 'late_add' to process inputs separately and the combine them by adding them.

            See :class FusionOptions: for list the valid options.

        :param channel_dropout:
            Whether to selectively drop RGB or LIDAR data.

        :param channel_dropout_ratios:
            A triple with the probability to dropout color, lidar, or neither.
            For examples (1,2,7) means that ther is a one in ten chance that
            color is dropped and a 2 in ten chance that lidar is dropped.

        :param obb_parametrization:
            The parametrization of the OBB

        :param num_hidden:
            The number of hidden layers to use for classification and regression.


        """
        super().__init__()

        self.shape = shape
        self.lidar_channels = lidar_channels
        self.rgb_channels = rgb_channels
        self.num_features = num_features

        self.fusion = kwargs.pop('fusion', FusionOptions.EARLY)
        self.obb_parametrization = kwargs.pop('obb_parametrization', ObbOptions.VECTOR_AND_WIDTH)
        self.channel_dropout = kwargs.pop('channel_dropout', ChannelDropoutOptions.CDROP)
        self.channel_dropout_ratios = kwargs.pop('channel_dropout_ratios', (1, 1, 5))

        self.synthetic = kwargs.pop('synthetic', SyntheticOptions.NO_PRETRAIN)
        self.class_loss = kwargs.pop('class_loss', ClassLossOptions.XENT_LOSS)
        self.regression_loss = kwargs.pop('regression_loss', RegressionLossOptions.SMOOTH_L1)

        self.num_hidden = kwargs.pop('num_hidden', 2048)

        proto = vgg.vgg13_bn(pretrained=True)

        # Choose a feature extraction subbnet based on the `fusion` argument.
        if self.fusion == FusionOptions.EARLY:
            self.features = EarlyFusion(lidar_channels, rgb_channels, proto.features)
        elif self.fusion == FusionOptions.LATE_ADD:
            self.features = LateFusionAdd(lidar_channels, rgb_channels, proto.features)
        elif self.fusion == FusionOptions.LATE_CAT:
            self.features = LateFusionCat(lidar_channels, rgb_channels, num_features, proto.features)

        # Classify (determine if it is a box)
        num_classes = 2  # 0 = 'background' (not a box), 1 = 'object' (it is a box)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.num_hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.num_hidden, num_classes),
        )

        # Regress (estimate box parameters)
        if self.obb_parametrization == ObbOptions.VECTOR_AND_WIDTH:
            self.num_obb_parameters = 2 + 2 + 1  # origin + length-vector + width
        elif self.obb_parametrization == ObbOptions.TWO_VECTORS:
            self.num_obb_parameters = 2 + 2 + 2
        elif self.obb_parametrization == ObbOptions.FOUR_POINTS:
            self.num_obb_parameters = 4 * 2
        else:
            raise ValueError("obb_parametrization must be on of the `ObbOptions` values")

        self.regressor = nn.Sequential(
            nn.Linear(self.num_features, self.num_hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.num_hidden, self.num_obb_parameters),
        )

    def forward(self, x):
        # pylint:disable=arguments-differ
        """Forward pass through then net

        :param x: batch sized input; shape=(bs, num_rgb_channels +
                  num_lidar_channels, height, width)

        """
        if self.channel_dropout == ChannelDropoutOptions.CDROP:
            dropout = np.random.choice(['rgb', 'lidar', 'none'], p=self.channel_dropout_ratios)
            if dropout == 'rgb':
                self.x[:, :self.rgb_channels] = 0
            elif dropout == 'lidar':
                self.x[:, self.rgb_channels:] = 0

        features = self.features(x)
        logits = self.classifier(features)
        parameters = self.regressor(features)
        return (logits, parameters)


class EvaluationData(object):
    """
    Evaluation Data
    ---------------
    Evaluation metrics to use for logging, plotting, etc.
    These will be saved in a history.
    """

    # pylint:disable=too-many-instance-attributes
    def __init__(self, epoch, trn_loss, val_loss, confusion_matrix):
        super().__init__()
        eps = sys.float_info.epsilon
        self.epoch = epoch
        self.trn_loss = trn_loss
        self.val_loss = val_loss
        self.confusion_matrix = confusion_matrix
        self.true_neg, self.false_neg, self.false_pos, self.true_pos = confusion_matrix.flat
        self.accuracy = (self.true_pos + self.true_neg) / (sum(confusion_matrix.flat))
        self.precision = self.true_pos / (self.true_pos + self.false_pos + eps)
        self.recall = self.true_pos / (self.true_pos + self.false_neg + eps)
        self.beta = 2
        self.f_measure = (1 + self.beta**2) * self.precision * self.recall / (
            self.beta**2 * self.precision + self.recall + eps)

    def __repr__(self):
        return (f"epoch {self.epoch: 5} " + f"trn_loss={self.trn_loss: 5.2} " + f"val_loss={self.val_loss: 5.2} " +
                f"F_{self.beta}={self.f_measure: 5.1%}  " + f"accuracy={self.accuracy: 5.1%}  " +
                f"precision={self.precision: 5.1%}  " + f"recall={self.recall: 5.1%}")


class Solver(object):
    """
    Solver
    ------

    Does the training loop in order to fit a model to data.

    """

    # pylint:disable=too-many-instance-attributes

    def __init__(self, trn_loader, val_loader, net, **kwargs):
        """__init__

        :param trn_loader: A traiing dataloader (iterable that yields batches)
        :param val_loader: A validation dataloader
        :param net: The network with parameters to solve-for.


        :param optimizer: The torch optimizer to use (default: 'adam')

        :param class_loss: The classification loss function (see ClassLossOptions)

        :param max_epochs:
            The maximum number of epochs to iterate over.

        :param max_overfit:
            The maximim number of iterations to go past  our 'best' iteration

        """

        super().__init__()
        self.trn_loader = trn_loader
        self.val_loader = val_loader
        self.net = net
        self.history = []  # TODO: Use pandas.
        self.best_epoch = -1
        self.best_val_loss = float('inf')
        self.epoch = 0

        self.optimizer = kwargs.pop('optimizer', 'adam')
        self.class_loss = kwargs.pop('class_loss', ClassLossOptions.XENT_LOSS)
        self.regression_loss = kwargs.pop('regression_loss', RegressionLossOptions.SMOOTH_L1)

        self.max_epochs = kwargs.pop('max_epochs', 100)
        self.max_overfit = kwargs.pop('max_overfit', 5)

        # Load the optimizer based on the option
        if self.optimizer == 'adam':
            learnable_parameters = [p for p in self.net.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(learnable_parameters, lr=0.001, weight_decay=0.001)

        # If a string was passed for class_loss, load the corresponding loss function.
        if self.class_loss == ClassLossOptions.XENT_LOSS:
            self.class_loss = nn.CrossEntropyLoss(weight=None)
        elif self.class_loss == ClassLossOptions.HING_LOSS:
            self.class_loss = nn.HingeEmbeddingLoss(margin=1.0)

        # If a string was passed for regression_loss, load the corresponding loss function.
        if self.regression_loss == RegressionLossOptions.L2:
            self.regression_loss = nn.MSELoss()
        elif self.regression_loss == RegressionLossOptions.SMOOTH_L1:
            self.regression_loss = nn.SmoothL1Loss()

    def save_checkpoint(self, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
        """Save the model.

        :param filename: The filename of the current checkpoint
        :param best_filename: The filname of the best checkpoint
        """

        state = {
            'epoch': self.epoch,
            'history': self.history,
            'state_dict': self.net.state_dict(),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
        }
        is_best = self.epoch == self.best_epoch
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_filename)

    def restore_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']

    def train(self):

        # Compute initial validation
        # TODO: Refactor to a method: self.eval_current()--> should return EvaluationData and add to history
        val_loss = 0
        confusion_matrix = np.zeros((2, 2))
        self.net.eval()
        val_progressbar = tqdm(self.val_loader, "computing validation", leave=False)

        # TODO: I am not sure how we are formatting 'y' now. I assume it is
        #       is_box, **box_params
        for x, y in val_progressbar:
            expected_is_box = y[0]
            expected_obb_parameters = y[1:]

            x = x.cuda()
            expected_is_box = expected_is_box.cuda()
            expected_obb_parameters = expected_obb_parameters.cuda()

            logits, obb_parameters = self.net.forward(x).squeeze()
            predicted_is_box = torch.argmax(logits, dim=1).squeeze()
            confusion_matrix += np.bincount(2 * y + predicted_is_box, minlength=4).reshape(2, 2)

            # The classification loss
            class_loss = self.class_loss(logits, expected_is_box)

            if predicted_is_box:
                regression_loss = self.regression_loss(obb_parameters, expected_obb_parameters)
            else:
                regression_loss = 0

            loss = class_loss + regression_loss

            val_loss += loss.item()
            val_progressbar.set_description("val_loss={val_loss:5.2}".format(val_loss=val_loss))
        val_progressbar.close()
        self.history.append(
            EvaluationData(
                epoch=self.epoch, trn_loss=float('nan'), val_loss=val_loss, confusion_matrix=confusion_matrix))
        print(self.history[-1])

        # loop over the dataset multiple times
        # NOTE: This does _up to_ max_epochs _additional_ epochs.
        for ignored in trange(self.max_epochs):
            # Training for one epoch
            self.net.train()
            trn_loss = 0.0
            trn_progressbar = tqdm(self.trn_loader, 'batches', leave=False)
            for (x, y) in trn_progressbar:
                # put input ionto the GPU
                expected_is_box = y[0]
                expected_obb_parameters = y[1:]

                x = x.cuda()
                expected_is_box = expected_is_box.cuda()
                expected_obb_parameters = expected_obb_parameters.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                logits, obb_parameters = self.net(x)
                predicted_is_box = torch.argmax(logits, dim=1).squeeze()

                class_loss = self.class_loss(logits.squeeze(), expected_is_box)
                if predicted_is_box:
                    regression_loss = self.regression_loss(obb_parameters, expected_obb_parameters)
                else:
                    regression_loss = 0

                # TODO: Assign weights to these
                loss = class_loss + regression_loss

                # NOTE: This is the only reall difference between eval and training, so there is a lot of repeated code.
                loss.backward()
                self.optimizer.step()

                # print statistics
                trn_loss += loss.item()
                trn_progressbar.set_description("trn_loss={trn_loss:5.2}".format(trn_loss=trn_loss))
            trn_progressbar.close()

            # Compute validation
            val_loss = 0
            confusion_matrix = np.zeros((2, 2))
            self.net.eval()
            val_progressbar = tqdm(self.val_loader, "computing validation", leave=False)
            for x, y in val_progressbar:
                x, y = x.cuda(), y.cuda()
                p = self.net.forward(x).squeeze()
                c = torch.argmax(p, dim=1).squeeze()
                confusion_matrix += np.bincount(2 * y + c, minlength=4).reshape(2, 2)
                loss = self.class_loss(p, y)
                val_loss += loss.item()
                val_progressbar.set_description("val_loss={val_loss:5.2}".format(val_loss=val_loss))
            val_progressbar.close()

            self.epoch += 1
            self.history.append(
                EvaluationData(
                    epoch=self.epoch, trn_loss=trn_loss, val_loss=val_loss, confusion_matrix=confusion_matrix))

            print(self.history[-1])

            if val_loss < self.best_val_loss:
                self.best_epoch = self.epoch
                self.best_val_loss = val_loss

            self.save_checkpoint()

            if (self.epoch - self.best_epoch) >= self.max_overfit:
                print("No improvement in {} epochs, you are done!".format(self.epoch - self.best_epoch))
                break

        print('Finished Training')
