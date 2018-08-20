""" Build the network architecture.


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
# pylint:disable=too-few-public-methods

import shutil
import sys
from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from srp.data.orientedboundingbox import OrientedBoundingBox
from torchvision.models import vgg

from srp.util import tqdm, trange

from srp.config import C

import torch.cuda

if torch.cuda.is_available():

    def to_device(x):
        return x.cuda()
else:

    def to_device(x):
        return x


class FusionOptions:
    """FusionOptions

    Enumeration of options for doing early fusion.
    """
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
        _change_num_channels(self.net[0], self.rgb_channels + self.lidar_channels)

        # The first 3 channels are RGB; those weights are okay to use as is
        self.net[0].weight.data[:, :self.rgb_channels, ...] = proto[0].weight[:, :self.rgb_channels, ...]

    def forward(self, x):  # pylint:disable=arguments-differ
        """forward

        :param x:
        """
        features = self.net(x)

        pooled_features = F.adaptive_max_pool2d(features, output_size=(1, 1))

        return pooled_features


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

        self.fusion = kwargs.pop('fusion', C.EXPERIMENT.FUSION)
        self.obb_parametrization = kwargs.pop('obb_parametrization', ObbOptions.VECTOR_AND_WIDTH)
        self.channel_dropout = kwargs.pop('channel_dropout', ChannelDropoutOptions.CDROP)
        self.channel_dropout_ratios = np.array(kwargs.pop('channel_dropout_ratios', (1, 1, 5)), dtype=np.float)
        self.channel_dropout_ratios /= self.channel_dropout_ratios.sum()

        self.synthetic = kwargs.pop('synthetic', SyntheticOptions.NO_PRETRAIN)
        self.class_loss_function = kwargs.pop('class_loss', ClassLossOptions.XENT_LOSS)
        self.regression_loss_function = kwargs.pop('regression_loss', RegressionLossOptions.SMOOTH_L1)

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

    def freeze_all(self):
        """Freeze the entire net;  presumably followed by unfreexing a part of it!
        """
        # TODO: Freeze all layers
        pass

    def unfreeze_input_layers(self):
        """Unfreaze the first learnable layer of the net; the layers closest to the input.

        Since we have never trained on this input, but it it _similar_ in structure,
        it may be useful to learn the first layer weights.
        """
        # TODO: Unfreeze the first conv layers
        pass

    def unfreeze_classification_layers(self):
        """Unfrease the classification layers"""
        # TODO IMplement me
        pass

    def unfreeze_regression_layers(self):
        """Unfreeze the regression layers"""
        # TODO: Unfreeze the regression layers
        pass

    def forward(self, x):
        # pylint:disable=arguments-differ
        """Forward pass through then net

        :param x: batch sized input; shape=(bs, num_rgb_channels +
                  num_lidar_channels, height, width)

        """
        if self.channel_dropout == ChannelDropoutOptions.CDROP:
            dropout = np.random.choice(['rgb', 'lidar', 'none'], p=self.channel_dropout_ratios)
            if dropout == 'rgb':
                x[:, :self.rgb_channels] = 0
            elif dropout == 'lidar':
                x[:, self.rgb_channels:] = 0
        
        # import pdb; pdb.set_trace()
        features = self.features(x)
        features = features.view(features.size(0), -1)
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

    - This accepts a number of controlling the training process.
    - The `iterate` method us used _both_ for trianing and evaluation; pass in `Solver.EVAL`
      or `Solver.TRAIN` to decide which.

    Class Attributes:

    :attribute EVAL: Indicate that the net should be used in evaluation mode (see Solver.iterate)
    :attribute TRAIN: Indicate that the net should be used in training modes (see Solver.iterate)

    """

    # pylint:disable=too-many-instance-attributes

    EVAL = 'eval'
    TRAIN = 'train'

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

        :param regression_weight:
            The weight to apply to regression loss; the total loss is a weighted sum
            of regression_loss and class_loss.

        :param classification_weight:
            The weight to apply to classification loss.

        :param learning_rate:
            The learning rate for the optimizer (default=1e-3)

        :param weight_decay:
            The amount of weight decay for the optimizer (default=1e-3)

        :param reaug_interval:
            The number of epochs to do before recomputing data augmentations (default=20)
        """

        super().__init__()
        self.trn_loader = trn_loader
        self.val_loader = val_loader
        self.net = net
        self.history = []  # TODO: Use pandas.
        self.best_epoch = -1
        self.best_val_loss = float('inf')
        self.epoch = 0

        self.optimizer = kwargs.pop('optimizer',  C.TRAIN.OPTIMIZER)
        self.class_loss_function = kwargs.pop('class_loss', ClassLossOptions.XENT_LOSS)
        self.regression_loss_function = kwargs.pop('regression_loss', RegressionLossOptions.SMOOTH_L1)

        self.classification_weight = kwargs.pop('classification_weight', C.TRAIN.CLASSIFICATION_WEIGHT)
        self.regression_weight = kwargs.pop('regression_weight', C.TRAIN.REGRESSION_WEIGHT)

        self.max_epochs = kwargs.pop('max_epochs', 100)
        self.max_overfit = kwargs.pop('max_overfit', 5)

        self.learning_rate = kwargs.pop('learning_rate', C.TRAIN.LEARNING_RATE)
        self.weight_decay = kwargs.pop('weight_decay', C.TRAIN.WEIGHT_DECAY)

        self.reaug_interval = kwargs.pop('reaug_interval', 20)

        # Summary data from the training loop
        self.validation_loss = float('nan')
        self.training_loss = float('nan')
        self.validation_confusion_matrix = np.zeros((2, 2))

        # Load the optimizer based on the option
        if self.optimizer == 'adam':
            learnable_parameters = [p for p in self.net.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(learnable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)

        # If a string was passed for class_loss, load the corresponding loss function.
        if self.class_loss_function == ClassLossOptions.XENT_LOSS:
            self.class_loss_function = nn.CrossEntropyLoss(weight=None)
        elif self.class_loss_function == ClassLossOptions.HING_LOSS:
            self.class_loss_function = nn.HingeEmbeddingLoss(margin=1.0)

        # If a string was passed for regression_loss, load the corresponding loss function.
        if self.regression_loss_function == RegressionLossOptions.L2:
            self.regression_loss_function = nn.MSELoss()
        elif self.regression_loss_function == RegressionLossOptions.SMOOTH_L1:
            self.regression_loss_function = nn.SmoothL1Loss()

    def save_checkpoint(self, filename=None, best_filename=None):
        """Save the model, possibly updating the best model.

        This saves the current model using the filename provided, and
        if the current model is also the best model (in terms of validation loss)
        then the model is also copied to (and overwites) the `best_filename` file.

        You may use a python format string (e.g. `checkpoint-{epoch:05}.pth.tar`) if
        you want to keep all models around; if you want to keep only the models which
        are the best-up-to-that-epoch, then you may use the same formatting string
        in the best_filename parameter.

        NOTE: None of the arguments passed to the constructor are saved in the
              checkpoint; the caller is responsible for saving those.

        :param filename: The filename of the current checkpoint.
        :param best_filename: The filename of the best checkpoint.

        """

        if filename is None: filename = C.TRAIN.CHECKPOINT_PATTERN
        if best_filename is None: best_filename = C.TRAIN.BEST_PATTERN

        filename = filename.format(epoch=self.epoch)
        best_filename = best_filename.format(epoch=self.epoch)

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
        """
        Restore the solver to a previously-saved state.

        We should be able to resume training from the saved state.
        The saved state includes the entire history so we can plot accuracy/etc. up
        to the point where the model was saved.

        :param filename: The name of a previously saved state.

        Examples:

        NOTE: These are not doctests because they rely on training histoty.

        ```
        s = Solver(trn_loader, val_loader, net)   # The saved state does not include these!
        s.restore_checkpoint('model_best.pth.tar')  # The best model from a prior run

        box = s.make_obb(s.net(some_data))  # Use the best net to get OBB parameters

        if (box):
            print("Found box:", box)
        else:
            print("Did not find a box.")
        ```
        """
        checkpoint = torch.load(filename)
        self.epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']

    # pylint:disable=inconsistent-return-statements
    @staticmethod
    def output_to_obb(output, obb_parametrization):
        """output_to_obb

        :param output:
        :param obb_parametrization:

        >>> output = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> print(Solver.output_to_obb(output, ObbOptions.FOUR_POINTS))
        None

        >>> output = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0])
        >>> print(Solver.output_to_obb(output, ObbOptions.TWO_VECTORS))
        None

        >>> output = torch.Tensor([1, 0, 0, 0, 0, 0, 0])
        >>> print(Solver.output_to_obb(output, ObbOptions.VECTOR_AND_WIDTH))
        None

        # Check that it generate a positive example correctly
        >>> box = OrientedBoundingBox.from_rot_length_width((1,2), deg=30.0, length=2.0, width=1.0)

        >>> output = torch.Tensor([0, 1, 1.616, 2.933, -0.116, 1.933, 0.384, 1.067, 2.116, 2.067])
        >>> other = Solver.output_to_obb(output, ObbOptions.FOUR_POINTS)
        >>> other.iou(box) > 0.99
        True

        >>> output = torch.Tensor([0.0, 1.0, 1.0, 2.0, 0.866, 0.5, -0.25, 0.433])
        >>> other = Solver.output_to_obb(output, ObbOptions.TWO_VECTORS)
        >>> other.iou(box) > 0.99
        True

        >>> output = torch.Tensor([0.0, 1.0, 1.0, 2.0, 0.866, 0.5, 1.0])
        >>> other = Solver.output_to_obb(output, ObbOptions.VECTOR_AND_WIDTH)
        >>> other.iou(box) > 0.99
        True

        """

        if output[0] > output[1]:  # Negative
            return None
        output = output[2:]
        output = output.numpy()
        if obb_parametrization == ObbOptions.FOUR_POINTS:
            points = output.reshape((4, 2))
            return OrientedBoundingBox.from_points(points)
        elif obb_parametrization == ObbOptions.TWO_VECTORS:
            center, u_vector, v_vector = output.reshape((3, 2))
            points = np.array([u_vector + v_vector, -u_vector + v_vector, -u_vector - v_vector, u_vector - v_vector])
            points += center
            return OrientedBoundingBox.from_points(points)
        elif obb_parametrization == ObbOptions.VECTOR_AND_WIDTH:
            center = output[0:2]
            vector = output[2:4]
            width = output[4]
            return OrientedBoundingBox(center[0], center[1], vector[0], vector[1], width / 2.)

    @staticmethod
    def obb_to_output(obb, obb_parametrization):
        """
        Generate the expected net outputs for an obb.
        If obb is None, then the net should predict all zeros.

        This dose _not_ predict anything regarding the label -- only the regressison outputs.

        Example:

        Passing `None` results in a negative sample (all 0's)

        >>> print(Solver.obb_to_output(None, ObbOptions.FOUR_POINTS))
        tensor([ 0., 0., 0., 0., 0., 0., 0., 0.])

        >>> print(Solver.obb_to_output(None, ObbOptions.TWO_VECTORS))
        tensor([ 0., 0., 0., 0., 0., 0.])

        >>> print(Solver.obb_to_output(None, ObbOptions.VECTOR_AND_WIDTH))
        tensor([ 0., 0., 0., 0., 0.])

        # Check that it generate a positive example correctly
        >>> box = OrientedBoundingBox.from_rot_length_width((1,2), deg=30.0, length=2.0, width=1.0)

        >>> print(Solver.obb_to_output(box, ObbOptions.FOUR_POINTS))
        tensor([ 1.616..., 2.933..., -0.116..., 1.933..., 0.38..., 1.067..., 2.116..., 2.067...])

        >>> print(Solver.obb_to_output(box, ObbOptions.TWO_VECTORS))
        tensor([ 1.0..., 2.0..., 0.866..., 0.50..., -0.25..., 0.433...])

        >>> print(Solver.obb_to_output(box, ObbOptions.VECTOR_AND_WIDTH))
        tensor([ 1.0..., 2.0..., 0.866..., 0.50..., 1.0...])

        """

        if obb_parametrization == ObbOptions.FOUR_POINTS:
            num_obb_params = 4 * 2
        elif obb_parametrization == ObbOptions.TWO_VECTORS:
            num_obb_params = 3 * 2
        elif obb_parametrization == ObbOptions.VECTOR_AND_WIDTH:
            num_obb_params = 5

        if obb is None:
            return torch.Tensor([0] * num_obb_params)
        elif obb_parametrization == ObbOptions.FOUR_POINTS:
            return torch.Tensor(list(obb.points().flat))
        elif obb_parametrization == ObbOptions.TWO_VECTORS:
            return torch.Tensor(list(obb.origin) + list(obb.u_vector) + list(obb.v_vector))
        # obb_parametrization == ObbOptions.VECTOR_AND_WIDTH:
        return torch.Tensor(list(obb.origin) + list(obb.u_vector) + [2 * obb.v_length])

    def iterate(self, mode='eval'):
        """Do a forward pass on the net and compute the loss and other stats; possibly updating weights.

        This is used both to evaluate the model and also to update the weights; since the code
        for each is nearly identical.

        If `mode=Solver.TRAIN`, then the solver goes through one epoch and the weights are updated.
        The `epoch` counter is increased by one.

        If `mode=Solver.EVAL`  the solver goes through the validation dataset; the epoch counter is
        not increased but out internal evaluation metrics are overwritten.

        :param mode: Whether to evaluate the net using the validation set (Solver.EVAL, or 'eval')
                     or to update the model using the training data (Solver.TRAIN, or 'train')

        """
        if mode == Solver.EVAL:
            self.net.eval()
            progressbar = tqdm(self.val_loader, "computing validation", leave=False)
        else:
            self.net.train()
            progressbar = tqdm(self.trn_loader, "computing validation", leave=False)
        loss = 0.0
        loss_count = 0
        #progressbar = tqdm(self.val_loader, "computing validation", leave=False)

        confusion_matrix = np.zeros((2, 2))
        for batch in progressbar:
            x, y = batch
            expected_is_box = y[0]
            expected_obb_points = y[1]

            # We need to convert the expected parameters to match our predictions
            expected_obbs = [OrientedBoundingBox.from_points(params) for params in expected_obb_points]
            expected_obb_parameters = [self.obb_to_output(obb, self.net.obb_parametrization) for obb in expected_obbs]
            expected_obb_parameters = torch.stack(expected_obb_parameters)

            x = to_device(x)
            expected_is_box = to_device(expected_is_box)
            expected_obb_parameters = to_device(expected_obb_parameters)

            logits, obb_parameters = self.net.forward(x)
            predicted_is_box = torch.argmax(logits, dim=1).squeeze()

            # Compute the loss
            class_loss = self.class_loss_function(logits, expected_is_box)

            # Only measure regression loss of there is a box
            regression_loss = self.regression_loss_function(obb_parameters[expected_is_box == 1],
                                                            expected_obb_parameters[expected_is_box == 1])

            total_loss = self.classification_weight * class_loss
            if sum(expected_is_box) > 0:
                total_loss += self.regression_weight * regression_loss

            loss += total_loss.item()
            if mode == Solver.EVAL:
                confusion_matrix += np.bincount(2 * expected_is_box + predicted_is_box, minlength=4).reshape(2, 2)
            else:  # mode == Solver.TRAIN
                total_loss.backward()
                self.optimizer.step()

            loss_count += 1
            progressbar.set_description("{mode} loss={loss:5.2}".format(mode=mode, loss=loss / loss_count))
        progressbar.close()

        if mode == Solver.EVAL:
            self.validation_loss = loss / loss_count
            self.validation_confusion_matrix = confusion_matrix
        else:  # mode == Solver.TRAIN
            self.training_loss = loss / loss_count
            self.epoch += 1

    def summarize(self):
        """Return and record evaluation-data for the current epoch.

        This creates a record of the current evaluation data (train and validation loss, confusion matrix, etc)
        and records it in our internal history. It also returns the newly created record.

        """
        summary = EvaluationData(
            epoch=self.epoch,
            trn_loss=self.training_loss,
            val_loss=self.validation_loss,
            confusion_matrix=self.validation_confusion_matrix)
        self.history.append(summary)
        return summary

    def train(self, num_epochs=None):
        """The training loop.

        This does multiple iterations of training and validation.
        It starts be evaluating the current net and then alternate between
        training the model and evaluating it.

        This loop uses early stoping; if the model fails to improve on the
        validation set after `self.max_overfit` epochs then the loop will stop early.
        Otherwise, it will proceed for 'num_epochs' additional epochs.

        Note that this does does _additional_ epochs; calling it multiple times
        will increase the number of epochs recording (not restart it)

        :param num_epochs: The number of additional epochs (default=self.max_epochs)
        """

        if num_epochs is None:
            num_epochs = self.max_epochs

        # Compute initial validation
        self.iterate(Solver.EVAL)
        print(self.summarize())

        # loop over the dataset multiple times
        # NOTE: This does _up to_ max_epochs _additional_ epochs.
        for dummy in trange(self.max_epochs):

            # TODO: Periodically recalculate the variations
            # if self.epoch % self.reaug_interval == 0:
            #     self.trn_loader.dataset.pre_augment()
            #     self.val_loader.dataset.pre_augment()

            self.iterate(Solver.TRAIN)
            self.iterate(Solver.EVAL)
            print(self.summarize())

            if self.validation_loss < self.best_val_loss:
                self.best_epoch = self.epoch
                self.best_val_loss = self.validation_loss

            self.save_checkpoint()

            if (self.epoch - self.best_epoch) >= self.max_overfit:
                print("No improvement in {} epochs, you are done!".format(self.epoch - self.best_epoch))
                break

        print('Finished Training')


if __name__ == '__main__':
    # pylint:disable=bare-except
    import doctest
    import pdb
    try:
        doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS, raise_on_error=True, verbose=True)
    except:  # noqa
        pdb.post_mortem()
