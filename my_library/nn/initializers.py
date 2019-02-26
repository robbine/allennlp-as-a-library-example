import logging
import re
import math
from typing import Callable, List, Tuple, Type, Iterable, Dict
import itertools

import torch
import torch.nn.init
from overrides import overrides

from allennlp.common import Registrable
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn.initializers import Initializer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Initializer.register('pretrained_v2')
class PretrainedModelInitializerV2(Initializer):
    """
    An initializer which allows initializing parameters using a pretrained model. The
    initializer will load all of the weights from the ``weights_file_path`` and use the
    name of the new parameters to index into the pretrained parameters. Therefore,
    by default, the names of the new and pretrained parameters must be the same.
    However, this behavior can be overridden using the ``parameter_name_overrides``,
    which remaps the name of the new parameter to the key which should be used
    to index into the pretrained parameters.

    The initializer will load all of the weights from the ``weights_file_path``
    regardless of which parameters will actually be used to initialize the new model.
    So, if you need to initialize several parameters using a pretrained model, the most
    memory-efficient way to do this is to use one ``PretrainedModelInitializer`` per
    weights file and use a regex to match all of the new parameters which need to be
    initialized.

    The below entry in the :class:`InitializerApplicator` parameters will initialize
    ``linear_1.weight`` and ``linear_2.weight`` using a pretrained model.
    ``linear_1.weight`` will be initialized to the pretrained
    parameters called ``linear_1.weight``, but ``linear_2.weight`` will be initialized
    to the pretrained parameters called ``linear_3.weight``::

       ["linear_1.weight|linear_2.weight",
           {
               "type": "pretrained_v2",
               "weights_file_path": "best.th",
               "parameter_name_overrides": {
                   "linear_2.weight": "linear_3.weight"
               }
           }
       ]

    Parameters
    ----------
    weights_file_path : ``str``, required
        The path to the weights file which has the pretrained model parameters.
    parameter_name_overrides : ``Dict[str, str]``, optional (default = None)
        The mapping from the new parameter name to the name which should be used
        to index into the pretrained model parameters. If a parameter name is not
        specified, the initializer will use the parameter's default name as the key.
    """

    def __init__(self,
                 weights_file_path: str,
                 parameter_name_overrides: Dict[str, str] = None) -> None:
        self.weights: Dict[str, torch.Tensor] = torch.load(
            weights_file_path, map_location=torch.device('cpu'))
        self.parameter_name_overrides = parameter_name_overrides or {}

    @overrides
    def __call__(self, tensor: torch.Tensor, parameter_name: str,
                 **kwargs) -> None:  # type: ignore
        # Select the new parameter name if it's being overridden
        if parameter_name in self.parameter_name_overrides:
            parameter_name = self.parameter_name_overrides[parameter_name]

        # If the size of the source and destination tensors are not the
        # same, then we need to raise an error
        source_weights = self.weights[parameter_name]
        if tensor.data.size() != source_weights.size():
            raise ConfigurationError(
                "Incompatible sizes found for parameter %s. "
                "Found %s and %s" % (parameter_name, tensor.data.size(),
                                     source_weights.size()))

        # Copy the parameters from the source to the destination
        tensor.data[:] = source_weights[:]
