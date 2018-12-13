"""
Assorted utilities for working with neural networks in AllenNLP.
"""
# pylint: disable=too-many-lines
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import logging
import math
import warnings

import torch

from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def get_one_tensor(size, device=-1):
	if device > -1:
		return torch.cuda.FloatTensor(*size, device=device).fill_(1)
	else:
		return torch.ones(*size)