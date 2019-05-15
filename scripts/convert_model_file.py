import logging
import re
import math
from typing import Callable, List, Tuple, Type, Iterable, Dict
import itertools
from overrides import overrides
import argparse
import sys
import os
import tarfile
import torch
import torch.nn.init
import numpy as np
import pprint
import json

from allennlp.common import Registrable
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TENSOR_MAPPINGS = {
    'bert.embeddings.position_embeddings.weight':
    '_transformer._position_embedding.weight',
    'bert.embeddings.token_type_embeddings.weight':
    '_transformer._token_type_embedding.weight',
    'bert.embeddings.LayerNorm.gamma':
    '_transformer._norm_layer.weight',
    'bert.embeddings.LayerNorm.beta':
    '_transformer._norm_layer.bias',
    'bert.encoder.layer.0.attention.self.query.weight':
    '_transformer._attention_layers.0._query_projection.weight',
    'bert.encoder.layer.0.attention.self.query.bias':
    '_transformer._attention_layers.0._query_projection.bias',
    'bert.encoder.layer.0.attention.self.key.weight':
    '_transformer._attention_layers.0._key_projection.weight',
    'bert.encoder.layer.0.attention.self.key.bias':
    '_transformer._attention_layers.0._key_projection.bias',
    'bert.encoder.layer.0.attention.self.value.weight':
    '_transformer._attention_layers.0._value_projection.weight',
    'bert.encoder.layer.0.attention.self.value.bias':
    '_transformer._attention_layers.0._value_projection.bias',
    'bert.encoder.layer.0.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.0.weight',
    'bert.encoder.layer.0.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.0.bias',
    'bert.encoder.layer.0.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.0.weight',
    'bert.encoder.layer.0.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.0.bias',
    'bert.encoder.layer.0.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.0.weight',
    'bert.encoder.layer.0.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.0.bias',
    'bert.encoder.layer.0.output.dense.weight':
    '_transformer._feedforward_layers.0.weight',
    'bert.encoder.layer.0.output.dense.bias':
    '_transformer._feedforward_layers.0.bias',
    'bert.encoder.layer.0.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.0.weight',
    'bert.encoder.layer.0.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.0.bias',
    'bert.encoder.layer.1.attention.self.query.weight':
    '_transformer._attention_layers.1._query_projection.weight',
    'bert.encoder.layer.1.attention.self.query.bias':
    '_transformer._attention_layers.1._query_projection.bias',
    'bert.encoder.layer.1.attention.self.key.weight':
    '_transformer._attention_layers.1._key_projection.weight',
    'bert.encoder.layer.1.attention.self.key.bias':
    '_transformer._attention_layers.1._key_projection.bias',
    'bert.encoder.layer.1.attention.self.value.weight':
    '_transformer._attention_layers.1._value_projection.weight',
    'bert.encoder.layer.1.attention.self.value.bias':
    '_transformer._attention_layers.1._value_projection.bias',
    'bert.encoder.layer.1.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.1.weight',
    'bert.encoder.layer.1.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.1.bias',
    'bert.encoder.layer.1.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.1.weight',
    'bert.encoder.layer.1.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.1.bias',
    'bert.encoder.layer.1.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.1.weight',
    'bert.encoder.layer.1.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.1.bias',
    'bert.encoder.layer.1.output.dense.weight':
    '_transformer._feedforward_layers.1.weight',
    'bert.encoder.layer.1.output.dense.bias':
    '_transformer._feedforward_layers.1.bias',
    'bert.encoder.layer.1.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.1.weight',
    'bert.encoder.layer.1.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.1.bias',
    'bert.encoder.layer.2.attention.self.query.weight':
    '_transformer._attention_layers.2._query_projection.weight',
    'bert.encoder.layer.2.attention.self.query.bias':
    '_transformer._attention_layers.2._query_projection.bias',
    'bert.encoder.layer.2.attention.self.key.weight':
    '_transformer._attention_layers.2._key_projection.weight',
    'bert.encoder.layer.2.attention.self.key.bias':
    '_transformer._attention_layers.2._key_projection.bias',
    'bert.encoder.layer.2.attention.self.value.weight':
    '_transformer._attention_layers.2._value_projection.weight',
    'bert.encoder.layer.2.attention.self.value.bias':
    '_transformer._attention_layers.2._value_projection.bias',
    'bert.encoder.layer.2.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.2.weight',
    'bert.encoder.layer.2.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.2.bias',
    'bert.encoder.layer.2.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.2.weight',
    'bert.encoder.layer.2.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.2.bias',
    'bert.encoder.layer.2.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.2.weight',
    'bert.encoder.layer.2.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.2.bias',
    'bert.encoder.layer.2.output.dense.weight':
    '_transformer._feedforward_layers.2.weight',
    'bert.encoder.layer.2.output.dense.bias':
    '_transformer._feedforward_layers.2.bias',
    'bert.encoder.layer.2.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.2.weight',
    'bert.encoder.layer.2.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.2.bias',
    'bert.encoder.layer.3.attention.self.query.weight':
    '_transformer._attention_layers.3._query_projection.weight',
    'bert.encoder.layer.3.attention.self.query.bias':
    '_transformer._attention_layers.3._query_projection.bias',
    'bert.encoder.layer.3.attention.self.key.weight':
    '_transformer._attention_layers.3._key_projection.weight',
    'bert.encoder.layer.3.attention.self.key.bias':
    '_transformer._attention_layers.3._key_projection.bias',
    'bert.encoder.layer.3.attention.self.value.weight':
    '_transformer._attention_layers.3._value_projection.weight',
    'bert.encoder.layer.3.attention.self.value.bias':
    '_transformer._attention_layers.3._value_projection.bias',
    'bert.encoder.layer.3.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.3.weight',
    'bert.encoder.layer.3.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.3.bias',
    'bert.encoder.layer.3.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.3.weight',
    'bert.encoder.layer.3.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.3.bias',
    'bert.encoder.layer.3.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.3.weight',
    'bert.encoder.layer.3.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.3.bias',
    'bert.encoder.layer.3.output.dense.weight':
    '_transformer._feedforward_layers.3.weight',
    'bert.encoder.layer.3.output.dense.bias':
    '_transformer._feedforward_layers.3.bias',
    'bert.encoder.layer.3.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.3.weight',
    'bert.encoder.layer.3.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.3.bias',
    'bert.encoder.layer.4.attention.self.query.weight':
    '_transformer._attention_layers.4._query_projection.weight',
    'bert.encoder.layer.4.attention.self.query.bias':
    '_transformer._attention_layers.4._query_projection.bias',
    'bert.encoder.layer.4.attention.self.key.weight':
    '_transformer._attention_layers.4._key_projection.weight',
    'bert.encoder.layer.4.attention.self.key.bias':
    '_transformer._attention_layers.4._key_projection.bias',
    'bert.encoder.layer.4.attention.self.value.weight':
    '_transformer._attention_layers.4._value_projection.weight',
    'bert.encoder.layer.4.attention.self.value.bias':
    '_transformer._attention_layers.4._value_projection.bias',
    'bert.encoder.layer.4.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.4.weight',
    'bert.encoder.layer.4.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.4.bias',
    'bert.encoder.layer.4.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.4.weight',
    'bert.encoder.layer.4.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.4.bias',
    'bert.encoder.layer.4.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.4.weight',
    'bert.encoder.layer.4.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.4.bias',
    'bert.encoder.layer.4.output.dense.weight':
    '_transformer._feedforward_layers.4.weight',
    'bert.encoder.layer.4.output.dense.bias':
    '_transformer._feedforward_layers.4.bias',
    'bert.encoder.layer.4.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.4.weight',
    'bert.encoder.layer.4.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.4.bias',
    'bert.encoder.layer.5.attention.self.query.weight':
    '_transformer._attention_layers.5._query_projection.weight',
    'bert.encoder.layer.5.attention.self.query.bias':
    '_transformer._attention_layers.5._query_projection.bias',
    'bert.encoder.layer.5.attention.self.key.weight':
    '_transformer._attention_layers.5._key_projection.weight',
    'bert.encoder.layer.5.attention.self.key.bias':
    '_transformer._attention_layers.5._key_projection.bias',
    'bert.encoder.layer.5.attention.self.value.weight':
    '_transformer._attention_layers.5._value_projection.weight',
    'bert.encoder.layer.5.attention.self.value.bias':
    '_transformer._attention_layers.5._value_projection.bias',
    'bert.encoder.layer.5.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.5.weight',
    'bert.encoder.layer.5.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.5.bias',
    'bert.encoder.layer.5.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.5.weight',
    'bert.encoder.layer.5.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.5.bias',
    'bert.encoder.layer.5.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.5.weight',
    'bert.encoder.layer.5.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.5.bias',
    'bert.encoder.layer.5.output.dense.weight':
    '_transformer._feedforward_layers.5.weight',
    'bert.encoder.layer.5.output.dense.bias':
    '_transformer._feedforward_layers.5.bias',
    'bert.encoder.layer.5.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.5.weight',
    'bert.encoder.layer.5.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.5.bias',
    'bert.encoder.layer.6.attention.self.query.weight':
    '_transformer._attention_layers.6._query_projection.weight',
    'bert.encoder.layer.6.attention.self.query.bias':
    '_transformer._attention_layers.6._query_projection.bias',
    'bert.encoder.layer.6.attention.self.key.weight':
    '_transformer._attention_layers.6._key_projection.weight',
    'bert.encoder.layer.6.attention.self.key.bias':
    '_transformer._attention_layers.6._key_projection.bias',
    'bert.encoder.layer.6.attention.self.value.weight':
    '_transformer._attention_layers.6._value_projection.weight',
    'bert.encoder.layer.6.attention.self.value.bias':
    '_transformer._attention_layers.6._value_projection.bias',
    'bert.encoder.layer.6.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.6.weight',
    'bert.encoder.layer.6.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.6.bias',
    'bert.encoder.layer.6.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.6.weight',
    'bert.encoder.layer.6.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.6.bias',
    'bert.encoder.layer.6.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.6.weight',
    'bert.encoder.layer.6.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.6.bias',
    'bert.encoder.layer.6.output.dense.weight':
    '_transformer._feedforward_layers.6.weight',
    'bert.encoder.layer.6.output.dense.bias':
    '_transformer._feedforward_layers.6.bias',
    'bert.encoder.layer.6.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.6.weight',
    'bert.encoder.layer.6.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.6.bias',
    'bert.encoder.layer.7.attention.self.query.weight':
    '_transformer._attention_layers.7._query_projection.weight',
    'bert.encoder.layer.7.attention.self.query.bias':
    '_transformer._attention_layers.7._query_projection.bias',
    'bert.encoder.layer.7.attention.self.key.weight':
    '_transformer._attention_layers.7._key_projection.weight',
    'bert.encoder.layer.7.attention.self.key.bias':
    '_transformer._attention_layers.7._key_projection.bias',
    'bert.encoder.layer.7.attention.self.value.weight':
    '_transformer._attention_layers.7._value_projection.weight',
    'bert.encoder.layer.7.attention.self.value.bias':
    '_transformer._attention_layers.7._value_projection.bias',
    'bert.encoder.layer.7.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.7.weight',
    'bert.encoder.layer.7.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.7.bias',
    'bert.encoder.layer.7.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.7.weight',
    'bert.encoder.layer.7.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.7.bias',
    'bert.encoder.layer.7.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.7.weight',
    'bert.encoder.layer.7.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.7.bias',
    'bert.encoder.layer.7.output.dense.weight':
    '_transformer._feedforward_layers.7.weight',
    'bert.encoder.layer.7.output.dense.bias':
    '_transformer._feedforward_layers.7.bias',
    'bert.encoder.layer.7.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.7.weight',
    'bert.encoder.layer.7.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.7.bias',
    'bert.encoder.layer.8.attention.self.query.weight':
    '_transformer._attention_layers.8._query_projection.weight',
    'bert.encoder.layer.8.attention.self.query.bias':
    '_transformer._attention_layers.8._query_projection.bias',
    'bert.encoder.layer.8.attention.self.key.weight':
    '_transformer._attention_layers.8._key_projection.weight',
    'bert.encoder.layer.8.attention.self.key.bias':
    '_transformer._attention_layers.8._key_projection.bias',
    'bert.encoder.layer.8.attention.self.value.weight':
    '_transformer._attention_layers.8._value_projection.weight',
    'bert.encoder.layer.8.attention.self.value.bias':
    '_transformer._attention_layers.8._value_projection.bias',
    'bert.encoder.layer.8.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.8.weight',
    'bert.encoder.layer.8.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.8.bias',
    'bert.encoder.layer.8.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.8.weight',
    'bert.encoder.layer.8.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.8.bias',
    'bert.encoder.layer.8.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.8.weight',
    'bert.encoder.layer.8.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.8.bias',
    'bert.encoder.layer.8.output.dense.weight':
    '_transformer._feedforward_layers.8.weight',
    'bert.encoder.layer.8.output.dense.bias':
    '_transformer._feedforward_layers.8.bias',
    'bert.encoder.layer.8.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.8.weight',
    'bert.encoder.layer.8.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.8.bias',
    'bert.encoder.layer.9.attention.self.query.weight':
    '_transformer._attention_layers.9._query_projection.weight',
    'bert.encoder.layer.9.attention.self.query.bias':
    '_transformer._attention_layers.9._query_projection.bias',
    'bert.encoder.layer.9.attention.self.key.weight':
    '_transformer._attention_layers.9._key_projection.weight',
    'bert.encoder.layer.9.attention.self.key.bias':
    '_transformer._attention_layers.9._key_projection.bias',
    'bert.encoder.layer.9.attention.self.value.weight':
    '_transformer._attention_layers.9._value_projection.weight',
    'bert.encoder.layer.9.attention.self.value.bias':
    '_transformer._attention_layers.9._value_projection.bias',
    'bert.encoder.layer.9.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.9.weight',
    'bert.encoder.layer.9.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.9.bias',
    'bert.encoder.layer.9.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.9.weight',
    'bert.encoder.layer.9.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.9.bias',
    'bert.encoder.layer.9.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.9.weight',
    'bert.encoder.layer.9.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.9.bias',
    'bert.encoder.layer.9.output.dense.weight':
    '_transformer._feedforward_layers.9.weight',
    'bert.encoder.layer.9.output.dense.bias':
    '_transformer._feedforward_layers.9.bias',
    'bert.encoder.layer.9.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.9.weight',
    'bert.encoder.layer.9.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.9.bias',
    'bert.encoder.layer.10.attention.self.query.weight':
    '_transformer._attention_layers.10._query_projection.weight',
    'bert.encoder.layer.10.attention.self.query.bias':
    '_transformer._attention_layers.10._query_projection.bias',
    'bert.encoder.layer.10.attention.self.key.weight':
    '_transformer._attention_layers.10._key_projection.weight',
    'bert.encoder.layer.10.attention.self.key.bias':
    '_transformer._attention_layers.10._key_projection.bias',
    'bert.encoder.layer.10.attention.self.value.weight':
    '_transformer._attention_layers.10._value_projection.weight',
    'bert.encoder.layer.10.attention.self.value.bias':
    '_transformer._attention_layers.10._value_projection.bias',
    'bert.encoder.layer.10.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.10.weight',
    'bert.encoder.layer.10.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.10.bias',
    'bert.encoder.layer.10.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.10.weight',
    'bert.encoder.layer.10.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.10.bias',
    'bert.encoder.layer.10.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.10.weight',
    'bert.encoder.layer.10.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.10.bias',
    'bert.encoder.layer.10.output.dense.weight':
    '_transformer._feedforward_layers.10.weight',
    'bert.encoder.layer.10.output.dense.bias':
    '_transformer._feedforward_layers.10.bias',
    'bert.encoder.layer.10.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.10.weight',
    'bert.encoder.layer.10.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.10.bias',
    'bert.encoder.layer.11.attention.self.query.weight':
    '_transformer._attention_layers.11._query_projection.weight',
    'bert.encoder.layer.11.attention.self.query.bias':
    '_transformer._attention_layers.11._query_projection.bias',
    'bert.encoder.layer.11.attention.self.key.weight':
    '_transformer._attention_layers.11._key_projection.weight',
    'bert.encoder.layer.11.attention.self.key.bias':
    '_transformer._attention_layers.11._key_projection.bias',
    'bert.encoder.layer.11.attention.self.value.weight':
    '_transformer._attention_layers.11._value_projection.weight',
    'bert.encoder.layer.11.attention.self.value.bias':
    '_transformer._attention_layers.11._value_projection.bias',
    'bert.encoder.layer.11.attention.output.dense.weight':
    '_transformer._feedforward_output_layers.11.weight',
    'bert.encoder.layer.11.attention.output.dense.bias':
    '_transformer._feedforward_output_layers.11.bias',
    'bert.encoder.layer.11.attention.output.LayerNorm.gamma':
    '_transformer._layer_norm_output_layers.11.weight',
    'bert.encoder.layer.11.attention.output.LayerNorm.beta':
    '_transformer._layer_norm_output_layers.11.bias',
    'bert.encoder.layer.11.intermediate.dense.weight':
    '_transformer._feedforward_intermediate_layers.11.weight',
    'bert.encoder.layer.11.intermediate.dense.bias':
    '_transformer._feedforward_intermediate_layers.11.bias',
    'bert.encoder.layer.11.output.dense.weight':
    '_transformer._feedforward_layers.11.weight',
    'bert.encoder.layer.11.output.dense.bias':
    '_transformer._feedforward_layers.11.bias',
    'bert.encoder.layer.11.output.LayerNorm.gamma':
    '_transformer._layer_norm_layers.11.weight',
    'bert.encoder.layer.11.output.LayerNorm.beta':
    '_transformer._layer_norm_layers.11.bias',
}


def parse_args():
    """Parse command line arguments.

    Args:
        None.

    Returns:
        A argparse.Namespace object which contains all parsed argument values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--print-tensors', help='whether print tensors', default=False)
    parser.add_argument('--serialization-dir', help='serialization dir path')
    parser.add_argument(
        '--weights-file-name', help='weigth file name', default='best.th')
    parser.add_argument(
        '--output-model-file',
        help='output model file name',
        default='google_bert.th')
    args = parser.parse_args()
    return args


def convert_weights(serialization_dir, weights_file_name, output_model_file):
    weights_file_path = os.path.join(serialization_dir, weights_file_name)
    weights: Dict[str, torch.Tensor] = torch.load(weights_file_path)
    for old_name, new_name in TENSOR_MAPPINGS.items():
        weights[new_name] = weights[old_name]
        del weights[old_name]
    output_weights_file_path = os.path.join(serialization_dir,
                                            output_model_file)
    torch.save(weights, output_weights_file_path)
    return


def main():
    args = parse_args()
    convert_weights(args.serialization_dir, args.weights_file_name,
                    args.output_model_file)


if __name__ == "__main__":
    sys.exit(main())
