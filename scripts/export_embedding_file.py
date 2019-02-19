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

from allennlp.common import Registrable
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

_TOKENS = "tokens.txt"


def parse_args():
    """Parse command line arguments.

    Args:
        None.

    Returns:
        A argparse.Namespace object which contains all parsed argument values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serialization-dir', help='serialization dir path')
    parser.add_argument(
        '--weights-file-name', help='weigth file name', default='best.th')
    parser.add_argument(
        '--embedder-name',
        help='embedder name',
        default='_text_field_embedder.token_embedder_tokens.weight')
    parser.add_argument(
        '--vocab-file-folder', help='vocab file folder', default='vocabulary')
    parser.add_argument(
        '--output-embedding-file',
        help='output embedding file name',
        default='exported_embedding.tar.gz')
    args = parser.parse_args()
    return args


def load_weights(serialization_dir, weights_file_name, embedder_name):
    weights_file_path = os.path.join(serialization_dir, weights_file_name)
    weights: Dict[str, torch.Tensor] = torch.load(weights_file_path)
    for name, weight in weights.items():
        print(name)
    embedder_weight = weights[embedder_name]
    print(embedder_weight.size())
    return embedder_weight.data.cpu().numpy()[1:, :]


def load_vocab(serialization_dir, vocab_file_folder):
    vocab_file_path = os.path.join(serialization_dir, vocab_file_folder,
                                   _TOKENS)
    tokens = []
    with open(vocab_file_path, 'r') as f:
        for line in f:
            tokens.append(line.strip())
    return tokens


def save_embedding_file(weights, tokens, serialization_dir,
                        output_embedding_file):
    embedding_fn = os.path.join(serialization_dir, 'exported_embedding.txt')
    with open(embedding_fn, 'w') as f:
        size, dim = weights.shape
        f.write('{} {}\n'.format(size, dim))
        for i in range(size):
            f.write('{} {}\n'.format(
                tokens[i], ' '.join([str(s) for s in weights[i].tolist()])))
    with tarfile.open(
            os.path.join(serialization_dir, output_embedding_file),
            'w:gz') as tar_file:
        tar_file.add(embedding_fn, 'exported_embedding.txt')


def main():
    args = parse_args()
    weights = load_weights(args.serialization_dir, args.weights_file_name,
                           args.embedder_name)
    tokens = load_vocab(args.serialization_dir, args.vocab_file_folder)
    assert len(tokens) == weights.shape[0]
    save_embedding_file(weights, tokens, args.serialization_dir,
                        args.output_embedding_file)


if __name__ == "__main__":
    sys.exit(main())
