from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import torch
import argparse
import sys
sys.path.append('.')
from allennlp.common.util import import_submodules
from allennlp.common.from_params import FromParams
from allennlp.models.model import Model
import logging
import os
from model import JointIntentSlotDepsModel
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders.embedding import Embedding

from model import Transformer
LEVEL = logging.INFO

sys.path.insert(
    0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-pt',
        type=str,
        default='model.pt',
        help='exported jit trace file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocabulary = os.path.join('.', 'vocabulary')
    vocab = Vocabulary.from_files(vocabulary)
    embedding = Embedding(
        num_embeddings=26729,
        embedding_dim=200,
        padding_index=0,
        trainable=False)
    token_embedders = {'tokens': embedding}
    basic_text_field_embedder = BasicTextFieldEmbedder(token_embedders)
    transformer = Transformer(
        attention_dropout_prob=0.1,
        attention_type="dot_product",
        dropout_prob=0.1,
        input_size=200,
        intermediate_act_fn="gelu",
        intermediate_size=3072,
        key_depth=1024,
        max_position_embeddings=256,
        memory_size=200,
        num_heads=16,
        num_hidden_layers=6,
        type_vocab_size=2,
        use_fp16=False,
        value_depth=1024,
        use_token_type=True,
        use_position_embeddings=True)
    model = JointIntentSlotDepsModel(
        text_field_embedder=basic_text_field_embedder,
        transformer=transformer,
        vocab=vocab,
        label_encoding="BIO",
        constrain_crf_decoding=True,
        calculate_span_f1=True,
        include_start_end_transitions=True,
        use_fp16=False)
    dummy_input = torch.ones(1, 14, 200, dtype=torch.float)
    dummy_mask = torch.ones(1, 14, dtype=torch.float)
    segment_ids = torch.ones(1, 14, dtype=torch.float)
    output = model._transformer(dummy_input, dummy_mask, segment_ids)
    model_state = torch.load('/tmp/model.th')
    model.load_state_dict(model_state)

    torch.onnx.export(
        model=model._transformer,
        args=(dummy_input, dummy_mask, segment_ids),
        f=args.output_pt,
        verbose=True,
        export_params=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
