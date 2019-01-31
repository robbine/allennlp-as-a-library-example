import torch
import argparse
import sys
sys.path.append('.')
from allennlp.common.util import import_submodules
from allennlp.common.from_params import FromParams
from allennlp.models.model import Model
import logging
import os
from my_library.models.joint_intent_slot_model import JointIntentSlotModel, JointIntentSlotModelGoogleBert
from my_library.modules.token_embedders.embedding_v2 import EmbeddingV2
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from my_library.modules.token_embedders.embedding_v2 import _read_pretrained_embeddings_file
from my_library.modules.seq2seq_encoders.transformer import Transformer
LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serialization-dir', type=str, default='', help='serialization dir path')
    parser.add_argument('--output-pt', type=str, default='model.pt', help='exported jit trace file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    vocabulary = os.path.join(args.serialization_dir, 'vocabulary')
    vocab = Vocabulary.from_files(vocabulary)
    # bert_embedder = PretrainedBertEmbedder(args.pretrained_model, True)
    # token_embedders = {'tokens': bert_embedder}
    # basic_text_field_embedder = BasicTextFieldEmbedder(token_embedders)
    # weight=_read_pretrained_embeddings_file("/home/icepine.hans/data/ChineseEmbedding.tar.gz", 200, vocab)
    embedding = EmbeddingV2(False,
                            num_embeddings=26729,
                            embedding_dim=200,
                            padding_index=0,
                            trainable=False)
    token_embedders = {'tokens': embedding}
    basic_text_field_embedder = BasicTextFieldEmbedder(token_embedders)
    transformer = Transformer(attention_dropout_prob=0.1,
        attention_type="dot_product",
        dropout_prob=0.1,
        input_size=200,
        intermediate_act_fn="gelu",
        intermediate_size=3072,
        key_depth=1024,
        max_position_embeddings=256,
        memory_size=200,
        num_heads=16,
        num_hidden_layers=16,
        type_vocab_size=2,
        use_fp16=False,
        value_depth=1024,
        use_token_type=False,
        use_position_embeddings=True)
    model = JointIntentSlotModel(
        text_field_embedder=basic_text_field_embedder,
        transformer=transformer,
        vocab=vocab,
        label_encoding="BIO",
        constrain_crf_decoding=True,
        calculate_span_f1=True,
        include_start_end_transitions=True,
        use_fp16=False)
    model_state = torch.load(os.path.join(args.serialization_dir, 'best.th'), map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    # An example input you would normally provide to your model's forward() method.
    example = torch.ones(1, 14, 200)
    example_mask = torch.ones(1, 14)
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_transformer = torch.jit.trace(example, example_mask)(model._transformer)
    output = traced_script_transformer((torch.ones(1, 14), torch.ones(1, 14)))
    print(output)
    torch.save(traced_script_transformer, args.output_pt)
    dummy_input = torch.ones(1, 14, 200, dtype=torch.float)
    dummy_mask = torch.ones(1, 14, dtype=torch.float)
    torch.onnx.export(model=model._transformer, args=(dummy_input, dummy_mask), f=args.output_pt, verbose=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
