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
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.vocabulary import Vocabulary

LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained-model', type=str, default='', help='pretrained model path')
    parser.add_argument('--vocabulary', type=str, default='', help='vocabulary path')
    parser.add_argument('--output-pt', type=str, default='model.pt', help='exported jit trace file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    vocab = Vocabulary.from_files(args.vocabulary)
    bert_embedder = PretrainedBertEmbedder(args.pretrained_model, True)
    token_embedders = {'tokens': bert_embedder}
    basic_text_field_embedder = BasicTextFieldEmbedder(token_embedders)
    model = JointIntentSlotModelGoogleBert(
        text_field_embedder=basic_text_field_embedder,
        vocab=vocab,
        label_encoding="BIO",
        constrain_crf_decoding=True,
        calculate_span_f1=True,
        include_start_end_transitions=True,
        use_fp16=False)
    # An example input you would normally provide to your model's forward() method.
    example = torch.ones(1, 14).long()
    example_mask = torch.ones(1, 14).long()
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example, example_mask)
    # output = traced_script_module([torch.ones(1, 14).long(), torch.ones(1, 14).long()])
    # print(output)
    torch.save(traced_script_module, args.output_pt)
    return 0

if __name__ == '__main__':
    sys.exit(main())
