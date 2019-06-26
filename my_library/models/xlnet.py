import logging
from typing import Dict, Optional
import math
from allennlp.models import Model
import torch
import torch.nn as nn
from torch.nn.functional import nll_loss
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules import FeedForward
from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("xlnet")
class XLNet(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 use_fp16,
                 text_field_embedder: TextFieldEmbedder,
                 transformer_encoder: Seq2SeqEncoder,
                 wait_user_input=False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self._use_fp16 = use_fp16
        self._vocab_bias = nn.Parameter(torch.zeros(vocab.get_vocab_size()))
        self._text_field_embedder = text_field_embedder
        self._transformer = transformer_encoder
        hidden_size = transformer_encoder.get_output_dim()
        self._feedforward = nn.Linear(transformer_encoder.get_output_dim(),
                                      hidden_size)
        self._next_sentence_feedforward = nn.Linear(hidden_size, 2)
        self._masked_lm_feedforward = nn.Linear(
            transformer_encoder.get_output_dim(),
            text_field_embedder.get_output_dim())
        self._norm_layer = nn.LayerNorm(text_field_embedder.get_output_dim())
        torch.nn.init.xavier_uniform(self._feedforward.weight)
        torch.nn.init.xavier_uniform(self._next_sentence_feedforward.weight)
        torch.nn.init.xavier_uniform(self._masked_lm_feedforward.weight)
        self._feedforward.bias.data.fill_(0)
        self._next_sentence_feedforward.bias.data.fill_(0)
        self._masked_lm_feedforward.bias.data.fill_(0)
        self._vocab_bias.data.fill_(0)
        self._masked_lm_accuracy = CategoricalAccuracy()
        self._next_sentence_accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        if self._use_fp16:
            self.half()
        # for name, p in self.named_parameters():
        # 	print(name, p.size())
        initializer(self)
        if wait_user_input:
            input("Press Enter to continue...")

    def _create_params(self):
        pass

    def forward(self):
        pass
