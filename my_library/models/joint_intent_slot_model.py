import logging
from typing import Dict, Optional

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


@Model.register("joint_intent_slot")
class JointIntentSlotModel(Model):
    def __init__(self, vocab: Vocabulary, use_fp16,
                 text_field_embedder: TextFieldEmbedder,
                 transformer: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 tag_namespace: str = "tags",
                 label_encoding: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self._use_fp16 = use_fp16
        self._text_field_embedder = text_field_embedder
        self._transformer = transformer
        self.num_intents = self.vocab.get_vocab_size(label_namespace)
        self.num_tags = self.vocab.get_vocab_size(tag_namespace)
        hidden_size = transformer.get_output_dim()

        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but "
                                         "no label_encoding was specified.")
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
                self.num_tags, constraints,
                include_start_end_transitions=include_start_end_transitions
        )
        self._feedforward = nn.Linear(transformer.get_output_dim(), hidden_size)
        self._intent_feedforward = nn.Linear(hidden_size, 2)
        self._masked_lm_feedforward = nn.Linear(transformer.get_output_dim(), text_field_embedder.get_output_dim())
        self._norm_layer = nn.LayerNorm(text_field_embedder.get_output_dim())
        torch.nn.init.xavier_uniform(self._feedforward.weight)
        torch.nn.init.xavier_uniform(self._next_sentence_feedforward.weight)
        torch.nn.init.xavier_uniform(self._masked_lm_feedforward.weight)
        self._feedforward.bias.data.fill_(0)
        self._next_sentence_feedforward.bias.data.fill_(0)
        self._masked_lm_feedforward.bias.data.fill_(0)
        self._masked_lm_accuracy = CategoricalAccuracy()
        self._next_sentence_accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but "
                                         "no label_encoding was specified.")
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=tag_namespace,
                                                 label_encoding=label_encoding)
        elif constraint_type is not None:
            # Maintain deprecated behavior if constraint_type is provided
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=tag_namespace,
                                                 label_encoding=constraint_type)
        if self._use_fp16:
            self.half()
        # for name, p in self.named_parameters():
        #     print(name, p.size())
        initializer(self)

    def forward(self, tokens: Dict[str, torch.LongTensor],
                input_mask: torch.LongTensor,
                segment_ids: torch.LongTensor,
                next_sentence_labels: torch.FloatTensor,
                masked_lm_positions: torch.LongTensor,
                masked_lm_weights: torch.LongTensor,
                masked_lm_labels: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        embedded_tokens = self._text_field_embedder(tokens)
        transformed_tokens = self._transformer(embedded_tokens, input_mask, segment_ids)
        first_token_tensor = transformed_tokens[:, 0, :]
        pooled_output = torch.tanh(self._feedforward(first_token_tensor))
        output_dict = {'encoded_layer': transformed_tokens, 'pooled_output': pooled_output}
        masked_lm_loss = None
        next_sentence_loss = None
        if masked_lm_labels is not None:
            (masked_lm_loss,
             masked_lm_example_loss, masked_lm_log_probs) = get_slot_output(self._use_fp16,
                transformed_tokens, self._norm_layer, self._masked_lm_feedforward,
                masked_lm_positions.long(), masked_lm_labels['tokens'], masked_lm_weights)
            output_dict['masked_lm_loss'] = masked_lm_loss
            output_dict['masked_lm_example_loss'] = masked_lm_example_loss
            output_dict['masked_lm_log_probs'] = masked_lm_log_probs
            self._masked_lm_accuracy(masked_lm_log_probs.float(), masked_lm_labels["tokens"].view(-1))
        if next_sentence_labels is not None:
            (next_sentence_loss, next_sentence_example_loss,
             next_sentence_log_probs) = get_intent_output(self._use_fp16,
                pooled_output, self._next_sentence_feedforward, next_sentence_labels)
            output_dict['next_sentence_loss'] = next_sentence_loss
            output_dict['next_sentence_example_loss'] = next_sentence_example_loss
            output_dict['next_sentence_log_probs'] = next_sentence_log_probs
            self._next_sentence_accuracy(next_sentence_log_probs.float(), next_sentence_labels)
        output_dict["loss"] = masked_lm_loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._masked_lm_accuracy.get_metric(reset),
            'next_sentence_accuracy': self._next_sentence_accuracy.get_metric(reset)
        }

    def get_slot_output(use_fp16, input_tensor, norm_layer, slot_feedforward, label_ids, label_weights):
        pass

    def get_intent_output(use_fp16, input_tensor, intent_feedforward, labels):
        pass
