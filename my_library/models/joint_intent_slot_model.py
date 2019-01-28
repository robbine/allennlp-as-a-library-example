import logging
from typing import Dict, Optional, List, Any, Union
import warnings

from overrides import overrides
from typing import Dict, Optional
import numpy as np
from allennlp.models import Model
import torch
import torch.nn as nn
from torch.nn.functional import nll_loss
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, F1Measure
from allennlp.modules import FeedForward, ConditionalRandomField
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
                 calculate_intent_f1: bool = None,
                 dropout: Optional[float] = None,
                 wait_user_input = False,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.label_namespace = label_namespace
        self.tag_namespace = tag_namespace
        self._use_fp16 = use_fp16
        self._verbose_metrics = verbose_metrics
        self._text_field_embedder = text_field_embedder
        self._transformer = transformer
        self.num_intents = self.vocab.get_vocab_size(label_namespace)
        self.num_tags = self.vocab.get_vocab_size(tag_namespace)
        hidden_size = transformer.get_output_dim()
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

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
            tag_labels = self.vocab.get_index_to_token_vocabulary(tag_namespace)
            constraints = allowed_transitions(label_encoding, tag_labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
                self.num_tags, constraints,
                include_start_end_transitions=include_start_end_transitions
        )
        self._feedforward = nn.Linear(transformer.get_output_dim(), hidden_size)
        self._intent_feedforward = nn.Linear(hidden_size, self.num_intents)
        self._tag_feedforward = nn.Linear(transformer.get_output_dim(), self.num_tags)
        self._norm_layer = nn.LayerNorm(transformer.get_output_dim())
        torch.nn.init.xavier_uniform(self._feedforward.weight)
        torch.nn.init.xavier_uniform(self._intent_feedforward.weight)
        torch.nn.init.xavier_uniform(self._tag_feedforward.weight)
        self._feedforward.bias.data.fill_(0)
        self._intent_feedforward.bias.data.fill_(0)
        self._tag_feedforward.bias.data.fill_(0)
        self._intent_accuracy = CategoricalAccuracy()
        self._intent_accuracy_3 = CategoricalAccuracy(top_k = 3)
        self.metrics = {
                "slot_acc": CategoricalAccuracy(),
                "slot_acc3": CategoricalAccuracy(top_k=3)
        }
        self._intent_loss = torch.nn.CrossEntropyLoss()
        self.calculate_span_f1 = calculate_span_f1
        self.calculate_intent_f1 = calculate_intent_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but "
                                         "no label_encoding was specified.")
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=tag_namespace,
                                                 label_encoding=label_encoding)
        if self._use_fp16:
            self.half()
        # for name, p in self.named_parameters():
        #     print(name, p.size())
        initializer(self)
        if wait_user_input:
            input("Press Enter to continue...")

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output = {}
        top_k = 3
        output_tags = [
                [self.vocab.get_token_from_index(tag, namespace=self.tag_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]
        predictions = output_dict['intent_probs'].cpu().data.numpy()
        argmax_indices = np.argsort(-predictions, axis=-1)[0, :top_k]
        labels = ['{}:{}'.format(self.vocab.get_token_from_index(x, namespace=self.label_namespace), predictions[0, x]) for x in argmax_indices]
        output['intent'] = [labels]
        output["slot"] = []
        extracted_results = []
        words = output_dict["words"][0][1:]
        for tag, word in zip(output_tags[0], words):
            if tag.startswith('B-'):
                extracted_results.append([word])
            elif tag.startswith('I-'):
                extracted_results[-1].append(word)
            else:
                continue
        for result in extracted_results:
            output["slot"].append(''.join(result))
        return output

    def forward(self, tokens: Dict[str, torch.LongTensor],
                input_mask: torch.LongTensor,
                tags: torch.LongTensor = None,
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        embedded_tokens = self._text_field_embedder(tokens)
        transformed_tokens = self._transformer(embedded_tokens, input_mask)
        first_token_tensor = transformed_tokens[:, 0, :]
        encoded_text = transformed_tokens[:, 1:, :]
        pooled_output = self._norm_layer(torch.tanh(self._feedforward(first_token_tensor)))
        tag_logits = self._tag_feedforward(encoded_text)
        mask = input_mask[:, 1:].long()
        best_paths = self.crf.viterbi_tags(tag_logits, mask)
        intent_logits = self._intent_feedforward(pooled_output)
        intent_probs = torch.nn.functional.softmax(intent_logits, dim=-1)
        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]
        output = {'tag_logits': tag_logits, 'mask': input_mask, 'tags': predicted_tags, 'intent_probs': intent_probs}
        if tags is not None:
            # Add negative log-likelihood as loss
            tags = tags[:, 1:]
            log_likelihood = self.crf(tag_logits, tags, mask)
            output["slot_loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = tag_logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            mask = mask.float()
            # for metric in self.metrics.values():
            #     metric(class_probabilities, tags.contiguous(), mask)
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask)
        if labels is not None:
            output["intents_loss"] = self._intent_loss(intent_logits, labels.long().view(-1))
            self._intent_accuracy(intent_logits, labels)
            self._intent_accuracy_3(intent_logits, labels)
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        if 'slot_loss' in output and 'intents_loss' in output:
            output["loss"] = output["slot_loss"] + output["intents_loss"]
        elif 'slot_loss' in output:
            output["loss"] = output["slot_loss"]
        elif 'intents_loss' in output:
            output["loss"] = output["intents_loss"]
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}

        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({
                        x: y for x, y in f1_dict.items() if
                        x == 'f1-measure-overall'})
        metrics_to_return['acc'] = self._intent_accuracy.get_metric(reset)
        metrics_to_return['acc3'] = self._intent_accuracy_3.get_metric(reset)
        return metrics_to_return


@Model.register("joint_intent_slot-googlebert")
class JointIntentSlotModelGoogleBert(Model):
    def __init__(self, vocab: Vocabulary, use_fp16,
                 text_field_embedder: TextFieldEmbedder,
                 label_namespace: str = "labels",
                 tag_namespace: str = "tags",
                 label_encoding: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 calculate_span_f1: bool = None,
                 calculate_intent_f1: bool = None,
                 dropout: Optional[float] = None,
                 wait_user_input = False,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.tag_namespace = tag_namespace
        self._use_fp16 = use_fp16
        self._verbose_metrics = verbose_metrics
        self._text_field_embedder = text_field_embedder
        self.num_intents = self.vocab.get_vocab_size(label_namespace)
        self.num_tags = self.vocab.get_vocab_size(tag_namespace)
        hidden_size = self._text_field_embedder.get_output_dim()
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

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
            tag_labels = self.vocab.get_index_to_token_vocabulary(tag_namespace)
            constraints = allowed_transitions(label_encoding, tag_labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
                self.num_tags, constraints,
                include_start_end_transitions=include_start_end_transitions
        )
        self._feedforward = nn.Linear(hidden_size, hidden_size)
        self._intent_feedforward = nn.Linear(hidden_size, self.num_intents)
        self._tag_feedforward = nn.Linear(hidden_size, self.num_tags)
        self._norm_layer = nn.LayerNorm(hidden_size)
        torch.nn.init.xavier_uniform(self._feedforward.weight)
        torch.nn.init.xavier_uniform(self._intent_feedforward.weight)
        torch.nn.init.xavier_uniform(self._tag_feedforward.weight)
        self._feedforward.bias.data.fill_(0)
        self._intent_feedforward.bias.data.fill_(0)
        self._tag_feedforward.bias.data.fill_(0)
        self._intent_accuracy = CategoricalAccuracy()
        self._intent_accuracy_3 = CategoricalAccuracy(top_k = 3)
        self.metrics = {
                "slot_acc": CategoricalAccuracy(),
                "slot_acc3": CategoricalAccuracy(top_k=3)
        }
        self._intent_loss = torch.nn.CrossEntropyLoss()
        self.calculate_span_f1 = calculate_span_f1
        self.calculate_intent_f1 = calculate_intent_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but "
                                         "no label_encoding was specified.")
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=tag_namespace,
                                                 label_encoding=label_encoding)
        if self._use_fp16:
            self.half()
        # for name, p in self.named_parameters():
        #     print(name, p.size())
        initializer(self)
        if wait_user_input:
            input("Press Enter to continue...")

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output = {}
        top_k = 3
        output_tags = [
                [self.vocab.get_token_from_index(tag, namespace=self.tag_namespace)
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]
        predictions = output_dict['intent_probs'].cpu().data.numpy()
        argmax_indices = np.argsort(-predictions, axis=-1)[0, :top_k]
        labels = ['{}:{}'.format(self.vocab.get_token_from_index(x, namespace=self.label_namespace), predictions[0, x]) for x in argmax_indices]
        output['intent'] = [labels]
        output["slot"] = []
        extracted_results = []
        words = output_dict["words"][0][1:]
        for tag, word in zip(output_tags[0], words):
            if tag.startswith('B-'):
                extracted_results.append([word])
            elif tag.startswith('I-'):
                extracted_results[-1].append(word)
            else:
                continue
        for result in extracted_results:
            output["slot"].append(''.join(result))
        return output

    def forward(self, tokens: Union[torch.Tensor, Dict[str, torch.LongTensor]],
                input_mask: torch.LongTensor,
                tags: torch.LongTensor = None,
                labels: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        transformed_tokens = self._text_field_embedder(tokens)
        first_token_tensor = transformed_tokens[:, 0, :]
        encoded_text = transformed_tokens[:, 1:, :]
        pooled_output = self._norm_layer(torch.tanh(self._feedforward(first_token_tensor)))
        tag_logits = self._tag_feedforward(encoded_text)
        mask = input_mask[:, 1:].long()
        best_paths = self.crf.viterbi_tags(tag_logits, mask)
        intent_logits = self._intent_feedforward(pooled_output)
        intent_probs = torch.nn.functional.softmax(intent_logits, dim=-1)
        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]
        output = {'tag_logits': tag_logits, 'mask': input_mask, 'tags': predicted_tags, 'intent_probs': intent_probs}
        if tags is not None:
            # Add negative log-likelihood as loss
            tags = tags[:, 1:]
            log_likelihood = self.crf(tag_logits, tags, mask)
            output["slot_loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = tag_logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            mask = mask.float()
            # for metric in self.metrics.values():
            #     metric(class_probabilities, tags.contiguous(), mask)
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask)
        if labels is not None:
            output["intents_loss"] = self._intent_loss(intent_logits, labels.long().view(-1))
            self._intent_accuracy(intent_logits, labels)
            self._intent_accuracy_3(intent_logits, labels)
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]

        if 'slot_loss' in output and 'intents_loss' in output:
            output["loss"] = output["slot_loss"] + output["intents_loss"]
        elif 'slot_loss' in output:
            output["loss"] = output["slot_loss"]
        elif 'intents_loss' in output:
            output["loss"] = output["intents_loss"]
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}

        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({
                        x: y for x, y in f1_dict.items() if
                        x == 'f1-measure-overall'})
        metrics_to_return['acc'] = self._intent_accuracy.get_metric(reset)
        metrics_to_return['acc3'] = self._intent_accuracy_3.get_metric(reset)
        return metrics_to_return
