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


@Model.register("bert")
class Bert(Model):
	def __init__(self, vocab: Vocabulary,
				 hidden_size,
				 text_field_embedder: TextFieldEmbedder,
				 transformer: Seq2SeqEncoder,
				 masked_lm_feedforward: Optional[FeedForward] = None,
				 next_sentence_feedforward: Optional[FeedForward] = None,
				 initializer: InitializerApplicator = InitializerApplicator(),
				 regularizer: Optional[RegularizerApplicator] = None) -> None:
		super().__init__(vocab, regularizer)
		self._text_field_embedder = text_field_embedder
		self._transformer = transformer
		self._feedforward = nn.Linear(text_field_embedder.get_output_dim(), hidden_size)
		self._next_sentence_feedforward = next_sentence_feedforward
		self._masked_lm_feedforward = masked_lm_feedforward
		self._norm_layer = nn.LayerNorm(text_field_embedder.get_output_dim())
		self._masked_lm_accuracy = CategoricalAccuracy()
		self._next_sentence_accuracy = CategoricalAccuracy()
		self._loss = torch.nn.CrossEntropyLoss()
		initializer(self)

	def forward(self, tokens: Dict[str, torch.LongTensor],
				input_mask: torch.LongTensor,
				segment_ids: torch.LongTensor,
				next_sentence_labels: torch.FloatTensor,
				masked_lm_positions: torch.LongTensor,
				masked_lm_weights: torch.LongTensor,
				masked_lm_labels: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
		embedded_tokens = self._text_field_embedder(tokens)
		input_shape = embedded_tokens.size()
		batch_size = input_shape[0]
		seq_length = input_shape[1]
		width = input_shape[2]
		transformed_tokens = self._transformer(embedded_tokens, input_mask, segment_ids)
		first_token_tensor = transformed_tokens[:, 0, :]
		pooled_output = torch.tanh(self._feedforward(first_token_tensor))
		output_dict = {}
		embedding_table = self._text_field_embedder.get_embedding_by_name('bert')
		if masked_lm_labels is not None:
			(masked_lm_loss,
			 masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
				transformed_tokens, self._norm_layer, self._masked_lm_feedforward, embedding_table,
				masked_lm_positions, masked_lm_labels['bert'], masked_lm_weights)

			(next_sentence_loss, next_sentence_example_loss,
			 next_sentence_log_probs) = get_next_sentence_output(
				pooled_output, self._next_sentence_feedforward, next_sentence_labels)

			total_loss = masked_lm_loss + next_sentence_loss
			output_dict["loss"] = total_loss
			output_dict['masked_lm_loss'] = masked_lm_loss
			output_dict['masked_lm_example_loss'] = masked_lm_example_loss
			output_dict['masked_lm_log_probs'] = masked_lm_log_probs
			output_dict['next_sentence_loss'] = next_sentence_loss
			output_dict['next_sentence_example_loss'] = next_sentence_example_loss
			output_dict['next_sentence_log_probs'] = next_sentence_log_probs
			self._masked_lm_accuracy(masked_lm_log_probs, masked_lm_labels)
			self._next_sentence_accuracy(next_sentence_log_probs, next_sentence_labels)
		return output_dict

	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {
			'masked_lm_accuracy': self._masked_lm_accuracy.get_metric(reset),
			'next_sentence_accuracy': self._next_sentence_accuracy.get_metric(reset)
		}


def get_masked_lm_output(input_tensor, norm_layer, masked_lm_feedforward, output_weights, positions,
						 vocab_size, label_ids, label_weights):
	"""Get loss and log probs for the masked LM."""
	input_tensor = gather_indexes(input_tensor, positions)
	input_tensor = masked_lm_feedforward(input_tensor)
	input_tensor = norm_layer(input_tensor)
	logits = torch.matmul(input_tensor, output_weights.transpose(0, 1))
	log_probs = torch.nn.functional.softmax(logits, dim=-1)
	label_ids = label_ids.view(-1)
	label_weights = label_weights.view(-1)
	one_hot_labels = torch.FloatTensor(label_ids.size(0), vocab_size)
	one_hot_labels.zero_()
	one_hot_labels.scatter_(1, label_ids, 1)
	# short to have the maximum number of predictions). The `label_weights`
	# tensor has a value of 1.0 for every real prediction and 0.0 for the
	# padding predictions.
	per_example_loss = -(log_probs * one_hot_labels).sum(-1)
	numerator = torch.sum(label_weights * per_example_loss)
	denominator = torch.sum(label_weights) + 1e-5
	loss = numerator / denominator
	return (loss, per_example_loss, log_probs)


def get_next_sentence_output(input_tensor, next_sentence_feedforward, labels):
	"""Get loss and log probs for the next sentence prediction."""
	# Simple binary classification. Note that 0 is "next sentence" and 1 is
	# "random sentence". This weight matrix is not used after pre-training.
	logits = next_sentence_feedforward(input_tensor)
	log_probs = torch.nn.functional.softmax(logits, dim=-1)
	labels = labels.view(-1)
	one_hot_labels = torch.FloatTensor(labels.size(0), 2)
	one_hot_labels.zero_()
	one_hot_labels.scatter_(1, labels, 1)
	per_example_loss = -(one_hot_labels * log_probs).sum(-1)
	loss = torch.sum(per_example_loss)
	return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
	"""Gathers the vectors at the specific positions over a minibatch."""
	sequence_shape = sequence_tensor.size()
	batch_size = sequence_shape[0]
	seq_length = sequence_shape[1]
	width = sequence_shape[2]
	flat_offsets = util.get_range_vector(batch_size, util.get_device_of(sequence_tensor)) * seq_length
	flat_offsets = flat_offsets.unsqueeze(-1)
	flat_positions = (positions + flat_offsets).view(-1)
	flat_sequence_tensor = sequence_tensor.view(batch_size * seq_length, width)
	output_tensor = torch.index_select(flat_sequence_tensor, 0, flat_positions)
	return output_tensor
