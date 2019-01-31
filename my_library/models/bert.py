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

def gelu(x):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

@Model.register("bert")
class Bert(Model):
	def __init__(self, vocab: Vocabulary, use_fp16,
				 text_field_embedder: TextFieldEmbedder,
				 transformer: Seq2SeqEncoder,
				 wait_user_input = False,
				 initializer: InitializerApplicator = InitializerApplicator(),
				 regularizer: Optional[RegularizerApplicator] = None) -> None:
		super().__init__(vocab, regularizer)
		self._use_fp16 = use_fp16
		self._vocab_bias = nn.Parameter(torch.zeros(vocab.get_vocab_size()))
		self._text_field_embedder = text_field_embedder
		self._transformer = transformer
		hidden_size = transformer.get_output_dim()
		self._feedforward = nn.Linear(transformer.get_output_dim(), hidden_size)
		self._next_sentence_feedforward = nn.Linear(hidden_size, 2)
		self._masked_lm_feedforward = nn.Linear(transformer.get_output_dim(), text_field_embedder.get_output_dim())
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

	def partial_half(self):
		def to_half(t):
			if t.is_floating_point():
				return t.half()
			else:
				return t
		fn = to_half
		for name, module in self.named_children():
			print(name)
			if isinstance(module, nn.Linear) == False and isinstance(module, nn.LayerNorm) == False:
				module._apply(fn)

		# for param in self._parameters.values():
		# 	if param is not None:
		# 		# Tensors stored in modules are graph leaves, and we don't
		# 		# want to create copy nodes, so we have to unpack the data.
		# 		param.data = fn(param.data)
		# 		if param._grad is not None:
		# 			param._grad.data = fn(param._grad.data)
		#
		# for key, buf in self._buffers.items():
		# 	if buf is not None:
		# 		self._buffers[key] = fn(buf)


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
		embedding_table = self._text_field_embedder.get_embedding_by_name('tokens')
		masked_lm_loss = None
		next_sentence_loss = None
		if masked_lm_labels is not None:
			(masked_lm_loss,
			 masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(self._use_fp16,
				transformed_tokens, self._norm_layer, self._vocab_bias, self._masked_lm_feedforward, embedding_table,
				masked_lm_positions.long(), masked_lm_labels['tokens'], masked_lm_weights)
			output_dict['masked_lm_loss'] = masked_lm_loss
			output_dict['masked_lm_example_loss'] = masked_lm_example_loss
			output_dict['masked_lm_log_probs'] = masked_lm_log_probs
			self._masked_lm_accuracy(masked_lm_log_probs.float(), masked_lm_labels["tokens"].view(-1))
		if next_sentence_labels is not None:
			(next_sentence_loss, next_sentence_example_loss,
			 next_sentence_log_probs) = get_next_sentence_output(self._use_fp16,
				pooled_output, self._next_sentence_feedforward, next_sentence_labels)
			output_dict['next_sentence_loss'] = next_sentence_loss
			output_dict['next_sentence_example_loss'] = next_sentence_example_loss
			output_dict['next_sentence_log_probs'] = next_sentence_log_probs
			self._next_sentence_accuracy(next_sentence_log_probs.float(), next_sentence_labels)
		output_dict["loss"] = masked_lm_loss + next_sentence_loss
		return output_dict

	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {
			'accuracy': self._masked_lm_accuracy.get_metric(reset),
			'next_sentence_accuracy': self._next_sentence_accuracy.get_metric(reset)
		}


def get_masked_lm_output(use_fp16, input_tensor, norm_layer, bias, masked_lm_feedforward, output_weights, positions,
						 label_ids, label_weights):
	"""Get loss and log probs for the masked LM."""
	input_tensor = gather_indexes(input_tensor, positions)
	input_tensor = masked_lm_feedforward(input_tensor)
	input_tensor = gelu(input_tensor)
	input_tensor = norm_layer(input_tensor)
	logits = torch.matmul(input_tensor, output_weights.data.transpose(0, 1))
	logits = logits + bias
	log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
	label_ids = label_ids.view(-1, 1)
	label_weights = label_weights.view(-1)
	if use_fp16:
		label_weights = label_weights.half()
	vocab_size = output_weights.size(0)
	if label_ids.is_cuda:
		one_hot_labels = torch.cuda.FloatTensor(label_ids.size(0), vocab_size, device=util.get_device_of(label_ids))
	else:
		one_hot_labels = torch.FloatTensor(label_ids.size(0), vocab_size)
	if use_fp16:
		one_hot_labels = one_hot_labels.half()
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


def get_next_sentence_output(use_fp16, input_tensor, next_sentence_feedforward, labels):
	"""Get loss and log probs for the next sentence prediction."""
	# Simple binary classification. Note that 0 is "next sentence" and 1 is
	# "random sentence". This weight matrix is not used after pre-training.
	logits = next_sentence_feedforward(input_tensor)
	log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
	labels = labels.view(-1, 1)
	if labels.is_cuda:
		one_hot_labels = torch.cuda.FloatTensor(labels.size(0), 2, device=util.get_device_of(labels))
	else:
		one_hot_labels = torch.FloatTensor(labels.size(0), 2)
	if use_fp16:
		one_hot_labels = one_hot_labels.half()
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
	flat_offsets = flat_offsets.unsqueeze(-1).long()
	flat_positions = (positions + flat_offsets).view(-1)
	flat_sequence_tensor = sequence_tensor.view(batch_size * seq_length, width)
	output_tensor = torch.index_select(flat_sequence_tensor, 0, flat_positions)
	return output_tensor
