import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
from my_library.modules.layers.pointer_network import *
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("rnet")
class RNet(Model):
	"""
	This class implements Minjoon Seo's `Bidirectional Attention Flow model
	<https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
	for answering reading comprehension questions (ICLR 2017).

	The basic layout is pretty simple: encode words as a combination of word embeddings and a
	character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
	attentions to put question information into the passage word representations (this is the only
	part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
	do a softmax over span start and span end.

	Parameters
	----------
	vocab : ``Vocabulary``
	text_field_embedder : ``TextFieldEmbedder``
		Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
	num_highway_layers : ``int``
		The number of highway layers to use in between embedding the input and passing it through
		the phrase layer.
	phrase_layer : ``Seq2SeqEncoder``
		The encoder (with its own internal stacking) that we will use in between embedding tokens
		and doing the bidirectional attention.
	modeling_layer : ``Seq2SeqEncoder``
		The encoder (with its own internal stacking) that we will use in between the bidirectional
		attention and predicting span start and end.
	dropout : ``float``, optional (default=0.2)
		If greater than 0, we will apply dropout with this probability after all encoders (pytorch
		LSTMs do not apply dropout to their last layer).
	mask_lstms : ``bool``, optional (default=True)
		If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
		with only a slight performance decrease, if any.  We haven't experimented much with this
		yet, but have confirmed that we still get very similar performance with much faster
		training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
		required when using masking with pytorch LSTMs.
	initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
		Used to initialize the model parameters.
	regularizer : ``RegularizerApplicator``, optional (default=``None``)
		If provided, will be used to calculate the regularization penalty during training.
	"""

	def __init__(self, vocab: Vocabulary,
				 hidden_size: int,
				 is_bidirectional: bool,
				 text_field_embedder: TextFieldEmbedder,
				 num_highway_layers: int,
				 phrase_layer: Seq2SeqEncoder,
				 gated_attention_layer: Seq2SeqEncoder,
				 self_attention_layer: Seq2SeqEncoder,
				 dropout: float = 0.2,
				 mask_lstms: bool = True,
				 initializer: InitializerApplicator = InitializerApplicator(),
				 regularizer: Optional[RegularizerApplicator] = None) -> None:
		super(RNet, self).__init__(vocab, regularizer)

		self.hidden_size = hidden_size
		self.is_bidirectional = is_bidirectional
		self._text_field_embedder = text_field_embedder
		self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
													  num_highway_layers))
		self._phrase_layer = phrase_layer
		self._gated_attention_layer = gated_attention_layer
		self._self_attention_layer = self_attention_layer

		encoding_dim = phrase_layer.get_output_dim()
		gated_attention_dim = gated_attention_layer.get_output_dim()
		self_attention_dim = self_attention_layer.get_output_dim()

		self._pointer_network = PointerNet(self_attention_dim, self.hidden_size, encoding_dim, self.is_bidirectional)
		self._span_start_accuracy = CategoricalAccuracy()
		self._span_end_accuracy = CategoricalAccuracy()
		self._span_accuracy = BooleanAccuracy()
		self._squad_metrics = SquadEmAndF1()
		if dropout > 0:
			self._dropout = torch.nn.Dropout(p=dropout)
		else:
			self._dropout = lambda x: x
		self._mask_lstms = mask_lstms

		initializer(self)

	def forward(self,  # type: ignore
				question: Dict[str, torch.LongTensor],
				passage: Dict[str, torch.LongTensor],
				span_start: torch.IntTensor = None,
				span_end: torch.IntTensor = None,
				metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
		# pylint: disable=arguments-differ
		"""
		Parameters
		----------
		question : Dict[str, torch.LongTensor]
			From a ``TextField``.
		passage : Dict[str, torch.LongTensor]
			From a ``TextField``.  The model assumes that this passage contains the answer to the
			question, and predicts the beginning and ending positions of the answer within the
			passage.
		span_start : ``torch.IntTensor``, optional
			From an ``IndexField``.  This is one of the things we are trying to predict - the
			beginning position of the answer with the passage.  This is an `inclusive` token index.
			If this is given, we will compute a loss that gets included in the output dictionary.
		span_end : ``torch.IntTensor``, optional
			From an ``IndexField``.  This is one of the things we are trying to predict - the
			ending position of the answer with the passage.  This is an `inclusive` token index.
			If this is given, we will compute a loss that gets included in the output dictionary.
		metadata : ``List[Dict[str, Any]]``, optional
			If present, this should contain the question ID, original passage text, and token
			offsets into the passage for each instance in the batch.  We use this for computing
			official metrics using the official SQuAD evaluation script.  The length of this list
			should be the batch size, and each dictionary should have the keys ``id``,
			``original_passage``, and ``token_offsets``.  If you only want the best span string and
			don't care about official metrics, you can omit the ``id`` key.

		Returns
		-------
		An output dictionary consisting of:
		span_start_logits : torch.FloatTensor
			A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
			probabilities of the span start position.
		span_start_probs : torch.FloatTensor
			The result of ``softmax(span_start_logits)``.
		span_end_logits : torch.FloatTensor
			A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
			probabilities of the span end position (inclusive).
		span_end_probs : torch.FloatTensor
			The result of ``softmax(span_end_logits)``.
		best_span : torch.IntTensor
			The result of a constrained inference over ``span_start_logits`` and
			``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
			and each offset is a token index.
		loss : torch.FloatTensor, optional
			A scalar loss to be optimised.
		best_span_str : List[str]
			If sufficient metadata was provided for the instances in the batch, we also return the
			string from the original passage that the model thinks is the best answer to the
			question.
		"""
		embedded_question = self._highway_layer(self._text_field_embedder(question))
		embedded_passage = self._highway_layer(self._text_field_embedder(passage))
		batch_size = embedded_question.size(0)
		passage_length = embedded_passage.size(1)
		question_mask = util.get_text_field_mask(question).float()
		passage_mask = util.get_text_field_mask(passage).float()
		question_lstm_mask = question_mask if self._mask_lstms else None
		passage_lstm_mask = passage_mask if self._mask_lstms else None

		encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
		encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
		encoding_dim = encoded_question.size(-1)


		self._gated_attention_layer.transform_match_input(encoded_question)
		hidden = torch.zeros(1, batch_size, self._gated_attention_layer.get_output_dim())
		gated_output_arr = []
		for seq_idx in range(passage_length):
			output, hidden = self._gated_attention_layer(encoded_passage[:, seq_idx, :].unsqueeze(0), hidden)
			gated_output_arr.append(output)
		gated_passage = torch.cat(gated_output_arr, dim=0)

		if self._self_attention_layer.is_bidirectional:
			hidden = torch.zeros(2, batch_size, self._self_attention_layer.get_output_dim())
		else:
			hidden = torch.zeros(1, batch_size, self._self_attention_layer.get_output_dim())
		output, hidden = self._self_attention_layer(gated_passage, hidden)
		modeled_passage = output
		modeling_dim = modeled_passage.size(-1)

		hidden = self._pointer_network.build_attention(encoded_question)
		span_start_logits, span_end_logits = self._pointer_network(modeled_passage, hidden)
		span_start_logits = span_start_logits.squeeze(2).t()
		span_start_probs = util.masked_softmax(span_start_logits, passage_mask)
		span_end_logits = span_end_logits.squeeze(2).t()
		span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
		span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
		span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
		best_span = self.get_best_span(span_start_logits, span_end_logits)

		output_dict = {
			"span_start_logits": span_start_logits,
			"span_start_probs": span_start_probs,
			"span_end_logits": span_end_logits,
			"span_end_probs": span_end_probs,
			"best_span": best_span,
		}

		# Compute the loss for training.
		if span_start is not None:
			loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1))
			self._span_start_accuracy(span_start_logits, span_start.squeeze(-1))
			loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1))
			self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
			self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))
			output_dict["loss"] = loss

		# Compute the EM and F1 on SQuAD and add the tokenized input to the output.
		if metadata is not None:
			output_dict['best_span_str'] = []
			question_tokens = []
			passage_tokens = []
			for i in range(batch_size):
				question_tokens.append(metadata[i]['question_tokens'])
				passage_tokens.append(metadata[i]['passage_tokens'])
				passage_str = metadata[i]['original_passage']
				offsets = metadata[i]['token_offsets']
				predicted_span = tuple(best_span[i].detach().cpu().numpy())
				start_offset = offsets[predicted_span[0]][0]
				end_offset = offsets[predicted_span[1]][1]
				best_span_string = passage_str[start_offset:end_offset]
				output_dict['best_span_str'].append(best_span_string)
				answer_texts = metadata[i].get('answer_texts', [])
				if answer_texts:
					self._squad_metrics(best_span_string, answer_texts)
			output_dict['question_tokens'] = question_tokens
			output_dict['passage_tokens'] = passage_tokens
		return output_dict

	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		exact_match, f1_score = self._squad_metrics.get_metric(reset)
		return {
			'start_acc': self._span_start_accuracy.get_metric(reset),
			'end_acc': self._span_end_accuracy.get_metric(reset),
			'span_acc': self._span_accuracy.get_metric(reset),
			'em': exact_match,
			'f1': f1_score,
		}

	@staticmethod
	def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
		if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
			raise ValueError("Input shapes must be (batch_size, passage_length)")
		batch_size, passage_length = span_start_logits.size()
		max_span_log_prob = [-1e20] * batch_size
		span_start_argmax = [0] * batch_size
		best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

		span_start_logits = span_start_logits.detach().cpu().numpy()
		span_end_logits = span_end_logits.detach().cpu().numpy()

		for b in range(batch_size):  # pylint: disable=invalid-name
			for j in range(passage_length):
				val1 = span_start_logits[b, span_start_argmax[b]]
				if val1 < span_start_logits[b, j]:
					span_start_argmax[b] = j
					val1 = span_start_logits[b, j]

				val2 = span_end_logits[b, j]

				if val1 + val2 > max_span_log_prob[b]:
					best_word_span[b, 0] = span_start_argmax[b]
					best_word_span[b, 1] = j
					max_span_log_prob[b] = val1 + val2
		return best_word_span

