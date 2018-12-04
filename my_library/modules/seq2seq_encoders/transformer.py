from typing import List

from allennlp.nn import Activation
from overrides import overrides
import torch
import torch.nn as nn
from torch.nn import Dropout
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from my_library.modules.layers import common_attention
from my_library.modules.seq2seq_encoders.multi_head_attention import MultiHeadAttention


@Seq2SeqEncoder.register("transformer")
class Transformer(Seq2SeqEncoder):
	# pylint: disable=line-too-long
	"""
	Implements a stacked self-attention encoder similar to the Transformer
	architecture in `Attention is all you Need
	<https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

	This encoder combines 3 layers in a 'block':

	1. A 2 layer FeedForward network.
	2. Multi-headed self attention, which uses 2 learnt linear projections
	   to perform a dot-product similarity between every pair of elements
	   scaled by the square root of the sequence length.
	3. Layer Normalisation.

	These are then stacked into ``num_layers`` layers.

	Parameters
	----------
	input_dim : ``int``, required.
		The input dimension of the encoder.
	hidden_dim : ``int``, required.
		The hidden dimension used for the _input_ to self attention layers
		and the _output_ from the feedforward layers.
	projection_dim : ``int``, required.
		The dimension of the linear projections for the self-attention layers.
	feedforward_hidden_dim : ``int``, required.
		The middle dimension of the FeedForward network. The input and output
		dimensions are fixed to ensure sizes match up for the self attention layers.
	num_layers : ``int``, required.
		The number of stacked self attention -> feedfoward -> layer normalisation blocks.
	num_attention_heads : ``int``, required.
		The number of attention heads to use per layer.
	use_positional_encoding: ``bool``, optional, (default = True)
		Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
		as without this feature, the self attention layers have no idea of absolute or relative
		position (as they are just computing pairwise similarity between vectors of elements),
		which can be important features for many tasks.
	dropout_prob : ``float``, optional, (default = 0.1)
		The dropout probability for the feedforward network.
	residual_dropout_prob : ``float``, optional, (default = 0.2)
		The dropout probability for the residual connections.
	attention_dropout_prob : ``float``, optional, (default = 0.1)
		The dropout probability for the attention distributions in each attention layer.
	"""

	def __init__(self,
				 num_hidden_layers: int,
				 hidden_dim: int,
				 intermediate_size: int,
				 intermediate_act_fn: str,
				 num_heads: int,
				 input_size: int,
				 memory_size: int,
				 key_depth: int,
				 value_depth: int,
				 max_position_embeddings: int = 512,
				 type_vocab_size: int = 3,
				 attention_dropout_prob: float = 0.1,
				 dropout_prob: float = 0.1,
				 attention_type: str = 'dot_product',
				 max_relative_position=5,
				 heads_share_relative_embedding=True,
				 add_relative_to_values=False,
				 block_length=64,
				 block_width=64) -> None:
		super(Transformer, self).__init__()
		self._norm_layer = nn.LayerNorm(input_size)
		self._token_type_embedding = nn.Embedding(type_vocab_size, input_size)
		self._position_embedding = nn.Embedding(max_position_embeddings, input_size)
		self._dropout = Dropout(dropout_prob)
		self._attention_layers: List[MultiHeadAttention] = []
		self._layer_norm_layers: List[LayerNorm] = []
		self._feedforward_layers: List[FeedForward] = []
		self._feedforward_output_layers: List[FeedForward] = []
		self._feedforward_intermediate_layers: List[FeedForward] = []
		self._feed_forward_layer_norm_layers: List[LayerNorm] = []

		if input_size != hidden_dim:
			raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
							 (input_size, hidden_dim))
		for i in range(num_hidden_layers):
			self_attention = MultiHeadAttention(num_heads,
												input_size,
												memory_size,
												key_depth,
												value_depth,
												max_position_embeddings,
												type_vocab_size,
												attention_dropout_prob,
												attention_type,
												max_relative_position,
												heads_share_relative_embedding,
												add_relative_to_values,
												block_length,
												block_width)
			layer_norm = LayerNorm(hidden_dim)
			feedforward_output = FeedForward(self_attention.get_output_dim(), num_layers=1,
											 hidden_dims=hidden_dim, activations=Activation.by_name("linear")(),
											 dropout=dropout_prob)
			feedforward_intermediate = FeedForward(hidden_dim, num_layers=1,
												   activations=Activation.by_name(intermediate_act_fn)(),
												   hidden_dims=intermediate_size)
			feedforward = FeedForward(intermediate_size, num_layers=1,
									  hidden_dims=hidden_dim, activations=Activation.by_name("linear")(),
									  dropout=dropout_prob)
			feedforward_layer_norm = LayerNorm(feedforward.get_output_dim())
			self.add_module(f"self_attention_{i}", self_attention)
			self.add_module(f"layer_norm_{i}", layer_norm)
			self.add_module(f"feedforward_{i}", feedforward)
			self.add_module(f"feedforward_layer_norm_{i}", feedforward_layer_norm)
			self.add_module(f"feedforward_output_{i}", feedforward_output)
			self.add_module(f"feedforward_intermediate_{i}", feedforward_intermediate)
			self._attention_layers.append(self_attention)
			self._layer_norm_layers.append(layer_norm)
			self._feedforward_layers.append(feedforward)
			self._feedforward_output_layers.append(feedforward_output)
			self._feedforward_intermediate_layers.append(feedforward_intermediate)
			self._feed_forward_layer_norm_layers.append(feedforward_layer_norm)

		self._input_dim = input_size
		self._output_dim = self._attention_layers[-1].get_output_dim()

	@overrides
	def get_input_dim(self) -> int:
		return self._input_dim

	@overrides
	def get_output_dim(self) -> int:
		return self._output_dim

	@overrides
	def is_bidirectional(self):
		return False

	@overrides
	def forward(self, embedded_tokens: torch.FloatTensor,
				input_mask: torch.LongTensor,
				segment_ids: torch.LongTensor):  # pylint: disable=arguments-differ
		embedded_tokens = common_attention.embedding_postprocessor(embedded_tokens,
																   input_mask.long(),
																   token_type_ids=segment_ids.long(),
																   use_token_type=True,
																   token_type_embedding=self._token_type_embedding,
																   use_position_embeddings=True,
																   position_embedding=self._position_embedding,
																   norm_layer=self._norm_layer,
																   dropout=self._dropout)
		encoder_self_attention_bias = common_attention.create_attention_mask_from_input_mask(embedded_tokens,
																							 input_mask)
		prev_output = embedded_tokens
		for (attention,
			 feedforward_output,
			 feedforward_intermediate,
			 feedforward,
			 feedforward_layer_norm,
			 layer_norm) in zip(self._attention_layers,
								self._feedforward_output_layers,
								self._feedforward_intermediate_layers,
								self._feedforward_layers,
								self._feed_forward_layer_norm_layers,
								self._layer_norm_layers):
			layer_input = prev_output
			attention_output = attention(layer_input, input_mask, encoder_self_attention_bias)
			attention_output = feedforward_output(attention_output)
			attention_output = layer_norm(attention_output + layer_input)
			intermediate_output = feedforward_intermediate(attention_output)
			# Project output of attention encoder through a feedforward
			# network and back to the input size for the next layer.
			# shape (batch_size, timesteps, input_size)
			layer_output = layer_norm(feedforward(intermediate_output) + attention_output)
			prev_output = layer_output

		return prev_output
