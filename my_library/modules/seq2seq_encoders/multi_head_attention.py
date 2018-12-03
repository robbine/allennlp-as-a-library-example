from typing import Dict

from allennlp.modules import Seq2SeqEncoder
from overrides import overrides
import torch
import torch.nn as nn
from torch.nn import Dropout, Linear

from my_library.modules.layers import common_attention


@Seq2SeqEncoder.register("multi_head_attention")
class MultiHeadAttention(Seq2SeqEncoder):
	# pylint: disable=line-too-long
	"""
	This class implements the key-value scaled dot product attention mechanism
	detailed in the paper `Attention is all you Need
	<https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

	The attention mechanism is a weighted sum of a projection V of the inputs, with respect
	to the scaled, normalised dot product of Q and K, which are also both linear projections
	of the input. This procedure is repeated for each attention head, using different parameters.

	Parameters
	----------
	num_heads : ``int``, required.
		The number of attention heads to use.
	input_dim : ``int``, required.
		The size of the last dimension of the input tensor.
	attention_dim ``int``, required.
		The total dimension of the query and key projections which comprise the
		dot product attention function. Must be divisible by ``num_heads``.
	values_dim : ``int``, required.
		The total dimension which the input is projected to for representing the values,
		which are combined using the attention. Must be divisible by ``num_heads``.
	output_projection_dim : ``int``, optional (default = None)
		The dimensionality of the final output projection. If this is not passed
		explicitly, the projection has size `input_size`.
	attention_dropout_prob : ``float``, optional (default = 0.1).
		The dropout probability applied to the normalised attention
		distributions.
	"""

	def __init__(self,
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
				 max_relative_position=None,
				 heads_share_relative_embedding=True,
				 add_relative_to_values=False,
				 block_length=64,
				 ) -> None:

		super(MultiHeadAttention, self).__init__()

		self._num_heads = num_heads
		self._input_size = input_size
		self._memory_size = memory_size
		self._output_dim = input_size
		self._key_depth = key_depth
		self._value_depth = value_depth

		if key_depth % num_heads != 0:
			raise ValueError(f"Key size ({key_depth}) must be divisible by the number of "
							 f"attention heads ({num_heads}).")

		if value_depth % num_heads != 0:
			raise ValueError(f"Value size ({value_depth}) must be divisible by the number of "
							 f"attention heads ({num_heads}).")

		self._key_projection = Linear(memory_size, key_depth)
		self._value_projection = Linear(memory_size, value_depth)
		self._query_projection = Linear(input_size, key_depth)
		self._scale = (input_size // num_heads) ** 0.5
		self._output_projection = Linear(value_depth, self._output_dim)
		self._attention_dropout = Dropout(attention_dropout_prob)
		self._dropout = Dropout(dropout_prob)
		self._attention_type = attention_type
		self._vocab_size = max_relative_position * 2 + 1
		self._rel_embed_length = block_length * 4
		self._max_relative_position_unmasked = max_relative_position * 2 - 1
		self._norm_layer = nn.LayerNorm(value_depth)
		self._token_type_embedding = nn.Embedding(type_vocab_size, input_size)
		self._position_embedding = nn.Embedding(max_position_embeddings, input_size)
		self._key_embedding = None
		self._value_embedding = None
		self._relative_key_embeddings = None
		self._relative_value_embeddings = None
		if attention_type == 'dot_product_relative':
			self._key_embedding = nn.Parameter(torch.randn(self._vocab_size, self._key_depth))
			self._value_embedding = nn.Parameter(torch.randn(self._vocab_size, self._value_depth))
		elif attention_type == 'dot_product_unmasked_relative_v2':
			if heads_share_relative_embedding:
				self._relative_key_embeddings = nn.Parameter(torch.randn(self._rel_embed_length, self._key_depth))
				self._relative_value_embeddings = nn.Parameter(torch.randn(self._rel_embed_length, self._value_depth))
			else:
				self._relative_key_embeddings = nn.Parameter(
					torch.randn(self._num_heads, self._rel_embed_length, self._key_depth))
				self._relative_value_embeddings = nn.Parameter(
					torch.randn(self._num_heads, self._rel_embed_length, self._value_depth))
		elif attention_type == 'local_relative_mask_right':
			if heads_share_relative_embedding:
				self._relative_key_embeddings = nn.Parameter(torch.randn(self._rel_embed_length, self._key_depth))
				self._relative_value_embeddings = nn.Parameter(torch.randn(self._rel_embed_length, self._value_depth))
			else:
				self._relative_key_embeddings = nn.Parameter(
					torch.randn(self._num_heads, self._rel_embed_length, self._key_depth))
				self._relative_value_embeddings = nn.Parameter(
					torch.randn(self._num_heads, self._rel_embed_length, self._value_depth))
		elif attention_type == 'dot_product_unmasked_self_attention_relative_v2':
			if heads_share_relative_embedding:
				self._relative_key_embeddings = nn.Parameter(
					torch.randn(self._max_relative_position_unmasked, self._key_depth))
				self._relative_value_embeddings = nn.Parameter(
					torch.randn(self._max_relative_position_unmasked, self._value_depth))
			else:
				self._relative_key_embeddings = nn.Parameter(
					torch.randn(self._num_heads, self._max_relative_position_unmasked, self._key_depth))
				self._relative_value_embeddings = nn.Parameter(
					torch.randn(self._num_heads, self._max_relative_position_unmasked, self._value_depth))


	def get_input_dim(self):
		return self._input_size


	def get_output_dim(self):
		return self._value_depth


	@overrides
	def is_bidirectional(self):
		return False


	@overrides
	def forward(self,  # pylint: disable=arguments-differ
				tokens: Dict[str, torch.LongTensor],
				input_mask: torch.LongTensor,
				segment_ids: torch.LongTensor) -> torch.FloatTensor:
		"""
		Parameters
		----------
		inputs : ``torch.FloatTensor``, required.
			A tensor of shape (batch_size, timesteps, input_dim)
		mask : ``torch.FloatTensor``, optional (default = None).
			A tensor of shape (batch_size, timesteps).
	
		Returns
		-------
		A tensor of shape (batch_size, timesteps, input_dim),
		"""
		embedded_tokens = self._text_field_embedder(tokens)
		embedded_tokens = common_attention.embedding_postprocessor(embedded_tokens,
																   input_mask,
																   token_type_ids=segment_ids,
																   use_token_type=True,
																   token_type_embedding=self._token_type_embedding,
																   use_position_embeddings=True,
																   position_embedding=self._position_embedding,
																   norm_layer=self._norm_layer,
																   dropout=self._dropout)
		encoder_self_attention_bias = common_attention.create_attention_mask_from_input_mask(embedded_tokens,
																							 input_mask)
		outputs = common_attention.multihead_attention(embedded_tokens,
													   embedded_tokens,
													   encoder_self_attention_bias,
													   self._key_depth,
													   self._value_depth,
													   self._output_dim,
													   self._num_heads,
													   self._attention_dropout,
													   key_embedding=self._key_embedding,
													   value_embedding=self._value_embedding,
													   relative_key_embeddings=self._relative_key_embeddings,
													   relative_value_embeddings=self._relative_value_embeddings,
													   key_projection=self._key_projection,
													   value_projection=self._value_projection,
													   query_projection=self._query_projection,
													   attention_type=self._attention_type,
													   heads_share_relative_embedding=self._heads_share_relative_embedding,
													   add_relative_to_values=self._add_relative_to_values,
													   block_length=self._block_length,
													   block_width=self._block_width,
													   )

		# Project back to original input size.
		# shape (batch_size, timesteps, input_size)
		outputs = self._output_projection(outputs)
		return outputs
