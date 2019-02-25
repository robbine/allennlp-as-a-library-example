import logging
from typing import Dict, Optional, List, Any, Union
import warnings
import sys
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
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder

import torch.nn.functional as F
from allennlp.nn import util
import math

from allennlp.nn import Activation
from torch.nn import Dropout

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def comma_separated_string_to_integer_list(s):
    return [int(i) for i in s.split(",") if i]


def embedding_to_padding(emb):
    """Calculates the padding mask based on which embeddings are all zero.
	We have hacked symbol_modality to return all-zero embeddings for padding.
	Args:
	  emb: a Tensor with shape [..., depth].
	Returns:
	  a float Tensor with shape [...]. Each element is 1 if its corresponding
	  embedding vector is all zero, and is 0 otherwise.
	"""
    emb_sum = torch.sum(torch.abs(emb), dim=-1)
    equal_zero = emb_sum == 0
    return equal_zero.float()


def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.
	  Args:
	    memory_padding: a float `Tensor` with shape [batch, memory_length].
	  Returns:
	    a `Tensor` with shape [batch, 1, 1, memory_length].
	"""
    ret = memory_padding * -1e9
    return torch.unsqueeze(torch.unsqueeze(ret, dim=1), dim=1)


def attention_bias_proximal(length, device=-1):
    """Bias for self-attention to encourage attention to close positions.
	  Args:
	    length: an integer scalar.
	  Returns:
	    a Tensor with shape [1, 1, length, length]
	"""
    r = util.get_range_vector(length, device).float()
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(
        torch.unsqueeze(-torch.log(1 + torch.abs(diff)), 0), 0)


def get_timing_signal_1d(length,
                         channels,
                         device,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Gets a bunch of sinusoids of different frequencies.
	  Each channel of the input Tensor is incremented by a sinusoid of a different
	  frequency and phase.
	  This allows attention to learn to use absolute and relative positions.
	  Timing signals should be added to some precursors of both the query and the
	  memory inputs to attention.
	  The use of relative position is possible because sin(x+y) and cos(x+y) can be
	  expressed in terms of y, sin(x) and cos(x).
	  In particular, we use a geometric sequence of timescales starting with
	  min_timescale and ending with max_timescale.  The number of different
	  timescales is equal to channels / 2. For each timescale, we
	  generate the two sinusoidal signals sin(timestep/timescale) and
	  cos(timestep/timescale).  All of these sinusoids are concatenated in
	  the channels dimension.
	  Args:
	    length: scalar, length of timing signal sequence.
	    channels: scalar, size of timing embeddings to create. The number of
	        different timescales is equal to channels / 2.
	    min_timescale: a float
	    max_timescale: a float
	    start_index: index of first position
	  Returns:
	    a Tensor of timing signals [1, length, channels]
	"""
    position = util.get_range_vector(length, device) + start_index
    position = position.float()
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) / max(
            num_timescales - 1.0, 1.0))
    inv_timescales = min_timescale * torch.exp(
        util.get_range_vector(num_timescales, device).float() *
        -log_timescale_increment)
    scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(
        inv_timescales, 0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    pad = nn.ConstantPad1d((0, channels % 2), 0)
    signal = pad(signal)
    signal = signal.view(1, length, channels)
    return signal


def add_timing_signal_1d(x,
                         device,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
	  Each channel of the input Tensor is incremented by a sinusoid of a different
	  frequency and phase.
	  This allows attention to learn to use absolute and relative positions.
	  Timing signals should be added to some precursors of both the query and the
	  memory inputs to attention.
	  The use of relative position is possible because sin(x+y) and cos(x+y) can be
	  experessed in terms of y, sin(x) and cos(x).
	  In particular, we use a geometric sequence of timescales starting with
	  min_timescale and ending with max_timescale.  The number of different
	  timescales is equal to channels / 2. For each timescale, we
	  generate the two sinusoidal signals sin(timestep/timescale) and
	  cos(timestep/timescale).  All of these sinusoids are concatenated in
	  the channels dimension.
	  Args:
	    x: a Tensor with shape [batch, length, channels]
	    min_timescale: a float
	    max_timescale: a float
	    start_index: index of first position
	  Returns:
	    a Tensor the same shape as x.
	"""
    length = x.size(1)
    channels = x.size(2)
    signal = get_timing_signal_1d(length, channels, device, min_timescale,
                                  max_timescale, start_index)
    return x + signal


def attention_bias_to_padding(attention_bias):
    """Inverse of attention_bias_ignore_padding().
	  Args:
	    attention_bias: a `Tensor` with shape [batch, 1, 1, memory_length], as
	      returned by attention_bias_ignore_padding().
	  Returns:
	    a Tensor with shape [batch, memory_length] with 1.0 in padding positions
	    and 0.0 in non-padding positions.
	  """
    # `attention_bias` is a large negative number in padding positions and 0.0
    # elsewhere.
    return torch.squeeze(
        torch.squeeze(((attention_bias < -1).float()), dim=1), dim=1)


def compute_qkv(use_fp16, query_antecedent, memory_antecedent,
                key_projection: nn.Linear, value_projection: nn.Linear,
                query_projection: nn.Linear):
    """Computes query, key and value.
	Args:
	  query_antecedent: a Tensor with shape [batch, length_q, channels]
	  memory_antecedent: a Tensor with shape [batch, length_m, channels]
	Returns:
	  q, k, v : [batch, length, depth] tensors
	"""
    if memory_antecedent is None:
        memory_antecedent = query_antecedent
    q = query_projection(query_antecedent)
    k = key_projection(memory_antecedent)
    v = value_projection(memory_antecedent)
    return q, k, v


def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).
	Args:
	  x: a Tensor with shape [batch, length, channels]
	  num_heads: an integer
	Returns:
	  a Tensor with shape [batch, num_heads, length, channels / num_heads]
	"""
    batch, length, channels = x.size()
    per_head = x.view(batch, length, num_heads, channels // num_heads)
    return per_head.transpose(1, 2).contiguous()


def dot_product_attention(q, k, v, bias=None, dropout=None):
    """Dot-product attention.
	Args:
	  q: Tensor with shape [..., length_q, depth_k].
	  k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
	    match with q.
	  v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
	    match with q.
	  bias: bias Tensor (see attention_bias())
	  dropout_rate: a float.
	  image_shapes: optional tuple of integer scalars.
	    see comments for attention_image_summary()
	  name: an optional string
	  make_image_summary: True if you want an image summary.
	  save_weights_to: an optional dictionary to capture attention weights
	    for visualization; the weights tensor will be appended there under
	    a string key created from the variable scope (including name).
	  dropout_broadcast_dims: an optional list of integers less than rank of q.
	    Specifies in which dimensions to broadcast the dropout decisions.
	Returns:
	  Tensor with shape [..., length_q, depth_v].
	"""
    logits = torch.matmul(q, k.transpose(-1, -2))  # [..., length_q, length_kv]
    if bias is not None:
        adder = (1.0 - bias) * -10000.0
        logits += adder
    weights = torch.nn.functional.softmax(logits, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, v)


def gelu(x):
    """Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
	"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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
	attention_dropout_prob : ``float``, optional (default = 0.1).
		The dropout probability applied to the normalised attention
		distributions.
	"""

    def __init__(
            self,
            use_fp16: bool,
            num_heads: int,
            input_size: int,
            memory_size: int,
            key_depth: int,
            value_depth: int,
            max_position_embeddings: int = 512,
            type_vocab_size: int = 3,
            attention_dropout_prob: float = 0.1,
            attention_type: str = 'dot_product',
            max_relative_position=5,
            heads_share_relative_embedding=True,
            add_relative_to_values=False,
            block_length=64,
            block_width=64,
    ) -> None:

        super(MultiHeadAttention, self).__init__()
        self._use_fp16 = use_fp16
        self._num_heads = num_heads
        self._input_size = input_size
        self._memory_size = memory_size
        self._key_depth = key_depth
        self._value_depth = value_depth

        if key_depth % num_heads != 0:
            raise ValueError(
                f"Key size ({key_depth}) must be divisible by the number of "
                f"attention heads ({num_heads}).")

        if value_depth % num_heads != 0:
            raise ValueError(
                f"Value size ({value_depth}) must be divisible by the number of "
                f"attention heads ({num_heads}).")

        self._value_projection = nn.Linear(memory_size, value_depth)
        self._key_projection = nn.Linear(memory_size, key_depth)
        self._query_projection = nn.Linear(input_size, key_depth)
        torch.nn.init.xavier_uniform_(self._key_projection.weight)
        torch.nn.init.xavier_uniform_(self._value_projection.weight)
        torch.nn.init.xavier_uniform_(self._query_projection.weight)
        self._key_projection.bias.data.fill_(0)
        self._value_projection.bias.data.fill_(0)
        self._query_projection.bias.data.fill_(0)
        self._attention_dropout = Dropout(attention_dropout_prob)

        self._attention_type = attention_type
        self._vocab_size = max_relative_position * 2 + 1
        self._rel_embed_length = block_length * 4
        self._max_relative_position_unmasked = max_relative_position * 2 - 1
        self._key_embedding = None
        self._value_embedding = None
        self._relative_key_embeddings = None
        self._relative_value_embeddings = None
        self._heads_share_relative_embedding = heads_share_relative_embedding
        self._add_relative_to_values = add_relative_to_values
        self._block_length = block_length
        self._block_width = block_width
        if attention_type == 'dot_product_relative':
            self._key_embedding = nn.Parameter(
                torch.Tensor(self._vocab_size, self._key_depth))
            self._value_embedding = nn.Parameter(
                torch.Tensor(self._vocab_size, self._value_depth))
            torch.nn.init.xavier_uniform(self._key_embedding)
            torch.nn.init.xavier_uniform(self._value_embedding)
        elif attention_type == 'dot_product_unmasked_relative_v2':
            if heads_share_relative_embedding:
                self._relative_key_embeddings = nn.Parameter(
                    torch.Tensor(self._rel_embed_length, self._key_depth))
                self._relative_value_embeddings = nn.Parameter(
                    torch.Tensor(self._rel_embed_length, self._value_depth))
            else:
                self._relative_key_embeddings = nn.Parameter(
                    torch.Tensor(self._num_heads, self._rel_embed_length,
                                 self._key_depth))
                self._relative_value_embeddings = nn.Parameter(
                    torch.Tensor(self._num_heads, self._rel_embed_length,
                                 self._value_depth))
            torch.nn.init.xavier_uniform(self._relative_key_embeddings)
            torch.nn.init.xavier_uniform(self._relative_value_embeddings)
        elif attention_type == 'local_relative_mask_right':
            if heads_share_relative_embedding:
                self._relative_key_embeddings = nn.Parameter(
                    torch.Tensor(self._rel_embed_length, self._key_depth))
                self._relative_value_embeddings = nn.Parameter(
                    torch.Tensor(self._rel_embed_length, self._value_depth))
            else:
                self._relative_key_embeddings = nn.Parameter(
                    torch.Tensor(self._num_heads, self._rel_embed_length,
                                 self._key_depth))
                self._relative_value_embeddings = nn.Parameter(
                    torch.Tensor(self._num_heads, self._rel_embed_length,
                                 self._value_depth))
            torch.nn.init.xavier_uniform(self._relative_key_embeddings)
            torch.nn.init.xavier_uniform(self._relative_value_embeddings)
        elif attention_type == 'dot_product_unmasked_self_attention_relative_v2':
            if heads_share_relative_embedding:
                self._relative_key_embeddings = nn.Parameter(
                    torch.Tensor(self._max_relative_position_unmasked,
                                 self._key_depth))
                self._relative_value_embeddings = nn.Parameter(
                    torch.Tensor(self._max_relative_position_unmasked,
                                 self._value_depth))
            else:
                self._relative_key_embeddings = nn.Parameter(
                    torch.Tensor(self._num_heads,
                                 self._max_relative_position_unmasked,
                                 self._key_depth))
                self._relative_value_embeddings = nn.Parameter(
                    torch.Tensor(self._num_heads,
                                 self._max_relative_position_unmasked,
                                 self._value_depth))
            torch.nn.init.xavier_uniform(self._relative_key_embeddings)
            torch.nn.init.xavier_uniform(self._relative_value_embeddings)

    def get_input_dim(self):
        return self._input_size

    def get_output_dim(self):
        return self._value_depth

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(
            self,  # pylint: disable=arguments-differ
            embedded_tokens: torch.FloatTensor,
            input_mask: torch.LongTensor,
            encoder_self_attention_bias: torch.LongTensor
    ) -> torch.FloatTensor:
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
        outputs = common_attention.multihead_attention(
            self._use_fp16,
            embedded_tokens,
            embedded_tokens,
            encoder_self_attention_bias,
            self._key_depth,
            self._value_depth,
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
            heads_share_relative_embedding=self.
            _heads_share_relative_embedding,
            add_relative_to_values=self._add_relative_to_values,
            block_length=self._block_length,
            block_width=self._block_width,
        )
        return outputs


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
                 use_fp16: bool,
                 num_hidden_layers: int,
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
                 use_token_type=True,
                 use_position_embeddings=True,
                 attention_type: str = 'dot_product',
                 max_relative_position=5,
                 heads_share_relative_embedding=True,
                 add_relative_to_values=False,
                 block_length=64,
                 block_width=64) -> None:
        super(Transformer, self).__init__()
        hidden_size = input_size
        self._use_fp16 = use_fp16
        self._use_token_type = use_token_type
        self._use_position_embeddings = use_position_embeddings
        self._norm_layer = nn.LayerNorm(input_size)
        self._token_type_embedding = Embedding(type_vocab_size, input_size)
        self._position_embedding = Embedding(max_position_embeddings,
                                             input_size)
        self._dropout = Dropout(dropout_prob)
        if intermediate_act_fn == 'gelu':
            self._activation = gelu
        else:
            self._activation = Activation.by_name(intermediate_act_fn)()
        self._attention_layers = nn.ModuleList()
        self._layer_norm_output_layers = nn.ModuleList()
        self._layer_norm_layers = nn.ModuleList()
        self._feedforward_layers = nn.ModuleList()
        self._feedforward_output_layers = nn.ModuleList()
        self._feedforward_intermediate_layers = nn.ModuleList()

        for i in range(num_hidden_layers):
            self_attention = MultiHeadAttention(
                use_fp16, num_heads, input_size, memory_size, key_depth,
                value_depth, max_position_embeddings, type_vocab_size,
                attention_dropout_prob, attention_type, max_relative_position,
                heads_share_relative_embedding, add_relative_to_values,
                block_length, block_width)
            layer_norm_output = nn.LayerNorm(hidden_size)
            layer_norm = nn.LayerNorm(hidden_size)
            feedforward_output = nn.Linear(self_attention.get_output_dim(),
                                           hidden_size)
            feedforward_intemediate = nn.Linear(hidden_size, intermediate_size)
            feedforward = nn.Linear(intermediate_size, hidden_size)
            self.add_module(f"self_attention_{i}", self_attention)
            self.add_module(f"layer_norm_output_{i}", layer_norm_output)
            self.add_module(f"layer_norm_{i}", layer_norm)
            self.add_module(f"feedforward_{i}", feedforward)
            self.add_module(f"feedforward_output_{i}", feedforward_output)
            self.add_module(f"feedforward_intermediate_{i}",
                            feedforward_intemediate)
            self._attention_layers.append(self_attention)
            self._layer_norm_output_layers.append(layer_norm_output)
            self._layer_norm_layers.append(layer_norm)
            self._feedforward_layers.append(feedforward)
            self._feedforward_output_layers.append(feedforward_output)
            self._feedforward_intermediate_layers.append(
                feedforward_intemediate)
            torch.nn.init.xavier_uniform_(feedforward_output.weight)
            torch.nn.init.xavier_uniform_(feedforward.weight)
            torch.nn.init.xavier_uniform_(feedforward_intemediate.weight)
            feedforward_output.bias.data.fill_(0)
            feedforward.bias.data.fill_(0)
            feedforward_intemediate.bias.data.fill_(0)

        self._input_dim = input_size
        self._output_dim = hidden_size

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
    def forward(self,
                embedded_tokens: torch.FloatTensor,
                input_mask: torch.LongTensor,
                segment_ids: torch.LongTensor = None):  # pylint: disable=arguments-differ
        embedded_tokens = common_attention.embedding_postprocessor(
            embedded_tokens,
            input_mask.long(),
            self._use_fp16,
            token_type_ids=segment_ids,
            use_token_type=self._use_token_type,
            token_type_embedding=self._token_type_embedding,
            use_position_embeddings=self._use_position_embeddings,
            position_embedding=self._position_embedding,
            norm_layer=self._norm_layer,
            dropout=self._dropout)
        encoder_self_attention_bias = common_attention.create_attention_mask_from_input_mask(
            embedded_tokens, input_mask, self._use_fp16)
        prev_output = embedded_tokens
        for (attention, feedforward_output, feedforward,
             feedforward_intemediate, layer_norm_output, layer_norm) in zip(
                 self._attention_layers, self._feedforward_output_layers,
                 self._feedforward_layers,
                 self._feedforward_intermediate_layers,
                 self._layer_norm_output_layers, self._layer_norm_layers):
            layer_input = prev_output
            attention_output = attention(layer_input, input_mask,
                                         encoder_self_attention_bias)
            attention_output = self._dropout(
                feedforward_output(attention_output))
            attention_output = layer_norm_output(attention_output +
                                                 layer_input)
            attention_intermediate = self._activation(
                feedforward_intemediate(attention_output))
            layer_output = self._dropout(feedforward(attention_intermediate))
            layer_output = layer_norm(layer_output + attention_output)
            prev_output = layer_output

        return prev_output


@Model.register("joint_intent_slot_deps")
class JointIntentSlotDepsModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 use_fp16,
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
                 wait_user_input=False,
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
        self.cnn_num_filters = 3
        self.cnn_ngram_filter_sizes = (2, 3)
        cnn_maxpool_output_dim = self.cnn_num_filters * len(
            self.cnn_ngram_filter_sizes)
        self.cnn_encoder = CnnEncoder(self.num_tags, self.cnn_num_filters,
                                      self.cnn_ngram_filter_sizes)
        hidden_size = transformer.get_output_dim()
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but "
                                         "no label_encoding was specified.")
            tag_labels = self.vocab.get_index_to_token_vocabulary(
                tag_namespace)
            constraints = allowed_transitions(label_encoding, tag_labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
            self.num_tags,
            constraints,
            include_start_end_transitions=include_start_end_transitions)
        self._feedforward = nn.Linear(transformer.get_output_dim(),
                                      hidden_size)
        self._intent_feedforward = nn.Linear(
            hidden_size + cnn_maxpool_output_dim, self.num_intents)
        self._tag_feedforward = nn.Linear(transformer.get_output_dim(),
                                          self.num_tags)
        self._norm_layer = nn.LayerNorm(transformer.get_output_dim())
        torch.nn.init.xavier_uniform_(self._feedforward.weight)
        torch.nn.init.xavier_uniform_(self._intent_feedforward.weight)
        torch.nn.init.xavier_uniform_(self._tag_feedforward.weight)
        self._feedforward.bias.data.fill_(0)
        self._intent_feedforward.bias.data.fill_(0)
        self._tag_feedforward.bias.data.fill_(0)
        self._intent_accuracy = CategoricalAccuracy()
        self._intent_accuracy_3 = CategoricalAccuracy(top_k=3)
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
            self._f1_metric = SpanBasedF1Measure(
                vocab,
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
    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output = {}
        top_k = 3
        output_tags = [[
            self.vocab.get_token_from_index(tag, namespace=self.tag_namespace)
            for tag in instance_tags
        ] for instance_tags in output_dict["tags"]]
        predictions = output_dict['intent_probs'].cpu().data.numpy()
        argmax_indices = np.argsort(-predictions, axis=-1)[0, :top_k]
        labels = [
            '{}:{}'.format(
                self.vocab.get_token_from_index(
                    x, namespace=self.label_namespace), predictions[0, x])
            for x in argmax_indices
        ]
        output['top 3 intents'] = [labels]
        output["slots"] = []
        extracted_results = []
        words = output_dict["words"][0][1:]
        slot_name = ''
        for tag, word in zip(output_tags[0], words):
            if tag.startswith('B-'):
                extracted_results.append([word])
                slot_name = tag.split('-')[1][2:-1]
            elif tag.startswith('I-'):
                extracted_results[-1].append(word)
            else:
                continue
        for result in extracted_results:
            output['slots'].append({slot_name: ''.join(result)})
        return output

    def forward(
            self,
            tokens: Dict[str, torch.LongTensor],
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
        pooled_output = self._norm_layer(
            torch.tanh(self._feedforward(first_token_tensor)))
        tag_logits = self._tag_feedforward(encoded_text)
        mask = input_mask[:, 1:].long()
        best_paths = self.crf.viterbi_tags(tag_logits, mask)
        intent_logits = self._intent_feedforward(
            torch.cat((pooled_output, self.cnn_encoder(tag_logits, mask)),
                      dim=-1))
        intent_probs = torch.nn.functional.softmax(intent_logits, dim=-1)
        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]
        output = {
            'tag_logits': tag_logits,
            'mask': input_mask,
            'tags': predicted_tags,
            'intent_probs': intent_probs
        }
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
            output["intents_loss"] = self._intent_loss(intent_logits,
                                                       labels.long().view(-1))
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
                    x: y
                    for x, y in f1_dict.items() if x == 'f1-measure-overall'
                })
        metrics_to_return['acc'] = self._intent_accuracy.get_metric(reset)
        metrics_to_return['acc3'] = self._intent_accuracy_3.get_metric(reset)
        return metrics_to_return


def main():
    embedding_dim = 200
    num_embeddings = 26729
    attention_dropout_prob = 0.1
    attention_type = "dot_product"
    dropout_prob = 0.1
    input_size = 200
    intermediate_act_fn = "gelu"
    intermediate_size = 3072
    key_depth = 1024
    max_position_embeddings = 256
    memory_size = 200
    num_heads = 16
    num_hidden_layers = 6
    type_vocab_size = 2
    use_fp16 = False
    value_depth = 1024
    use_token_type = False
    use_position_embeddings = True
    vocabulary = './vocabulary'

    vocab = Vocabulary.from_files(vocabulary)
    transformer = Transformer(
        use_fp16,
        num_hidden_layers,
        intermediate_size,
        intermediate_act_fn,
        num_heads,
        input_size,
        memory_size,
        key_depth,
        value_depth,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size)
    embedding = Embedding(num_embeddings, embedding_dim)
    text_field_embedder = BasicTextFieldEmbedder({'tokens': embedding})
    model = JointIntentSlotDepsModel(vocab, use_fp16, text_field_embedder,
                                     transformer)
    torch.save(model.state_dict(), '/tmp/model.th')


if __name__ == "__main__":
    sys.exit(main())
