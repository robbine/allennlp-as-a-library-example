from typing import List
import math

from allennlp.nn import Activation
from overrides import overrides
import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from my_library.modules.token_embedders.embedding_v2 import EmbeddingV2
from my_library.modules.layers import common_attention
from my_library.modules.seq2seq_encoders.multi_head_attention import MultiHeadAttention


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@Seq2SeqEncoder.register("transformer_encoder")
class TransformerEncoder(Seq2SeqEncoder):
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
                 type_vocab_size: int = 2,
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
        super(TransformerEncoder, self).__init__()
        hidden_size = input_size
        self._use_fp16 = use_fp16
        self._use_token_type = use_token_type
        self._use_position_embeddings = use_position_embeddings
        self._norm_layer = nn.LayerNorm(hidden_size)
        self._token_type_embedding = EmbeddingV2(self._use_fp16,
                                                 type_vocab_size, hidden_size)
        self._position_embedding = EmbeddingV2(
            self._use_fp16, max_position_embeddings, hidden_size)
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
        self.embed_scale = math.sqrt(hidden_size)

        for i in range(num_hidden_layers):
            self_attention = MultiHeadAttention(
                use_fp16, num_heads, input_size, memory_size, key_depth,
                value_depth, attention_dropout_prob, attention_type,
                max_relative_position, heads_share_relative_embedding,
                add_relative_to_values, block_length, block_width)
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
        embedded_tokens = embedded_tokens * self.embed_scale
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

        encoder_padding_mask = input_mask.eq(0)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        prev_output = embedded_tokens
        for (attention, feedforward_output, feedforward,
             feedforward_intemediate, layer_norm_output, layer_norm) in zip(
                 self._attention_layers, self._feedforward_output_layers,
                 self._feedforward_layers,
                 self._feedforward_intermediate_layers,
                 self._layer_norm_output_layers, self._layer_norm_layers):
            layer_input = prev_output
            attention_output = attention(
                layer_input,
                encoder_self_attention_bias,
                key_padding_mask=encoder_padding_mask)
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
