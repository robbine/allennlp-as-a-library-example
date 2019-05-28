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


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self,
                 use_fp16,
                 decoder_embed_dim,
                 decoder_ffn_embed_dim,
                 decoder_attention_heads,
                 attention_dropout,
                 activation_fn='relu',
                 dropout=0,
                 activation_dropout=0,
                 relu_dropout=0.1,
                 decoder_normalize_before=True,
                 add_bias_kv=False,
                 add_zero_attn=False):
        super().__init__()
        self.embed_dim = decoder_embed_dim
        self.self_attn = MultiheadAttention(
            use_fp16=use_fp16,
            num_heads=decoder_attention_heads,
            input_size=self.embed_dim,
            memory_size=self.embed_dim,
            key_depth=self.embed_dim,
            value_depth=self.embed_dim,
            attention_dropout_prob=attention_dropout,
        )
        self.dropout = dropout
        self.activation_fn = Activation.by_name(activation_fn)
        self.activation_dropout = activation_dropout
        if self.activation_dropout == 0:
            self.activation_dropout = relu_dropout
        self.normalize_before = decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = MultiheadAttention(
            use_fp16=use_fp16,
            num_heads=decoder_attention_heads,
            input_size=self.embed_dim,
            memory_size=self.embed_dim,
            key_depth=self.embed_dim,
            value_depth=self.embed_dim,
            attention_dropout_prob=attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self._position_embedding = EmbeddingV2(
            use_fp16, max_position_embeddings, embedding_dim=input_size)

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        decoder_self_attention_bias = common_attention.attention_bias_lower_triangle(
            x.size(1))
        x = self.self_attn(x, decoder_self_attention_bias)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
        # encoder_decoder_attention_bias = common_attention.attention_bias_ignore_padding(
        # encoder_padding_mask)
        x = self.encoder_attn(
            x, None, encoder_out, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(
            self,
            use_fp16,
            text_field_embedder,
            decoder_layers,
            dropout,
            decoder_embed_dim,
            decoder_ffn_embed_dim,
            decoder_attention_heads,
            decoder_output_dim,
            max_target_positions,
            attention_dropout,
    ):
        super().__init__(dictionary)
        self.dropout = dropout
        self._text_field_embedder = text_field_embedder
        input_embed_dim = text_field_embedder.get_output_dim()
        embed_dim = decoder_embed_dim
        self.output_embed_dim = decoder_output_dim

        self.max_target_positions = max_target_positions

        self.project_in_dim = nn.Linear(
            input_embed_dim, embed_dim,
            bias=False) if embed_dim != input_embed_dim else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(use_fp16, decoder_embed_dim,
                                    decoder_ffn_embed_dim,
                                    decoder_attention_heads, attention_dropout)
            for _ in range(decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = nn.Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim else None

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            prev_output_tokens,
            encoder_out=None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        embedded_tokens = self._text_field_embedder(prev_output_tokens)
        encoder_padding_mask = common_attention.embedding_to_padding(
            embedded_tokens)
        x = self.extract_features(embedded_tokens, encoder_out,
                                  encoder_padding_mask)
        x = self.output_layer(x)
        return x

    def extract_features(self,
                         embedded_tokens,
                         encoder_out=None,
                         encoder_padding_mask=None,
                         **unused):
        """
        Similar to *forward* but only return features.
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        x = util.add_positional_features(embedded_tokens)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
            )

        if self.normalize:
            x = self.layer_norm(x)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        embedding_table = self._text_field_embedder.get_embedding_by_name(
            'tokens')
        return F.linear(features, embedding_table)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions,
                   self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(
                self, '_future_mask'
        ) or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)),
                1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(
                name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(
                        name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(
                            name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(
                state_dict.get('{}.version'.format(name), torch.Tensor(
                    [1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict
