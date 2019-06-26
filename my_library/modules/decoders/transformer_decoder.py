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


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self,
                 use_fp16,
                 decoder_embed_dim,
                 decoder_ffn_embed_dim,
                 decoder_attention_heads,
                 attention_dropout,
                 dropout=0,
                 activation_dropout=0,
                 relu_dropout=0.1,
                 decoder_normalize_before=True,
                 no_encoder_attn=False):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = decoder_embed_dim
        self.self_attn = MultiHeadAttention(
            use_fp16=use_fp16,
            num_heads=decoder_attention_heads,
            input_size=self.embed_dim,
            memory_size=self.embed_dim,
            key_depth=self.embed_dim,
            value_depth=self.embed_dim,
            attention_dropout_prob=attention_dropout,
        )
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        if self.activation_dropout == 0:
            self.activation_dropout = relu_dropout
        self.normalize_before = decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiHeadAttention(
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

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            future_mask=None,
            prev_self_attn_state=None,
            prev_attn_state=None,
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
        if prev_self_attn_state is not None:
            if incremental_state is None:
                increment_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
            self.self_attn._set_input_buffer(increment_state, saved_state)
        x = self.self_attn(x, future_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(
                self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if increment_state is None:
                    increment_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
                self.encoder_attn._set_input_buffer(increment_state,
                                                    saved_state)
            x = self.encoder_attn(
                x, None, encoder_out, key_padding_mask=encoder_padding_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(
                self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = gelu(self.fc1(x))
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
            no_encoder_attn=False,
    ):
        super().__init__()
        self.dropout = dropout
        self._text_field_embedder = text_field_embedder
        input_embed_dim = text_field_embedder.get_output_dim()
        embed_dim = decoder_embed_dim
        self.output_embed_dim = decoder_output_dim

        self.max_target_positions = max_target_positions
        self.embed_scale = math.sqrt(embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(
                use_fp16,
                decoder_embed_dim,
                decoder_ffn_embed_dim,
                decoder_attention_heads,
                attention_dropout,
                no_encoder_attn=no_encoder_attn) for _ in range(decoder_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self._position_embedding = EmbeddingV2(use_fp16, max_target_positions,
                                               input_embed_dim)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        future_mask = torch.triu(
            common_attention.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return future_mask

    def forward(
            self,
            prev_output_tokens,
            encoder_out=None,
            encoder_padding_mask=None,
            decoder_padding_mask=None,
            positions=None,
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
        embedded_tokens = self._text_field_embedder(
            prev_output_tokens) * self.embed_scale
        if positions is not None:
            position_embedding_res = self._position_embedding(positions.long())

            position_embedding_res = position_embedding_res.float(
            ).masked_fill(decoder_padding_mask.unsqueeze(-1), 0)
            embedded_tokens += position_embedding_res
        x = self.extract_features(
            embedded_tokens,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask)
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
        x = embedded_tokens

        x = F.dropout(x, p=self.dropout, training=self.training)
        future_mask = self.buffered_future_mask(x)
        # decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                future_mask=future_mask,
            )
        x = self.layer_norm(x)
        return x

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        embedding_table = self._text_field_embedder.get_embedding_by_name(
            'tokens')
        return F.linear(features, embedding_table.data)
