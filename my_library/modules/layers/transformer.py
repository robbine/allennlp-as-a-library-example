import torch
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn import util

from my_library.modules.layers.common_attention import embedding_to_padding, attention_bias_ignore_padding, \
	attention_bias_proximal, add_timing_signal_1d, comma_separated_string_to_integer_list, attention_bias_to_padding, \
	multihead_attention


def transformer_prepare_encoder(inputs, params, features=None):
	""" Prepare one shard of the model for the encoder
	Args:
	:param inputs: a Tensor 
	:return: 
		encoder_input: a Tensor, bottom of encoder stack
		encoder_self_attention_bias: a bias tensor for use in encoder self-attention
		encoder_decoder_attention_bias: a bias for use in encoder-decoder attention
	"""
	ishape_static = inputs.size()
	encoder_input = inputs
	if features and "inputs_segmentation" in features:
		raise NotImplementedError("not implemented yet")
	else:
		encoder_padding = embedding_to_padding(encoder_input)
		ignore_padding = attention_bias_ignore_padding(encoder_padding)
		encoder_self_attention_bias = ignore_padding
		encoder_decoder_attention_bias = ignore_padding
	if params.proximity_bias:
		encoder_self_attention_bias += attention_bias_proximal(torch.size(encoder_input, 1),
															   util.get_device_of(encoder_input))
	if params.get("use_target_space_embedding", True):
		raise NotImplementedError("not implemented yet")
	if params.pos == "timing":
		encoder_input = add_timing_signal_1d(encoder_input, util.get_device_of(encoder_input))
	elif params.pos == "emb":
		raise NotImplementedError("not implemented yet")
	if params.activation_dtype == 'float16':
		encoder_self_attention_bias = encoder_self_attention_bias.half()
		encoder_decoder_attention_bias = encoder_decoder_attention_bias.half()
	return (encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias)


def transformer_encoder(encoder_input, encoder_self_attention_bias, params, nonpadding=None, losses=None):
	"""A stack of transformer layers.
	  Args:
	    encoder_input: a Tensor
	    encoder_self_attention_bias: bias Tensor for self-attention
	       (see common_attention.attention_bias())
	    hparams: hyperparameters for model
	    name: a string
	    nonpadding: optional Tensor with shape [batch_size, encoder_length]
	      indicating what positions are not padding.  This must either be
	      passed in, which we do for "packed" datasets, or inferred from
	      encoder_self_attention_bias.  The knowledge about padding is used
	      for pad_remover(efficiency) and to mask out padding in convolutional
	      layers.
	    losses: optional list onto which to append extra training losses
	  Returns:
	    y: a Tensors
	"""
	x = encoder_input
	attention_dropout_broadcast_dims = (
		comma_separated_string_to_integer_list(getattr(params, "attention_dropout_broadcast_dims", "")))
	padding = attention_bias_to_padding(encoder_self_attention_bias)
	if nonpadding is not None:
		padding = 1.0 - nonpadding
	else:
		padding = attention_bias_to_padding(encoder_self_attention_bias)
		nonpadding = 1.0 - padding
	for layer in range(params.num_encoder_layers):
		y = multihead_attention(
			layer_preprocess
		)

@Seq2SeqEncoder.register('transformer')
class Transformer(Seq2SeqEncoder):
	def __init__(self, dropout=0):
		super(Transformer, self).__init__()
		if dropout > 0:
			self._dropout = torch.nn.Dropout(p=dropout)
		else:
			self._dropout = lambda x: x

	def encode(self, inputs, losses=None):
		"""
		Args:
		:param inputs: Transformer inputs [batch_size, input_length, hidden_size] 
		:param losses: optional list onto which to append extra training losses
		:return: 
			Tuple of:
				encoder_output: Encoder representation. 
					[batch_size, input_length, hidden_size]
				encoder_decoder_attention_bias: Bias and mask weigth for encoder decoder attention. 
					[batch_size, input_length]
		"""
		encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
			transformer_prepare_encoder(inputs)
		)
		encoder_input = self._dropout(encoder_input)
		encoder_output = transformer_encoder(encoder_input, self_attention_bias, losses)
		return encoder_output, encoder_decoder_attention_bias




	def get_output_dim(self) -> int:
		pass

	def get_input_dim(self) -> int:
		pass

	def is_bidirectional(self) -> bool:
		pass
