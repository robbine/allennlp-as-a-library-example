import logging
import math

import numpy as np
import torch
import torch.nn as nn
from allennlp.nn import util

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
	return torch.unsqueeze(torch.unsqueeze(-torch.log(1 + torch.abs(diff)), 0), 0)


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
		math.log(float(max_timescale) / float(min_timescale)) /
		max(num_timescales - 1.0, 1.0)
	)
	inv_timescales = min_timescale * torch.exp(
		util.get_range_vector(num_timescales, device).float() * -log_timescale_increment)
	scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
	signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
	pad = nn.ConstantPad1d((0, channels % 2), 0)
	signal = pad(signal)
	signal = signal.view(1, length, channels)
	return signal


def add_timing_signal_1d(x, device, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
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
	signal = get_timing_signal_1d(length, channels, device, min_timescale, max_timescale, start_index)
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
	return torch.squeeze(torch.squeeze(((attention_bias < -1).float()), dim=1), dim=1)


def compute_qkv(use_fp16,
				query_antecedent,
				memory_antecedent,
				key_projection: nn.Linear,
				value_projection: nn.Linear,
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
	q = query_projection(query_antecedent.float())
	k = key_projection(memory_antecedent.float())
	v = value_projection(memory_antecedent.float())
	if use_fp16:
		q = q.half()
		k = k.half()
		v = v.half()
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
	per_head = x.float().view(batch, length, num_heads, int(channels / num_heads))
	return per_head.transpose(1, 2).contiguous()


def dot_product_attention(q,
						  k,
						  v,
						  bias=None,
						  dropout=None):
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
		logits += bias
	weights = torch.nn.functional.softmax(logits, dim=-1)
	if dropout is not None:
		weights = dropout(weights)
	return torch.matmul(weights, v)


def _generate_relative_positions_matrix(length, max_relative_position, device=-1):
	"""Generates matrix of relative positions between inputs."""
	range_vec = util.get_range_vector(length, device)
	range_mat = range_vec.repeat(length, 1)
	distance_mat = range_mat - range_mat.t()
	distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position,
									   max_relative_position)
	# Shift values to be >= 0. Each integer still uniquely identifies a relative
	# position difference.
	final_mat = distance_mat_clipped + max_relative_position
	return final_mat


def _generate_relative_positions_embeddings(length, depth, max_relative_position, embedding, device=-1):
	"""Generates tensor of size [length, length, depth]."""
	relative_positions_matrix = _generate_relative_positions_matrix(
		length, max_relative_position, device)
	embeddings = torch.index_select(embedding, 0, relative_positions_matrix.view(-1))
	# embeddings = torch.gather(embedding, 1, relative_positions_matrix)
	return embeddings.view(length, length, depth)


def _relative_attention_inner(x, y, z, transpose):
	"""Relative position-aware dot-product attention inner calculation.
	This batches matrix multiply calculations to avoid unnecessary broadcasting.
	Args:
	  x: Tensor with shape [batch_size, heads, length, length or depth].
	  y: Tensor with shape [batch_size, heads, length, depth].
	  z: Tensor with shape [length, length, depth].
	  transpose: Whether to transpose inner matrices of y and z. Should be true if
		  last dimension of x is depth, not length.
	Returns:
	  A Tensor with shape [batch_size, heads, length, length or depth].
	"""
	batch_size, heads, length, depth = x.size()

	# xy_matmul is [batch_size, heads, length, length or depth]
	if transpose:
		xy_matmul = torch.matmul(x, y.transpose(2, 3))
	else:
		xy_matmul = torch.matmul(x, y)
	# x_t is [length, batch_size, heads, length or depth]
	x_t = x.permute(2, 0, 1, 3)
	# x_t_r is [length, batch_size * heads, length or depth]
	x_t_r = x_t.view(length, heads * batch_size, -1)
	# x_tz_matmul is [length, batch_size * heads, length or depth]
	if transpose:
		z = z.transpose(-1, -2)
	x_tz_matmul = torch.matmul(x_t_r, z)
	# x_tz_matmul_r is [length, batch_size, heads, length or depth]
	x_tz_matmul_r = x_tz_matmul.view(length, batch_size, heads, -1)
	# x_tz_matmul_r_t is [batch_size, heads, length, length or depth]
	x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
	return xy_matmul + x_tz_matmul_r_t


def dot_product_attention_relative(q,
								   k,
								   v,
								   bias,
								   max_relative_position,
								   dropout,
								   key_embedding,
								   value_embedding):
	"""Calculate relative position-aware dot-product self-attention.
	  The attention calculation is augmented with learned representations for the
	  relative position between each element in q and each element in k and v.
	  Args:
	    q: a Tensor with shape [batch, heads, length, depth].
	    k: a Tensor with shape [batch, heads, length, depth].
	    v: a Tensor with shape [batch, heads, length, depth].
	    bias: bias Tensor.
	    max_relative_position: an integer specifying the maximum distance between
	        inputs that unique position embeddings should be learned for.
	    dropout_rate: a floating point number.
	  Returns:
	    A Tensor.
	  Raises:
	    ValueError: if max_relative_position is not > 0.
	"""
	if not max_relative_position:
		raise ValueError("Max relative position (%s) should be > 0 when using "
						 "relative self attention." % (max_relative_position))
	assert q.size() == k.size()
	assert q.size() == v.size()
	batch, num_heads, length, depth = q.size()
	device = util.get_device_of(q)
	relations_keys = _generate_relative_positions_embeddings(length, depth, max_relative_position,
															 key_embedding, device)
	relations_values = _generate_relative_positions_embeddings(length, depth, max_relative_position,
															   value_embedding, device)
	# Compute self attention considering the relative position embeddings.
	logits = _relative_attention_inner(q, k, relations_keys, True)
	if bias is not None:
		logits += bias
	weights = torch.nn.functional.softmax(logits, dim=-1)
	weights = dropout(weights)
	return _relative_attention_inner(weights, v, relations_values, False)


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
	"""Matrix band part of ones."""
	# Needed info is constant, so we construct in numpy
	if num_lower < 0:
		num_lower = rows - 1
	if num_upper < 0:
		num_upper = cols - 1
	lower_mask = np.tri(cols, rows, num_lower).T
	upper_mask = np.tri(rows, cols, num_upper)
	band = np.ones((rows, cols)) * lower_mask * upper_mask
	if out_shape:
		band = band.reshape(out_shape)
	band = torch.from_numpy(band).float()
	return band


def attention_bias_local(length, max_backward, max_forward):
	"""Create an bias tensor to be added to attention logits.
	A position may attend to positions at most max_distance from it,
	forward and backwards.
	This does not actually save any computation.
	Args:
	  length: int
	  max_backward: int, maximum distance backward to attend. Negative values
	    indicate unlimited.
	  max_forward: int, maximum distance forward to attend. Negative values
	    indicate unlimited.
	Returns:
	  a `Tensor` with shape [1, 1, length, length].
	"""
	band = ones_matrix_band_part(
		length,
		length,
		max_backward,
		max_forward,
		out_shape=[1, 1, length, length])
	return -1e9 * (1.0 - band)


def attention_bias_lower_triangle(length):
	"""Create an bias tensor to be added to attention logits.
	Allows a query to attend to all positions up to and including its own.
	Args:
	 length: a Scalar.
	Returns:
	  a `Tensor` with shape [1, 1, length, length].
	"""
	return attention_bias_local(length, -1, 0)


def _make_local_block(x, depth, batch, heads, num_blocks, block_length):
	"""Helper function to create a local version of the keys or values for 1d."""
	prev_block = x.narrow(2, 0, num_blocks - 1)
	cur_block = x.narrow(2, 1, num_blocks - 1)
	local_block = torch.cat((prev_block, cur_block), dim=3)
	return local_block.view(batch, heads, num_blocks - 1, block_length * 2, depth)


def masked_local_attention_1d(q,
							  k,
							  v,
							  block_length=64,
							  dropout=None):
	"""Attention to the source position and a neighborhood to the left of it.
	The sequence is divided into blocks of length block_length. Attention for a
	given query position can only see memory positions less than or equal to the
	query position, in the corresponding block and the previous block.
	Args:
	  q: a Tensor with shape [batch, heads, length, depth_k]
	  k: a Tensor with shape [batch, heads, length, depth_k]
	  v: a Tensor with shape [batch, heads, length, depth_v]
	  block_length: an integer
	  make_image_summary: a boolean, whether to make an attention image summary.
	  dropout_rate: Dropout rate for attention dropout
	  name: an optional string
	Returns:
	  a Tensor of shape [batch, heads, length, depth_v]
	"""
	batch, heads, length, depth_k = q.size()
	depth_v = v.size(3)
	if isinstance(block_length, torch.Tensor):
		block_length = int(block_length.item())
	block_length = length if length < block_length * 2 else block_length
	# Pad query, key, value to ensure multiple of block length.
	original_length = length
	padding_size = (-length) % block_length
	length += padding_size
	pad = nn.ConstantPad2d((0, 0, 0, padding_size), 0)
	q = pad(q)
	k = pad(k)
	v = pad(v)
	num_blocks = length // block_length
	first_q = q.narrow(2, 0, block_length)
	first_k = k.narrow(2, 0, block_length)
	first_v = v.narrow(2, 0, block_length)
	first_output = dot_product_attention(
		first_q,
		first_k,
		first_v,
		attention_bias_lower_triangle(block_length),
		dropout
	)
	# Compute attention for all subsequent query blocks.
	q = q.view(batch, heads, num_blocks, block_length, depth_k)
	k = k.view(batch, heads, num_blocks, block_length, depth_k)
	v = v.view(batch, heads, num_blocks, block_length, depth_v)

	local_k = _make_local_block(k, depth_k, batch, heads, num_blocks,
								block_length)
	local_v = _make_local_block(v, depth_v, batch, heads, num_blocks,
								block_length)
	tail_q = q.narrow(2, 1, num_blocks - 1)
	tail_q = tail_q.view(batch, heads, num_blocks - 1, block_length, depth_k)
	local_length = local_k.size(3)
	good_part = ones_matrix_band_part(
		block_length,
		local_length,
		-1,
		block_length,
		out_shape=[1, 1, 1, block_length, local_length])
	bias = (1.0 - good_part) * -1e9
	tail_output = dot_product_attention(
		tail_q,
		local_k,
		local_v,
		bias,
		dropout)
	tail_output = tail_output.view(batch, heads, (num_blocks - 1) * block_length, depth_v)
	output = torch.cat([first_output, tail_output], dim=2)
	# Remove the padding if introduced.
	output = output.narrow(2, 0, original_length)
	output = output.view(batch, heads, original_length, depth_v)
	return output


def get_relative_embeddings_left(relative_embeddings, max_relative_position, length, depth,
								 num_heads, heads_share_relative_embedding):
	"""Instantiate or retrieve relative embeddings, sliced according to length.
	Use for masked case where the relative attention is only looking left.
	Args:
	  max_relative_position: an Integer for the number of entries in the relative
		embedding, which corresponds to the max relative distance that is
		considered.
	  length: an Integer, specifies the length of the input sequence for which
		this relative embedding is retrieved for.
	  depth: an Integer, specifies the depth for relative embeddings.
	  num_heads: an Integer, specifies the number of heads.
	  heads_share_relative_embedding: a Boolean specifying if the relative
		embedding is shared across heads.
	  name: a string giving the name of the embedding variables.
	Returns:
	  a Tensor with shape [length, depth]
	"""
	# Pad first before slice to avoid using tf.cond.
	pad_length = max(length - max_relative_position, 0)
	start_slice_position = max(max_relative_position - length, 0)
	pad = nn.ConstantPad2d((0, 0, pad_length, 0), 0)
	padded_relative_embeddings = pad(relative_embeddings)
	if heads_share_relative_embedding:
		used_relative_embeddings = padded_relative_embeddings.narrow(0, start_slice_position, length)
	else:
		used_relative_embeddings = padded_relative_embeddings.narrow(1, start_slice_position, length)
	return used_relative_embeddings


def matmul_with_relative_keys(x, y, heads_share_relative_embedding):
	if heads_share_relative_embedding:
		ret = torch.einsum("bhld,md->bhlm", (x, y))
	else:
		ret = torch.einsum("bhld,hmd->bhlm", (x, y))
	return ret


def matmul_with_relative_values(x, y, heads_share_relative_embedding):
	if heads_share_relative_embedding:
		ret = torch.einsum("bhlm,md->bhld", (x, y))
	else:
		ret = torch.einsum("bhlm,hmd->bhld", (x, y))
	return ret


def _relative_position_to_absolute_position_masked(x):
	"""Helper to dot_product_self_attention_relative_v2.
	Rearrange an attention logits or weights Tensor.
	The dimensions of the input represent:
	[batch, heads, query_position, memory_position - query_position + length - 1]
	The dimensions of the output represent:
	[batch, heads, query_position, memory_position]
	Only works with masked_attention.  Undefined behavior for regions of the
	input where memory_position > query_position.
	Args:
	  x: a Tensor with shape [batch, heads, length, length]
	Returns:
	  a Tensor with shape [batch, heads, length, length]
	"""
	batch, heads, length, _ = x.size()
	pad = nn.ConstantPad1d((1, 0), 0)
	x = pad(x)
	x = x.view(batch, heads, 1 + length, length)
	x = x.narrow(2, 1, length)
	return x


def _relative_position_to_absolute_position_unmasked(x):
	"""Converts tensor from relative to aboslute indexing for local attention.
	Args:
	  x: a Tensor of shape [batch (or batch*num_blocks), heads,
							length, 2 * length - 1]
	Returns:
	  A Tensor of shape [batch (or batch*num_blocks), heads, length, length-1]
	"""
	x_shape = x.size()
	batch = x_shape[0]
	heads = x_shape[1]
	length = x_shape[2]
	# Concat columns of pad to shift from relative to absolute indexing.
	col_pad = torch.zeros(batch, heads, length, 1)
	x = torch.cat((x, col_pad), dim=3)

	# Concat extra elements so to add up to shape (len+1, 2*len-1).
	flat_x = x.view(batch, heads, length * 2 * length)
	flat_pad = torch.zeros(batch, heads, length - 1)
	flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

	# Reshape and slice out the padded elements.
	final_x = flat_x_padded.view(batch, heads, length + 1, 2 * length - 1)
	final_x = final_x[:, :, :length, length - 1:]
	return final_x


def _absolute_position_to_relative_position_unmasked(x):
	"""Helper function for dot_product_unmasked_self_attention_relative_v2.
	  Rearrange an attention logits or weights Tensor.
	  The dimensions of the input represent:
	  [batch, heads, query_position, memory_position]
	  The dimensions of the output represent:
	  [batch, heads, query_position, memory_position - query_position + length - 1]
	  Only works with unmasked_attention.
	  Args:
	    x: a Tensor with shape [batch, heads, length, length]
	  Returns:
	    a Tensor with shape [batch, heads, length, 2*length-1]
	"""
	batch, heads, length, _ = x.size()
	# padd along column
	pad = nn.ConstantPad1d((0, length - 1), 0)
	x = pad(x)
	x_flat = x.view(batch, heads, length ** 2 + length * (length - 1))
	# add 0's in the beginning that will skew the elements after reshape
	pad = nn.ConstantPad1d((length, 0), 0)
	x_flat = pad(x_flat)
	x = x_flat.view(batch, heads, length, 2 * length)
	x = x.narrow(3, 1, 2 * length - 1)
	return x


def _absolute_position_to_relative_position_masked(x):
	"""Helper to dot_product_self_attention_relative_v2.
	Rearrange an attention logits or weights Tensor.
	The dimensions of the input represent:
	[batch, heads, query_position, memory_position]
	The dimensions of the output represent:
	[batch, heads, query_position, memory_position - query_position + length - 1]
	Only works with masked_attention.  Undefined behavior for regions of the
	input where memory_position > query_position.
	Args:
	  x: a Tensor with shape [batch, heads, length, length]
	Returns:
	  a Tensor with shape [batch, heads, length, length]
	"""
	batch, heads, length, _ = x.size()
	pad = nn.ConstantPad2d((0, 0, 1, 0), 0)
	x = pad(x)
	x = x.view(batch, heads, length, length + 1)
	x = x.narrow(3, 1, length)
	return x


def masked_relative_local_attention_1d(q,
									   k,
									   v,
									   block_length=64,
									   relative_key_embeddings=None,
									   relative_value_embeddings=None,
									   dropout=None,
									   heads_share_relative_embedding=False,
									   add_relative_to_values=False
									   ):
	"""Masked local 1d attention with relative positions.
	  The sequence is divided into blocks of length block_size.
	  Attention for a given query position can only see memory positions
	  less than or equal to the query position, in the corresponding block
	  and the previous block.
	  If mask_right is True, then a target position cannot see greater source
	  positions.
	  Args:
		q: a Tensor with shape [batch, heads, length, depth_k]
		k: a Tensor with shape [batch, heads, length, depth_k]
		v: a Tensor with shape [batch, heads, length, depth_v]
		block_length: an integer
		heads_share_relative_embedding: a boolean for sharing relative embeddings.
		add_relative_to_values: a boolean for whether to add relative component to
			values.
	  Returns:
		a Tensor of shape [batch, heads, length, depth_v]
	  Raises:
		ValueError: when the name for the variable scope is not passed.
	"""
	default_block_length = block_length
	batch, heads, length, depth_k = q.size()
	# If (length < 2 * block_length), then we use only one block.
	block_length = length if length < block_length * 2 else block_length

	depth_k = k.size(3)
	depth_v = v.size(3)
	original_length = length
	padding_size = (-length) % block_length
	length += padding_size
	pad = nn.ConstantPad2d((0, 0, 0, padding_size), 0)
	q = pad(q)
	k = pad(k)
	v = pad(v)

	num_blocks = length // block_length
	# compute attention for the first query block.
	first_q = q.narrow(2, 0, block_length)
	first_k = k.narrow(2, 0, block_length)
	first_v = v.narrow(2, 0, block_length)
	# Relative embeddings will be used later as well.
	# Needs to be known at static shape inference time, hence cannot be
	# 2 * block_length.
	rel_embed_length = 4 * default_block_length
	# We only multiply with the needed embeddings as we slice them out.
	first_rel_embeddings = get_relative_embeddings_left(relative_key_embeddings,
														rel_embed_length, block_length, depth_k, heads,
														heads_share_relative_embedding)
	first_rel_logits = matmul_with_relative_keys(
		first_q, first_rel_embeddings, heads_share_relative_embedding)
	first_logits = torch.matmul(first_q, first_k.transpose(-1, -2))
	first_logits += (
		_relative_position_to_absolute_position_masked(first_rel_logits))
	# adding a mask
	first_logits += attention_bias_lower_triangle(block_length)
	first_att = torch.nn.functional.softmax(first_logits, dim=-1)
	# dropping out the attention links for each of the heads
	first_att = dropout(first_att)
	first_output = torch.matmul(first_att, first_v)

	# compute attention for all subsequent query blocks.
	q = q.view(batch, heads, num_blocks, block_length, depth_k)
	k = k.view(batch, heads, num_blocks, block_length, depth_k)
	v = v.view(batch, heads, num_blocks, block_length, depth_v)
	local_k = _make_local_block(k, depth_k, batch, heads, num_blocks,
								block_length)
	local_v = _make_local_block(v, depth_v, batch, heads, num_blocks,
								block_length)
	tail_q = q.narrow(2, 1, num_blocks - 1)
	tail_q = tail_q.view(batch, heads, num_blocks - 1, block_length, depth_k)
	local_length = local_k.size(3)

	# collapsing num blocks and batch size so that we can reuse
	# functions
	def _reshape_for_relative(x):
		x_shape = x.size()
		# [batch, num_blocks, heads, length, depth]
		x = x.transpose(2, 1).contiguous()
		x = x.view(batch * x_shape[2], heads, x_shape[3], x_shape[4])
		return x

	rel_tail_q = _reshape_for_relative(tail_q)
	rel_k = _reshape_for_relative(local_k)
	rel_v = _reshape_for_relative(local_v)
	rel_embeddings = get_relative_embeddings_left(relative_key_embeddings,
												  rel_embed_length, 2 * block_length, depth_k, heads,
												  heads_share_relative_embedding)
	rel_logits = matmul_with_relative_keys(
		rel_tail_q, rel_embeddings, heads_share_relative_embedding)
	b, h, l, m = rel_logits.size()
	# Computing relative logits separately for the masked and unmasked parts
	# because the reshaping logic is different for both
	masked_rel_logits = rel_logits.narrow(3, block_length, m - block_length)
	masked_rel_logits = _relative_position_to_absolute_position_masked(
		masked_rel_logits)
	unmasked_rel_logits = rel_logits.narrow(3, 0, 2 * block_length - 1)
	unmasked_rel_logits = _relative_position_to_absolute_position_unmasked(
		unmasked_rel_logits)
	all_rel_logits = torch.cat((unmasked_rel_logits, masked_rel_logits), dim=3)
	all_logits = (
		torch.matmul(rel_tail_q, rel_k.transpose(-1, -2).contiguous()) + all_rel_logits)
	# make sure source_pos <= target_pos
	good_part = ones_matrix_band_part(block_length,
									  local_length,
									  -1, block_length)
	mask = (1.0 - good_part) * -1e9
	all_logits += mask.view(1, 1, block_length, local_length)
	weights = torch.nn.functional.softmax(all_logits, dim=-1)
	# [batch (* num_blocks), heads, query_length (=block_length),
	# key_length (=2*block_length)]
	weights = dropout(weights)

	output = torch.matmul(weights, rel_v)
	if add_relative_to_values:
		# Adds the contribution of the weighted relative embeddings to the values.
		weights_for_unmasked, weights_for_masked = (
			torch.split(weights, local_length // 2, dim=3))
		rel_weights_unmasked = _absolute_position_to_relative_position_unmasked(
			weights_for_unmasked)
		rel_weights_masked = _absolute_position_to_relative_position_masked(
			weights_for_masked)

		value_rel_embeddings_unmasked = get_relative_embeddings_left(relative_value_embeddings,
																	 rel_embed_length, 2 * block_length, depth_v,
																	 heads, heads_share_relative_embedding)
		# The unmasked part starts with index -1 as opposed 0 has take uptil last.
		if heads_share_relative_embedding:
			value_rel_embeddings_unmasked = value_rel_embeddings_unmasked[:-1, :]
		else:
			value_rel_embeddings_unmasked = value_rel_embeddings_unmasked[:, :-1, :]
		value_rel_embeddings_masked = get_relative_embeddings_left(relative_value_embeddings,
																   rel_embed_length, block_length, depth_v,
																   heads, heads_share_relative_embedding)

		# [batch (*num_blocks), heads, query length, key length]
		rel_weights = torch.cat(
			[rel_weights_unmasked, rel_weights_masked], dim=3)
		if heads_share_relative_embedding:
			value_rel_embeddings_concat_axis = 0
		else:
			value_rel_embeddings_concat_axis = 1
		value_rel_embeddings = torch.cat(
			[value_rel_embeddings_unmasked, value_rel_embeddings_masked],
			dim=value_rel_embeddings_concat_axis)
		output_rel = matmul_with_relative_values(
			rel_weights, value_rel_embeddings, heads_share_relative_embedding)
		output += output_rel

	# bring to [batch, heads, num_blocks-1, block_length, depth]
	output = output.view(batch, num_blocks - 1, heads, block_length, depth_v)
	output = output.transpose(2, 1).contiguous()

	output = output.view(batch, heads, (num_blocks - 1) * block_length, depth_v)
	output = torch.cat((first_output, output), dim=2)
	output = output.narrow(2, 0, original_length)
	output = output.view(batch, heads, original_length, depth_v)
	return output


def masked_within_block_local_attention_1d(q, k, v, block_length=64):
	"""Attention to the source and a neighborhood to the left within a block.
	The sequence is divided into blocks of length block_length. Attention for a
	given query position can only see memory positions less than or equal to the
	query position in the corresponding block.
	Args:
	  q: a Tensor with shape [batch, heads, length, depth_k]
	  k: a Tensor with shape [batch, heads, length, depth_k]
	  v: a Tensor with shape [batch, heads, length, depth_v]
	  block_length: an integer
	  name: an optional string
	Returns:
	  a Tensor of shape [batch, heads, length, depth_v]
	"""
	batch, heads, length, depth_k = q.size()
	depth_v = v.size(-1)
	# Pad query, key, value to ensure multiple of block length.
	original_length = length
	padding_size = (-length) % block_length
	length += padding_size
	pad = nn.ConstantPad2d((0, 0, 0, padding_size), 0)
	q = pad(q)
	k = pad(k)
	v = pad(v)

	# Compute attention for all subsequent query blocks.
	num_blocks = length // block_length
	q = q.view(batch, heads, num_blocks, block_length, depth_k)
	k = k.view(batch, heads, num_blocks, block_length, depth_k)
	v = v.view(batch, heads, num_blocks, block_length, depth_v)
	# [batch, heads, num_blocks, block_length, block_length]
	attention = torch.matmul(q, k.transpose(-1, -2))
	attention += attention_bias_lower_triangle(block_length).view(1, 1, 1, block_length, block_length)
	attention = torch.nn.functional.softmax(attention, dim=-1)
	# [batch, heads, num_blocks, block_length, depth_v]
	output = torch.matmul(attention, v)
	output = output.view(batch, heads, -1, depth_v)

	# Remove the padding if introduced.
	output = output.narrow(2, 0, original_length)
	return output


def get_relative_embeddings_left_right(relative_embeddings, max_relative_position, length, depth,
									   num_heads,
									   heads_share_relative_embedding):
	"""Instantiate or retrieve relative embeddings, sliced according to length.
	Use for unmasked case where the relative attention looks both left and right.
	Args:
	  max_relative_position: an Integer for the number of entries in the relative
		embedding, which corresponds to the max relative distance that is
		considered.
	  length: an Integer, specifies the length of the input sequence for which
		this relative embedding is retrieved for.
	  depth: an Integer, specifies the depth for relative embeddings.
	  num_heads: an Integer, specifies the number of heads.
	  heads_share_relative_embedding: a Boolean specifying if the relative
		embedding is shared across heads.
	  name: a string giving the name of the embedding variables.
	Returns:
	  a Tensor with shape [length, depth]
	"""
	# Pad first before slice to avoid using tf.cond.
	pad_length = max(length - max_relative_position, 0)
	slice_start_position = max(max_relative_position - length, 0)
	pad = nn.ConstantPad2d((0, 0, pad_length, pad_length), 0)
	padded_relative_embeddings = pad(relative_embeddings)
	if heads_share_relative_embedding:
		used_relative_embeddings = padded_relative_embeddings.narrow(0, slice_start_position, 2 * length - 1)
	else:
		used_relative_embeddings = padded_relative_embeddings.narrow(1, slice_start_position, 2 * length - 1)
	return used_relative_embeddings


def dot_product_unmasked_self_attention_relative_v2(
		q, k, v, bias, relative_key_embeddings=None, relative_value_embeddings=None, max_relative_position=None,
		dropout=None,
		heads_share_relative_embedding=False,
		add_relative_to_values=False):
	"""Calculate relative position-aware dot-product self-attention.
	The attention calculation is augmented with learned representations for the
	relative position between each element in q and each element in k and v.
	Args:
	  q: a Tensor with shape [batch, heads, length, depth].
	  k: a Tensor with shape [batch, heads, length, depth].
	  v: a Tensor with shape [batch, heads, length, depth].
	  bias: bias Tensor.
	  max_relative_position: an integer the max relative embedding considered.
		Changing this invalidates checkpoints.
	  dropout_rate: a floating point number.
	  image_shapes: optional tuple of integer scalars.
	  name: an optional string.
	  make_image_summary: Whether to make an attention image summary.
	  heads_share_relative_embedding: a boolean indicating wheather to share
		relative embeddings between attention heads.
	  add_relative_to_values: a boolean for whether to add relative component to
		values.
	Returns:
	  A Tensor.
	Raises:
	  ValueError: if max_relative_position is not > 0.
	"""
	if not max_relative_position:
		raise ValueError("Max relative position (%s) should be > 0 when using "
						 "relative self attention." % (max_relative_position))

	# This calculation only works for self attention.
	# q, k and v must therefore have the same shape.
	assert q.size() == k.size()
	assert q.size() == v.size()

	# [batch, num_heads, query_length, memory_length]
	logits = torch.matmul(q, k.transpose(-1, -2))

	length = q.size(2)
	k_shape = k.size()
	num_heads = k_shape[1]
	depth_k = k_shape[-1]

	key_relative_embeddings = get_relative_embeddings_left_right(relative_key_embeddings,
																 max_relative_position, length, depth_k, num_heads,
																 heads_share_relative_embedding)
	unmasked_rel_logits = matmul_with_relative_keys(
		q, key_relative_embeddings, heads_share_relative_embedding)
	unmasked_rel_logits = _relative_position_to_absolute_position_unmasked(
		unmasked_rel_logits)
	logits += unmasked_rel_logits

	if bias is not None:
		logits += bias
	weights = torch.nn.functional.softmax(logits, dim=-1)
	# dropping out the attention links for each of the heads
	weights = dropout(weights)
	ret = torch.matmul(weights, v)
	if add_relative_to_values:
		# Adds the contribution of the weighted relative embeddings to the values.
		# [batch, num_heads, query_length, 2*memory_length-1]
		relative_weights = _absolute_position_to_relative_position_unmasked(
			weights)
		depth_v = v.size(3)
		value_relative_embeddings = get_relative_embeddings_left_right(relative_value_embeddings,
																	   max_relative_position, length, depth_v,
																	   num_heads,
																	   heads_share_relative_embedding)
		ret += matmul_with_relative_values(
			relative_weights, value_relative_embeddings,
			heads_share_relative_embedding)
	return ret


def combine_heads(x):
	"""Inverse of split_heads.
	Args:
	  x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
	Returns:
	  a Tensor with shape [batch, length, channels]
	"""
	batch, num_heads, length, depth = x.size()
	x = x.transpose(1, 2).contiguous()
	outputs = x.view(batch, length, -1)
	return outputs


def multihead_attention(use_fp16,
						query_antecedent,
						memory_antecedent,
						bias,
						total_key_depth,
						total_value_depth,
						num_heads,
						dropout,
						key_embedding=None,
						value_embedding=None,
						relative_key_embeddings=None,
						relative_value_embeddings=None,
						key_projection=None,
						value_projection=None,
						query_projection=None,
						attention_type="dot_product",
						max_relative_position=None,
						heads_share_relative_embedding=False,
						add_relative_to_values=False,
						block_length=128,
						block_width=128):
	"""Multihead scaled-dot-product attention with input/output transformations.
	Args:
	  query_antecedent: a Tensor with shape [batch, length_q, channels]
	  memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
	  bias: bias Tensor (see attention_bias())
	  total_key_depth: an integer
	  total_value_depth: an integer
	  num_heads: an integer dividing total_key_depth and total_value_depth
	  dropout_rate: a floating point number
	  attention_type: a string, either "dot_product", "dot_product_relative",
					  "local_mask_right", "local_unmasked", "masked_dilated_1d",
					  "unmasked_dilated_1d", graph, or any attention function
					  with the signature (query, key, value, **kwargs)
	  max_relative_position: Maximum distance between inputs to generate
							 unique relation embeddings for. Only relevant
							 when using "dot_product_relative" attention.
	  heads_share_relative_embedding: boolean to share relative embeddings
	  add_relative_to_values: a boolean for whether to add relative component to
							  values.
	  block_length: an integer - relevant for "local_mask_right"
	  block_width: an integer - relevant for "local_unmasked"
	  q_filter_width: An integer specifying how wide you want the query to be.
	  kv_filter_width: An integer specifying how wide you want the keys and values
					   to be.
	  q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
				 kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
				 no padding.
	  cache: dict containing Tensors which are the results of previous
			 attentions, used for fast decoding. Expects the dict to contrain two
			 keys ('k' and 'v'), for the initial call the values for these keys
			 should be empty Tensors of the appropriate shape.
				 'k' [batch_size, 0, key_channels]
				 'v' [batch_size, 0, value_channels]
	  gap_size: Integer option for dilated attention to indicate spacing between
				memory blocks.
	  num_memory_blocks: Integer option to indicate how many memory blocks to look
						 at.
	  name: an optional string.
	  save_weights_to: an optional dictionary to capture attention weights
		for vizualization; the weights tensor will be appended there under
		a string key created from the variable scope (including name).
	  make_image_summary: Whether to make an attention image summary.
	  dropout_broadcast_dims:  an optional list of integers less than 4
		specifying in which dimensions to broadcast the dropout decisions.
		saves memory.
	  vars_3d: use 3-dimensional variables for input/output transformations
	  **kwargs (dict): Parameters for the attention function
	Caching:
	  WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
	  the caching assumes that the bias contains future masking.
	  The caching works by saving all the previous key and value values so that
	  you are able to send just the last query location to this attention
	  function. I.e. if the cache dict is provided it assumes the query is of the
	  shape [batch_size, 1, hidden_dim] rather than the full memory.
	Returns:
	  The result of the attention transformation. The output shape is
		  [batch_size, length_q, hidden_dim]
	  unless the cache dict is provided in which case only the last memory
	  position is calculated and the output shape is [batch_size, 1, hidden_dim]
	  Optionally returns an additional loss parameters (ex: load balance loss for
	  the experts) returned by the attention_type function.
	Raises:
	  ValueError: if the key depth or value depth are not divisible by the
		number of attention heads.
	"""
	if total_key_depth % num_heads != 0:
		raise ValueError("Key depth (%d) must be divisible by the number of "
						 "attention heads (%d)." % (total_key_depth, num_heads))
	if total_value_depth % num_heads != 0:
		raise ValueError("Value depth (%d) must be divisible by the number of "
						 "attention heads (%d)." % (total_value_depth, num_heads))
	q, k, v = compute_qkv(use_fp16, query_antecedent, memory_antecedent, key_projection, value_projection, query_projection)

	q = split_heads(q, num_heads)
	k = split_heads(k, num_heads)
	v = split_heads(v, num_heads)

	key_depth_per_head = total_key_depth // num_heads
	q *= key_depth_per_head ** -0.5

	if attention_type == "dot_product":
		x = dot_product_attention(q, k, v, bias, dropout)
	elif attention_type == "dot_product_relative":
		x = dot_product_attention_relative(
			q,
			k,
			v,
			bias,
			max_relative_position,
			dropout,
			key_embedding=key_embedding,
			value_embedding=value_embedding
		)
	elif attention_type == "dot_product_unmasked_relative_v2":
		x = dot_product_unmasked_self_attention_relative_v2(
			q,
			k,
			v,
			bias,
			relative_key_embeddings=relative_key_embeddings,
			relative_value_embeddings=relative_value_embeddings,
			max_relative_position=max_relative_position,
			dropout=dropout,
			heads_share_relative_embedding=heads_share_relative_embedding,
			add_relative_to_values=add_relative_to_values)
	elif attention_type == "dot_product_relative_v2":
		raise NotImplementedError("not implemented yet")
	elif attention_type == "local_within_block_mask_right":
		x = masked_within_block_local_attention_1d(
			q, k, v, block_length=block_length)
	elif attention_type == "local_relative_mask_right":
		x = masked_relative_local_attention_1d(
			q,
			k,
			v,
			block_length=block_length,
			relative_key_embeddings=relative_key_embeddings,
			relative_value_embeddings=relative_value_embeddings,
			dropout=dropout,
			heads_share_relative_embedding=heads_share_relative_embedding,
			add_relative_to_values=add_relative_to_values)
	elif attention_type == "local_mask_right":
		x = masked_local_attention_1d(
			q,
			k,
			v,
			block_length=block_length,
			dropout=dropout)
	elif attention_type == "local_unmasked":
		raise NotImplementedError("not implemented yet")
	elif attention_type == "masked_dilated_1d":
		raise NotImplementedError("not implemented yet")
	else:
		assert attention_type == "unmasked_dilated_1d"
		raise NotImplementedError("not implemented yet")
	x = combine_heads(x)
	return x


def embedding_postprocessor(input_tensor,
							input_mask_tensor,
							use_fp16,
							token_type_ids=None,
							use_token_type=False,
							token_type_embedding=None,
							use_position_embeddings=True,
							position_embedding=None,
							norm_layer=None,
							dropout=None):
	"""Performs various post-processing on a word embedding tensor.
  
	Args:
	  input_tensor: float Tensor of shape [batch_size, seq_length,
		embedding_size].
	  use_token_type: bool. Whether to add embeddings for `token_type_ids`.
	  token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
		Must be specified if `use_token_type` is True.
	  token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
	  token_type_embedding_name: string. The name of the embedding table variable
		for token type ids.
	  use_position_embeddings: bool. Whether to add position embeddings for the
		position of each token in the sequence.
	  position_embedding_name: string. The name of the embedding table variable
		for positional embeddings.
	  initializer_range: float. Range of the weight initialization.
	  max_position_embeddings: int. Maximum sequence length that might ever be
		used with this model. This can be longer than the sequence length of
		input_tensor, but cannot be shorter.
	  dropout_prob: float. Dropout probability applied to the final output tensor.
  
	Returns:
	  float tensor with same shape as `input_tensor`.
  
	Raises:
	  ValueError: One of the tensor shapes or input values is invalid.
	"""
	input_shape = input_tensor.size()
	batch_size = input_shape[0]
	seq_length = input_shape[1]
	width = input_shape[2]

	output = input_tensor

	if use_token_type:
		if token_type_ids is None:
			raise ValueError("`token_type_ids` must be specified if"
							 "`use_token_type` is True.")
		# This vocab will be small so we always do one-hot here, since it is always
		# faster for a small vocabulary.
		token_type_embedding_res = token_type_embedding(token_type_ids)
		if use_fp16:
			output = torch.add(output.float(), token_type_embedding_res.float()).half()
		else:
			output += token_type_embedding_res

	if use_position_embeddings:
		range_tensor = util.get_range_vector(seq_length, util.get_device_of(input_mask_tensor))
		range_tensor = range_tensor.view(1, -1).long()
		position_embedding_res = position_embedding(input_mask_tensor * range_tensor)
		if use_fp16:
			output = torch.add(output.float(), position_embedding_res.float()).half()
		else:
			output += position_embedding_res

	output = layer_norm_and_dropout(use_fp16, output, norm_layer, dropout)
	return output


def layer_norm(use_fp16, input_tensor, norm_layer):
	"""Run layer normalization on the last dimension of the tensor."""
	if use_fp16:
		return norm_layer(input_tensor.float().cuda()).half()
	else:
		return norm_layer(input_tensor)


def layer_norm_and_dropout(use_fp16, input_tensor, norm_layer, dropout):
	"""Runs layer normalization followed by dropout."""
	output_tensor = layer_norm(use_fp16, input_tensor, norm_layer)
	output_tensor = dropout(output_tensor.float()).half()
	return output_tensor


def create_attention_mask_from_input_mask(from_tensor, to_mask):
	"""Create 3D attention mask from a 2D tensor mask.
  
	Args:
	  from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
	  to_mask: int32 Tensor of shape [batch_size, to_seq_length].
  
	Returns:
	  float Tensor of shape [batch_size, from_seq_length, to_seq_length].
	"""
	from_shape = from_tensor.size()
	batch_size = from_shape[0]
	from_seq_length = from_shape[1]
	to_mask = to_mask.unsqueeze(1).float()

	# We don't assume that `from_tensor` is a mask (although it could be). We
	# don't actually care if we attend *from* padding tokens (only *to* padding)
	# tokens so we create a tensor of all ones.
	#
	# `broadcast_ones` = [batch_size, from_seq_length, 1]
	broadcast_ones = torch.ones(batch_size, from_seq_length, 1).float()
	# Here we broadcast along two dimensions to create the mask.
	mask = broadcast_ones * to_mask
	return mask.unsqueeze(1)
