import logging
import unittest

import torch

from my_library.modules.layers.common_attention import get_relative_embeddings_left_right, \
	dot_product_unmasked_self_attention_relative_v2

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TestCommonAttention(unittest.TestCase):
	def setUp(self):
		super().setUp()

	'''
	test rembedding_to_padding
	'''

	# def test_embedding_to_padding(self):
	# 	emb = torch.randn(2, 3, 4)
	# 	actual = embedding_to_padding(emb)
	# 	expect = np.zeros((2, 3), dtype=float)
	# 	res = (expect == actual.data.numpy()).all()
	# 	self.assertTrue(res)
	#
	# def test_attention_bias_proximal(self):
	# 	actual = attention_bias_proximal(5)
	# 	self.assertEqual(list(actual.size()), [1, 1, 5, 5])
	#
	# def test_get_timing_signal_1d(self):
	# 	actual = get_timing_signal_1d(5, 3, -1)
	# 	self.assertEqual(list(actual.size()), [1, 5, 3])
	#
	# def test_add_timing_signal_1d(self):
	# 	x = torch.randn(2, 3, 5)
	# 	actual = add_timing_signal_1d(x, -1)
	# 	self.assertEqual(list(actual.size()), [2, 3, 5])
	#
	# def test_attention_bias_to_padding(self):
	# 	actual = attention_bias_to_padding(torch.randn(2, 1, 1, 5))
	# 	self.assertEqual(list(actual.size()), [2, 5])
	#
	# def test_compute_qkv(self):
	# 	batch = 2
	# 	length_q = 3
	# 	length_m = 5
	# 	channel = 7
	# 	key_size = 11
	# 	value_size = 13
	# 	query_antecedent = torch.randn(batch, length_q, channel)
	# 	memory_antecedent = torch.randn(batch, length_m, channel)
	# 	key_projection = nn.Linear(channel, key_size)
	# 	value_projection = nn.Linear(channel, value_size)
	# 	query_projection = nn.Linear(channel, key_size)
	# 	q, k, v = compute_qkv(query_antecedent, memory_antecedent, key_projection=key_projection,
	# 						  value_projection=value_projection, query_projection=query_projection)
	# 	self.assertEqual(list(q.size()), [2, 3, 11])
	# 	self.assertEqual(list(k.size()), [2, 5, 11])
	# 	self.assertEqual(list(v.size()), [2, 5, 13])
	#
	# def test_dot_product_attention(self):
	# 	batch = 2
	# 	length_q = 3
	# 	length_m = 5
	# 	channel = 7
	# 	key_size = 11
	# 	value_size = 13
	# 	query_antecedent = torch.randn(batch, length_q, channel)
	# 	memory_antecedent = torch.randn(batch, length_m, channel)
	# 	key_projection = nn.Linear(channel, key_size)
	# 	value_projection = nn.Linear(channel, value_size)
	# 	query_projection = nn.Linear(channel, key_size)
	# 	q, k, v = compute_qkv(query_antecedent, memory_antecedent, key_projection=key_projection,
	# 						  value_projection=value_projection, query_projection=query_projection)
	# 	actual = dot_product_attention(q, k, v)
	# 	self.assertEqual(list(actual.size()), [2, 3, 13])
	#
	# def test__generate_relative_positions_matrix(self):
	# 	actual = _generate_relative_positions_matrix(10, 3)
	# 	self.assertEqual(list(actual.size()), [10, 10])
	#
	# def test__generate_relative_positions_embeddings(self):
	# 	max_relative_position = 3
	# 	length = 10
	# 	depth = 8
	# 	vocab_size = max_relative_position * 2 + 1
	# 	embedding = torch.randn(vocab_size, depth)
	# 	actual = _generate_relative_positions_embeddings(length, depth, max_relative_position, embedding)
	# 	self.assertEqual(list(actual.size()), [10, 10, 8])
	#
	# def test__relative_attention_inner(self):
	# 	batch_size = 2
	# 	heads = 3
	# 	length = 5
	# 	depth = 7
	# 	x = torch.randn(batch_size, heads, length, length)
	# 	y = torch.randn(batch_size, heads, length, depth)
	# 	z = torch.randn(length, length, depth)
	# 	actual = _relative_attention_inner(x, y, z, False)
	# 	self.assertEqual(list(actual.size()), [2, 3, 5, 7])
	#
	# 	x = torch.randn(batch_size, heads, length, depth)
	# 	y = torch.randn(batch_size, heads, length, depth)
	# 	z = torch.randn(length, length, depth)
	# 	actual = _relative_attention_inner(x, y, z, True)
	# 	self.assertEqual(list(actual.size()), [2, 3, 5, 5])
	#
	# def test_dot_product_attention_relative(self):
	# 	batch = 2
	# 	heads = 3
	# 	length = 5
	# 	depth = 7
	# 	q = torch.randn(batch, heads, length, depth)
	# 	k = torch.randn(batch, heads, length, depth)
	# 	v = torch.randn(batch, heads, length, depth)
	# 	bias = None
	# 	max_relative_position = 3
	# 	vocab_size = max_relative_position * 2 + 1
	# 	dropout = torch.nn.functional.dropout
	# 	key_embedding = torch.randn(vocab_size, depth)
	# 	value_embedding = torch.randn(vocab_size, depth)
	# 	actual = dot_product_attention_relative(q, k, v, bias, max_relative_position, dropout, key_embedding,
	# 											value_embedding)
	# 	self.assertEqual(list(actual.size()), [2, 3, 5, 7])
	#
	# def test_ones_matrix_band_part(self):
	# 	rows = 2
	# 	cols = 3
	# 	num_lower = 5
	# 	num_upper = 10
	# 	actual = ones_matrix_band_part(rows, cols, num_lower, num_upper)
	# 	self.assertEqual(list(actual.size()), [2, 3])
	#
	# def test_attention_bias_lower_triangle(self):
	# 	actual = attention_bias_lower_triangle(10)
	# 	self.assertEqual(list(actual.size()), [1, 1, 10, 10])
	#
	# def test__make_local_block(self):
	# 	batch = 2
	# 	heads = 3
	# 	num_blocks = 5
	# 	block_length = 7
	# 	depth = 11
	# 	x = torch.randn(batch, heads, num_blocks, block_length, depth)
	# 	actual = _make_local_block(x, depth, batch, heads, num_blocks, block_length)
	# 	self.assertEqual(list(actual.size()), [batch, heads, num_blocks - 1, block_length * 2, depth])

	# def test_masked_local_attention_1d(self):
	# 	batch = 2
	# 	heads = 3
	# 	length = 200
	# 	depth_k = 7
	# 	depth_v = 11
	# 	q = torch.randn(batch, heads, length, depth_k)
	# 	k = torch.randn(batch, heads, length, depth_k)
	# 	v = torch.randn(batch, heads, length, depth_v)
	# 	actual = masked_local_attention_1d(q, k, v, block_length=64)
	# 	self.assertEqual(list(actual.size()), [2, 3, 200, 11])

	# def test_get_relative_embeddings_left(self):
	# 	num_heads = 5
	# 	max_relative_position = 3
	# 	depth = 11
	# 	length = 7
	# 	relative_embeddings = torch.randn(max_relative_position, depth)
	# 	actual = get_relative_embeddings_left(relative_embeddings, max_relative_position, length, depth,
	# 										  num_heads, True)
	# 	self.assertEqual(list(actual.size()), [7, 11])
	# 	relative_embeddings = torch.randn(num_heads, max_relative_position, depth)
	# 	actual = get_relative_embeddings_left(relative_embeddings, max_relative_position, length, depth,
	# 										  num_heads, False)
	# 	self.assertEqual(list(actual.size()), [5, 7, 11])
	#
	# def test_matmul_with_relative_values(self):
	# 	b = 2
	# 	h = 3
	# 	l = 5
	# 	m = 7
	# 	d = 11
	# 	x = torch.randn(b, h, l, m)
	# 	y = torch.randn(m, d)
	# 	actual = matmul_with_relative_values(x, y, True)
	# 	self.assertEqual(list(actual.size()), [b, h, l, d])
	# 	y = torch.randn(h, m, d)
	# 	actual = matmul_with_relative_values(x, y, False)
	# 	self.assertEqual(list(actual.size()), [b, h, l, d])
	#
	# def test__relative_position_to_absolute_position_masked(self):
	# 	batch = 2
	# 	heads = 3
	# 	length = 5
	# 	x = torch.randn(batch, heads, length, length)
	# 	actual = _relative_position_to_absolute_position_masked(x)
	# 	self.assertEqual(list(actual.size()), [2, 3, 5, 5])
	#
	# def test__relative_position_to_absolute_position_unmasked(self):
	# 	batch = 2
	# 	heads = 3
	# 	length = 5
	# 	x = torch.randn(batch, heads, length, 2 * length - 1)
	# 	actual = _relative_position_to_absolute_position_unmasked(x)
	# 	self.assertEqual(list(actual.size()), [batch, heads, length, length])
	#
	# def test__absolute_position_to_relative_position_unmasked(self):
	# 	batch = 2
	# 	heads = 3
	# 	length = 5
	# 	x = torch.randn(batch, heads, length, length)
	# 	actual = _absolute_position_to_relative_position_unmasked(x)
	# 	self.assertEqual(list(actual.size()), [2, 3, 5, 9])
	#
	# def test__absolute_position_to_relative_position_masked(self):
	# 	batch = 2
	# 	heads = 3
	# 	length = 5
	# 	x = torch.randn(batch, heads, length, length)
	# 	actual = _absolute_position_to_relative_position_masked(x)
	# 	self.assertEqual(list(actual.size()), [2, 3, 5, 5])

	# def test_masked_relative_local_attention_1d(self):
	# 	batch = 2
	# 	heads = 3
	# 	length = 200
	# 	depth_k = 7
	# 	depth_v = 11
	# 	q = torch.randn(batch, heads, length, depth_k)
	# 	k = torch.randn(batch, heads, length, depth_k)
	# 	v = torch.randn(batch, heads, length, depth_v)
	# 	heads_share_relative_embedding = True
	# 	add_relative_to_values = True
	# 	block_length = 64
	# 	rel_embed_length = block_length * 4
	# 	relative_key_embeddings = torch.randn(rel_embed_length, depth_k)
	# 	relative_value_embeddings = torch.randn(rel_embed_length, depth_v)
	# 	actual = masked_relative_local_attention_1d(q, k, v, block_length=block_length, relative_key_embeddings=relative_key_embeddings,
	# 												relative_value_embeddings=relative_value_embeddings,
	# 												dropout=torch.nn.functional.dropout,
	# 												heads_share_relative_embedding=heads_share_relative_embedding,
	# 												add_relative_to_values=add_relative_to_values)
	# 	self.assertEqual(list(actual.size()), [2, 3, 200, 11])

	# def test_masked_relative_local_attention_1d__2(self):
	# 	batch = 2
	# 	heads = 3
	# 	length = 200
	# 	depth_k = 7
	# 	depth_v = 11
	# 	q = torch.randn(batch, heads, length, depth_k)
	# 	k = torch.randn(batch, heads, length, depth_k)
	# 	v = torch.randn(batch, heads, length, depth_v)
	# 	heads_share_relative_embedding = False
	# 	add_relative_to_values = True
	# 	block_length = 64
	# 	rel_embed_length = block_length * 4
	# 	relative_key_embeddings = torch.randn(heads, rel_embed_length, depth_k)
	# 	relative_value_embeddings = torch.randn(heads, rel_embed_length, depth_v)
	# 	actual = masked_relative_local_attention_1d(q, k, v, block_length=block_length, relative_key_embeddings=relative_key_embeddings,
	# 												relative_value_embeddings=relative_value_embeddings,
	# 												dropout=torch.nn.functional.dropout,
	# 												heads_share_relative_embedding=heads_share_relative_embedding,
	# 												add_relative_to_values=add_relative_to_values)
	# 	self.assertEqual(list(actual.size()), [2, 3, 200, 11])

	# def test_masked_within_block_local_attention_1d(self):
	# 	batch = 2
	# 	heads = 3
	# 	length = 200
	# 	depth_k = 7
	# 	depth_v = 11
	# 	q = torch.randn(batch, heads, length, depth_k)
	# 	k = torch.randn(batch, heads, length, depth_k)
	# 	v = torch.randn(batch, heads, length, depth_v)
	# 	actual = masked_within_block_local_attention_1d(q, k, v)
	# 	self.assertEqual(list(actual.size()), [2, 3, 200, 11])

	def test_get_relative_embeddings_left_right(self):
		max_relative_position = 6
		length = 5
		depth = 7
		num_heads = 8
		heads_share_relative_embedding = False
		if heads_share_relative_embedding:
			relative_embeddings = torch.randn(2 * max_relative_position - 1, depth)
		else:
			relative_embeddings = torch.randn(num_heads, 2 * max_relative_position - 1, depth)
		actual = get_relative_embeddings_left_right(relative_embeddings, max_relative_position, length, depth,
													num_heads, heads_share_relative_embedding)
		self.assertTrue(True)

	def test_dot_product_unmasked_self_attention_relative_v2(self):
		batch = 2
		heads = 3
		length = 200
		depth = 7
		max_relative_position = 5
		q = torch.randn(batch, heads, length, depth)
		k = torch.randn(batch, heads, length, depth)
		v = torch.randn(batch, heads, length, depth)
		heads_share_relative_embedding = True
		add_relative_to_values = True
		max_relative_position_unmasked = max_relative_position * 2 - 1
		if heads_share_relative_embedding:
			relative_key_embeddings = torch.randn((max_relative_position_unmasked, depth), requires_grad=True)
			relative_value_embeddings = torch.randn(max_relative_position_unmasked, depth)
		else:
			relative_key_embeddings = torch.randn(heads, max_relative_position_unmasked, depth)
			relative_value_embeddings = torch.randn(heads, max_relative_position_unmasked, depth)

		actual = dot_product_unmasked_self_attention_relative_v2(q, k, v, bias=None,
																 relative_key_embeddings=relative_key_embeddings,
																 relative_value_embeddings=relative_value_embeddings,
																 max_relative_position=max_relative_position,
																 dropout=torch.nn.functional.dropout,
																 heads_share_relative_embedding=heads_share_relative_embedding,
																 add_relative_to_values=add_relative_to_values)
		self.assertEqual(list(actual.size()), [2, 3, 200, 7])


if __name__ == '__main__':
	unittest.main()
