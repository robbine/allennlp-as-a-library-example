import torch
import torch.nn as nn
from overrides import overrides

from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn.activations import Activation


@MatrixAttention.register("soft-align")
class SoftAlignmentMatrixAttention(MatrixAttention):
	"""
	Computes attention between every entry in matrix_1 with every entry in matrix_2 using cosine
	similarity.
	"""

	def __init__(self,
				 input_dim: int,
				 hidden_dim: int,
				 activation: Activation = None) -> None:
		super().__init__()
		self._weight_matrix = nn.Linear(input_dim, hidden_dim)
		self._activation = activation or Activation.by_name('relu')()

	@overrides
	def forward(self,  # pylint: disable=arguments-differ
				matrix_1: torch.Tensor,
				matrix_2: torch.Tensor) -> torch.Tensor:
		matrix_1 = self._activation(self._weight_matrix(matrix_1))
		matrix_2 = self._activation(self._weight_matrix(matrix_2))
		return torch.bmm(matrix_1, matrix_2.transpose(-1, -2))


def main():
	batch_size = 1
	input_size = 5
	hidden_size = 10
	p = torch.rand(batch_size, 7, input_size)
	q = torch.rand(batch_size, 11, input_size)
	att = SoftAlignmentMatrixAttention(input_size, hidden_size)
	print(att(p, q).size())

if __name__== "__main__":
	main()