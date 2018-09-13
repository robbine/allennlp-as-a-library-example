import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.functional import nll_loss
from torch.nn.functional import cross_entropy
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.nn import Activation
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable


class PointerNet(torch.nn.Module):
	def __init__(self, input_size, hidden_size, match_size, is_bidirectional):
		super(PointerNet, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.is_bidirectional = is_bidirectional
		self.weight_UP = nn.Linear(input_size, hidden_size)
		self.weight_VP = nn.Linear(hidden_size, hidden_size)
		self.weight_V = nn.Linear(hidden_size, 1)
		self.weight_UQ = nn.Linear(match_size, hidden_size)
		self.param_V = Parameter(torch.zeros(hidden_size), requires_grad=True)
		self.weight_V2 = nn.Linear(hidden_size, 1)
		# self.weight_RQ = torch.zeros((hidden_size, hidden_size))
		self.c = Parameter(torch.zeros(1), requires_grad=True)
		self.softmax = F.softmax
		if self.is_bidirectional:
			raise NotImplementedError
		self.gru = nn.GRU(input_size=input_size,
						  hidden_size=hidden_size,
						  bidirectional=self.is_bidirectional)

	def forward(self, input, hidden):
		# hidden size: [1, batch_size, hidden_size]
		start_span_probs, end_span_probs = None, None
		seq_size = input.size(0)
		for z in range(2):
			weight_1 = self.weight_UP(input)
			weight_2 = self.weight_VP(hidden)
			weight_sum = weight_1 + weight_2
			weights = self.weight_V(torch.tanh(weight_sum))
			logits = weights + self.c # size: [seq_size, batch_size, 1]
			beta = self.softmax(logits, dim=0)
			if z == 0:
				start_span_probs = weights
			else:
				end_span_probs = weights
			beta = beta.view(-1, 1, seq_size) # size: [batch_size, 1, seq_size]
			HrBeta = torch.transpose(torch.matmul(beta, torch.transpose(input, 0, 1)), 0, 1) #size: [1, batch_size, input_size]
			_, hidden = self.gru(HrBeta, hidden) # size: [1, batch_size, hidden_size]
		return start_span_probs, end_span_probs


	def build_attention(self, input):
		input = torch.transpose(input, 0, 1)
		weight_sum = self.weight_UQ(input) + self.param_V
		weights = self.weight_V2(torch.tanh(weight_sum))
		return (input * weights).sum(0).unsqueeze(0)


def main():
	batch_size = 11
	input_size = 5
	hidden_size = 10
	seq_size = 13
	spans = torch.empty(batch_size, 2)
	spans[:, 0] = 0
	spans[:, 1] = 1
	p = torch.rand(7, batch_size, input_size)
	q = torch.rand(batch_size, seq_size, hidden_size)
	net = PointerNet(input_size, hidden_size, False)
	hidden = net.build_attention(q)
	print(hidden.size())
	net(p, hidden)

if __name__ == "__main__":
	main()
