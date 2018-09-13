import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common.params import Params


@Seq2SeqEncoder.register('gated_attention_rnn')
class GatedAttentionRNN(Seq2SeqEncoder):
	def get_input_dim(self) -> int:
		return self.input_size

	def is_bidirectional(self) -> bool:
		return self.is_bidirectional

	def get_output_dim(self) -> int:
		return self.hidden_size

	def __init__(self, input_size, hidden_size, is_bidirectional, match_size):
		super(GatedAttentionRNN, self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.match_size = match_size
		self.is_bidirectional = is_bidirectional
		self.weight_UP = nn.Linear(input_size, hidden_size)
		self.weight_VP = nn.Linear(hidden_size, hidden_size)
		self.weight_UQ = nn.Linear(match_size, hidden_size)
		self.weight_V = nn.Linear(hidden_size, 1)
		self.weight_W = nn.Linear(input_size + match_size, input_size + match_size)
		self.softmax = F.softmax
		if self.is_bidirectional:
			raise NotImplementedError
		self.gru = nn.GRU(input_size=self.input_size + self.match_size, hidden_size=self.hidden_size,
						  bidirectional=self.is_bidirectional)

	def transform_match_input(self, match_input):
		self.match_input = torch.transpose(match_input, 0, 1)
		self.weighted_match_input = self.weight_UQ(self.match_input)
		return

	def forward(self, input, hidden):
		weight_2 = self.weight_UP(input)
		weight_3 = self.weight_VP(hidden)
		weight_sum = self.weighted_match_input + weight_2 + weight_3
		weights = self.softmax(self.weight_V(torch.tanh(weight_sum)), dim=1)
		weight_input = torch.cat(((self.match_input * weights).sum(0).unsqueeze(0), input), 2)
		gated_input = torch.sigmoid(self.weight_W(weight_input)) * weight_input
		output, hidden = self.gru(gated_input, hidden)
		return output, hidden


def main():
	batch_size = 2
	input_size = 5
	hidden_size = 10
	p = torch.rand(3, batch_size, input_size)
	q = torch.rand(13, batch_size, 6)
	hidden = torch.zeros(1, batch_size, hidden_size)

	rnn = GatedAttentionRNN(input_size, hidden_size, False, q.size(2))
	rnn.transform_match_input(q)
	for i in range(p.size(0)):
		output, hidden = rnn(torch.unsqueeze(p[i], 0), hidden)
		print(output)


if __name__ == "__main__":
	main()
