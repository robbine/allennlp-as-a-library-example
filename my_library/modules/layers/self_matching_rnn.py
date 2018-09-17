import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

@Seq2SeqEncoder.register('self_matching_rnn')
class SelfMatchAttentionRNN(Seq2SeqEncoder):
    def get_input_dim(self) -> int:
        return self.input_size

    def is_bidirectional(self) -> bool:
        return self.is_bidirectional

    def get_output_dim(self) -> int:
        return self.hidden_size

    def __init__(self, input_size, hidden_size, is_bidirectional):
        super(SelfMatchAttentionRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_bidirectional = is_bidirectional
        self.weight_UQ = nn.Linear(input_size, hidden_size)
        self.weight_UP = nn.Linear(input_size, hidden_size)
        self.weight_V = nn.Linear(hidden_size, 1)
        self.weight_W = nn.Linear(input_size + input_size, input_size + input_size)
        self.softmax = F.softmax
        self.gru = nn.GRU(input_size=input_size+input_size, hidden_size=hidden_size//2 if is_bidirectional else hidden_size, bidirectional=self.is_bidirectional)

    def forward(self, input, hidden):
        weight_sum = self.weight_UQ(input) + self.weight_UP(input)
        weights = self.softmax(self.weight_V(torch.tanh(weight_sum)), dim=1)
        weight_c = (input * weights).sum(0).unsqueeze(0).repeat(input.size(0), 1, 1)
        weight_input = torch.cat((weight_c, input), 2)
        gated_input = torch.sigmoid(self.weight_W(weight_input)) * weight_input
        output, hidden = self.gru(gated_input, hidden)
        return output, hidden

def main():
    batch_size = 1
    input_size = 5
    hidden_size = 10
    p = torch.rand(3, batch_size, input_size)
    hidden = torch.zeros(1, batch_size, hidden_size)
    rnn = SelfMatchAttentionRNN(input_size, hidden_size)
    print(rnn(p, hidden))

if __name__== "__main__":
    main()
