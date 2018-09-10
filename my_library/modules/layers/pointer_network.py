import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import nll_loss
from torch.nn.functional import cross_entropy
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.nn import Activation

class PointerNet(nn.Module):
    def __init__(self, seq_size, input_size, hidden_size):
        super(PointerNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_UP = nn.Linear(input_size, hidden_size)
        self.weight_VP = nn.Linear(hidden_size, hidden_size)
        self.weight_V = nn.Linear(hidden_size, 1)
        self.softmax = F.softmax
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden, spans):
        start_span_probs, end_span_probs = None, None
        loss = torch.zeros(1)
        for z in range(2):
            weight_1 = self.weight_UP(input)
            weight_2 = self.weight_VP(hidden)
            weight_sum = weight_1 + weight_2
            weights = self.softmax(self.weight_V(torch.tanh(weight_sum)), dim=1)
            if z == 0:
                start_span_probs = weights
            else:
                end_span_probs = weights
            labels = torch.where(spans[:, z] < input.size(0), spans[:, z], spans.new_full((1, spans.size(0)), input.size(0)))
            loss += cross_entropy(weights.squeeze(2).transpose(0, 1), labels.squeeze().type(torch.LongTensor))
            input = input * weights
            _, hidden = self.gru(input, hidden)
        return loss, start_span_probs, end_span_probs

def build_attention(input, hidden_size, weight):
    att = LinearMatrixAttention(hidden_size, hidden_size, 'x+y', Activation.by_name('tanhshrink')())
    attention = att.forward(input, weight)
    return torch.unsqueeze(torch.sum(input * attention, dim=0), 0)

def main():
    batch_size = 11
    input_size = 5
    hidden_size = 10
    seq_size = 13
    spans = torch.empty(batch_size, 2)
    spans[:, 0] = 0
    spans[:, 1] = 1
    weight = torch.empty(seq_size, 1, hidden_size)
    torch.nn.init.xavier_uniform_(weight)
    p = torch.rand(7, batch_size, input_size)
    q = torch.rand(seq_size, batch_size, hidden_size)
    net = PointerNet(seq_size, input_size, hidden_size)
    hidden = build_attention(q, hidden_size, weight)
    loss, start_span_probs, end_span_probs = net(p, hidden, spans)
    print(loss)

if __name__== "__main__":
    main()
