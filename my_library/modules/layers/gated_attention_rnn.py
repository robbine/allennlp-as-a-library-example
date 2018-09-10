import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GatedAttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.weight_UP = nn.Linear(input_size, hidden_size)
        self.weight_VP = nn.Linear(hidden_size, hidden_size)
        self.weight_V = nn.Linear(hidden_size, 1)
        self.weight_W = torch.empty(1, hidden_size + input_size)
        torch.nn.init.xavier_uniform_(self.weight_W)
        self.softmax = F.softmax
        self.gru = nn.GRU(input_size=input_size + hidden_size, hidden_size=hidden_size)

    def forward(self, input, hidden, match_input):
        weight_2 = self.weight_UP(input)
        weight_3 = self.weight_VP(hidden)
        weight_sum = match_input + weight_2 + weight_3
        weights = self.softmax(self.weight_V(torch.tanh(weight_sum)), dim=1)
        weight_input = torch.cat(((match_input * weights).sum(0).unsqueeze(0), input), 2)
        print(weight_input.size())
        gated_input = torch.sigmoid(self.weight_W * weight_input) * weight_input
        output, hidden = self.gru(gated_input, hidden)
        return output, hidden



def main():
    batch_size = 2
    input_size = 5
    hidden_size = 10
    p = torch.rand(3, batch_size, input_size)
    q = torch.rand(13, batch_size, 6)
    weight_UQ = nn.Linear(q.size(2), hidden_size)
    weighted_q = weight_UQ(q)
    hidden = torch.zeros(1, batch_size, hidden_size)
    rnn = GatedAttentionRNN(input_size, hidden_size)
    for i in range(p.size(0)):
        output, hidden = rnn(torch.unsqueeze(p[i], 0), hidden, weighted_q)
        print(output)

if __name__== "__main__":
    main()
