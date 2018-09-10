import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfMatchAttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, match_input):
        super(SelfMatchAttentionRNN, self).__init__()

        self.hidden_size = hidden_size
        self.match_input = match_input
        self.match_input_size = match_input.size(2)

        self.weight_UQ = nn.Linear(self.match_input_size, hidden_size)
        self.weight_UP = nn.Linear(input_size, hidden_size)
        self.weight_VP = nn.Linear(hidden_size, hidden_size)
        self.weight_V = nn.Linear(hidden_size, 1)
        self.weight_W = torch.empty(1, self.match_input_size + input_size)
        torch.nn.init.xavier_uniform_(self.weight_W)
        self.softmax = F.softmax
        self.weighted_q = self.weight_UQ(self.match_input)
        self.gru = nn.GRU(input_size=input_size+self.match_input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        weight_2 = self.weight_UP(input)
        weight_sum = self.weighted_q + weight_2
        weights = self.softmax(self.weight_V(torch.tanh(weight_sum)), dim=1)
        weights = weights.squeeze(2).transpose(0, 1).unsqueeze(1)
        weighted_q = torch.bmm(weights, self.match_input.transpose(0, 1))
        weight_input = torch.cat((torch.transpose(weighted_q, 0, 1).repeat(input.size(0), 1, 1), input), 2)
        gated_input = torch.sigmoid(self.weight_W * weight_input) * weight_input
        output, hidden = self.gru(gated_input, hidden)
        return output, hidden

def main():
    batch_size = 1
    input_size = 5
    hidden_size = 10
    p = torch.rand(3, batch_size, input_size)
    hidden = torch.zeros(1, batch_size, hidden_size)
    rnn = SelfMatchAttentionRNN(input_size, hidden_size, p)
    print(rnn(p, hidden))

if __name__== "__main__":
    main()
