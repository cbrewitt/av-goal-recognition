import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, in_shape, hidden_shape, out_shape, num_layers=1, bias=True, dropout=0.0):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(in_shape, hidden_shape, num_layers=num_layers, bias=bias,
                            dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_shape, out_shape)

    def forward(self, x, use_encoding=False):
        encoding, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze()
        output = self.linear(h_n)
        if not use_encoding:
            return output, (None, None)
        else:
            encoding, lengths = nn.utils.rnn.pad_packed_sequence(encoding, batch_first=True)
            return output, (self.linear(encoding), lengths)

