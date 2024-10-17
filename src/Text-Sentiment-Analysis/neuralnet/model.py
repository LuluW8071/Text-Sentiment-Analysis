import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim)
        self.key_layer = nn.Linear(key_dim, hidden_dim)
        self.energy_layer = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, values):
        query = self.query_layer(query).unsqueeze(1)
        keys = self.key_layer(keys)

        # Compute energy scores
        energy = torch.tanh(query + keys)
        attention_scores = self.energy_layer(energy).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return context, attention_weights
    

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional=False, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim, 
                            num_layers,
                            bidirectional=bidirectional,
                            batch_first=True, 
                            dropout=dropout)

        # Attention
        attention_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AdditiveAttention(query_dim=attention_dim,
                                           key_dim=attention_dim,
                                           hidden_dim=hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(attention_dim, output_dim)


    def forward(self, x):
        # LSTM
        lstm_out, (hn, _) = self.lstm(x)   # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)

        # Set query as last hidden state of the LSTM
        if self.bidirectional:
            # Concatenate the last states of the two directions
            query = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            # Use the last hidden state for unidirectional LSTM
            query = hn[-1]

        # Use LSTM output as keys and values … to compute attention
        context, attention_weights = self.attention(query, lstm_out, lstm_out)

        # Passing the context vector through the output layer
        out = self.fc(context)
        return out, attention_weights


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional=False, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU Layer
        self.gru = nn.GRU(input_dim, 
                          hidden_dim, 
                          num_layers,
                          bidirectional=bidirectional,
                          batch_first=True, 
                          dropout=dropout)

        # Attention
        attention_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AdditiveAttention(query_dim=attention_dim,
                                           key_dim=attention_dim,
                                           hidden_dim=hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(attention_dim, output_dim)


    def forward(self, x):
        # GRU
        gru_out, hn = self.gru(x)   # gru_out: (batch_size, seq_len, hidden_dim * num_directions)

        # Set query as last hidden state of the GRU
        if self.bidirectional:
            # Concatenate the last states of the two directions
            query = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            # Use the last hidden state for unidirectional GRU
            query = hn[-1]

        # Use GRU output as keys and values to compute attention
        context, attention_weights = self.attention(query, gru_out, gru_out)

        # Passing the context vector through the output layer
        out = self.fc(context)
        return out, attention_weights