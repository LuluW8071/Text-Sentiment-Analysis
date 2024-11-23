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
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers,
                            bidirectional=False,
                            batch_first=True,
                            dropout=dropout)

        # Attention layer
        self.attention = AdditiveAttention(query_dim=hidden_dim,
                                           key_dim=hidden_dim,
                                           hidden_dim=hidden_dim)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass input through embedding layer
        x = self.embedding(x)

        # Pass through LSTM layer
        lstm_out, (hn, _) = self.lstm(x)

        # Set query as the last hidden state of the LSTM
        query = hn[-1]

        # Compute attention
        context, attention_weights = self.attention(query, lstm_out, lstm_out)

        # Output layer with sigmoid activation
        out = self.fc(context)
        out = self.sigmoid(out).squeeze(1)
        return out, attention_weights