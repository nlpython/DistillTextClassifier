from torch import nn

class LstmClassifier(nn.Module):

    def __init__(self, args):
        super(LstmClassifier, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim // 2,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(args.hidden_dim, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        hidden_state, _ = self.lstm(embed)
        last_hidden_state = hidden_state[:, -1, :]
        out = self.dropout(last_hidden_state)
        logits = self.fc(out)

        return logits
