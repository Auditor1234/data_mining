import torch.nn as nn   

class RNNModel(nn.Module):
    def __init__(self,
                 rnn_type="lstm", hidden_dim=256, class_num=10, n_layers=2, bidirectional=True, dropout=0.3, batch_first=True):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.embedding_size = 300
        # 定义词嵌入层
        self.embedding = nn.Embedding(4802, self.embedding_size)
        """
        if batch_first:
            output, hidden = [bitch_size, max_seq, hidden_size * bidirectional] \ 
            [num_layers * bidirectional, batch_size, hidden_size]
        else:
            output, hidden = [bitch_size, max_seq, hidden_size * bidirectional] \ 
            [num_layers * bidirectional, max_seq, hidden_size]
        """
        self.rnn = nn.LSTM(self.embedding_size,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first,
                           dropout=dropout)
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim * 2, class_num)

    def forward(self, x):
        x = self.embedding(x)
        self.rnn.flatten_parameters() # 扁平化
        output, (hidden, cell) = self.rnn(x)
        x = output[:, -1, :]
        x = self.dropout(x)
        logits = self.fc(x)

        return logits
