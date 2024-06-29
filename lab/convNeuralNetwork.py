import torch
from torch import nn
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, max_norm=True)
        self.rnn = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU()
        )

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(output[:, -1, :])


train_vocab_size = tokenizer.vocab_size
train_embed_dim = 128
train_hidden_dim = 256
train_output_dim = train_vocab_size
model = NeuralNetwork(train_vocab_size, train_embed_dim, train_hidden_dim).to(device)
