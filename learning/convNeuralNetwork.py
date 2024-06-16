import torch
from torch import nn
from preprocess import tokenizer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(output[:, -1, :])


vocab_size = tokenizer.vocab_size
embed_dim = 128
hidden_dim = 256
output_dim = vocab_size
model = NeuralNetwork(vocab_size, embed_dim, hidden_dim, output_dim).to(device)
