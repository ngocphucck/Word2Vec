from torch import nn


class Word2Vec(nn.Module):

    def __init__(self, word_size, embedding_size):
        super(Word2Vec, self).__init__()
        self.linear1 = nn.Linear(word_size, embedding_size)
        self.linear2 = nn.Linear(embedding_size, word_size)
        self.classifier = nn.Softmax()

    def forward(self, X):
        out = self.linear1(X)
        out = self.linear2(out)
        out = self.classifier(out)
        return out
