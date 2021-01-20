# 1.Import
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from tqdm import tqdm

from utils import get_groups
from dataloader import WordDataset
from model import Word2Vec


# 2.Dataloader
BATCH_SIZE = 10

groups, dictionary = get_groups(window_size=5)
train_dataset = WordDataset(groups, dictionary)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# 3.Model
model = Word2Vec(word_size=len(dictionary), embedding_size=20)

# 3.Train
N_EPOCHS = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.1)
losses = []

for epoch in range(N_EPOCHS):
    batch_loss = []
    for X, y in tqdm(train_dataloader):
        output = model(X)
        loss = loss_fn(output, y)
        batch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(sum(batch_loss) / len(batch_loss))
print(losses)
