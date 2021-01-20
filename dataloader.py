import torch
from torch.utils.data import Dataset
from utils import get_groups


class WordDataset(Dataset):
    def __init__(self, word_groups, dictionary):
        super(WordDataset, self).__init__()
        self.word_groups = word_groups
        self.dict_len = len(dictionary)

    def __getitem__(self, item):
        group = self.word_groups[item]
        x = torch.tensor([1 if i in group and i != group[len(group) // 2] else 0
                          for i in range(self.dict_len)], dtype=torch.float)
        y = torch.tensor(self.word_groups[item][-1])
        return x, y

    def __len__(self):
        return len(self.word_groups)


if __name__ == '__main__':
    dataset = WordDataset(*get_groups(window_size=3))
    print(dataset[0])
