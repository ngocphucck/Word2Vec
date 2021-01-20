import re


def remove_punctual(doc):
    rm_punctual_doc = re.sub(r'[^\w\s]', '', doc)
    return rm_punctual_doc


def tokenize(doc):
    doc = remove_punctual(doc)
    return doc.split(' ')


def get_corpus(file_path='./data/raw_data.txt'):
    with open(file_path, 'r') as f:
        corpus = [doc for doc in f.readlines()]
    return corpus


def get_vocab(corpus):
    vocab = set()
    for doc in corpus:
        vocab.update([word for word in tokenize(doc)])
    return vocab


def make_dictionary(vocab):
    dictionary = {word: index for index, word in enumerate(vocab)}
    return dictionary


def get_groups(file_path='./data/raw_data.txt', window_size=2):
    corpus = get_corpus(file_path)
    vocab = get_vocab(corpus)
    dictionary = make_dictionary(vocab)
    groups = []
    for doc in corpus:
        tokens = tokenize(doc)
        begin = 0
        end = begin + window_size

        while end < len(tokens):
            group = tuple(dictionary[tokens[i]] for i in range(begin, end))
            groups.append(group)
            begin += 1
            end += 1
    return groups, dictionary


if __name__ == '__main__':
    groups, _ = get_groups()
    print(groups)
    pass
