import os
import torch
from tqdm import tqdm
import pickle as pkl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setSeed(seed=42)

labels = ['体育','娱乐','家居','教育','时政','游戏','社会','科技','股票','财经']
LABEL2ID = { x:i for (x,i) in zip(labels,range(len(labels)))}

# word set
word_set = {}

def load_word_set(filename):
    """
    create word set from input file
    """
    with open(filename, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f'creating word set from {filename}'):
            fields = line.strip()
            for w in fields:
                if w not in word_set:
                    word_set[w] = len(word_set)
    # label the unknown word
    word_set['[UNK]'] = len(word_set)
    print('size of word set:', len(word_set))

def tokenize_textCNN(s):
    """
    map word to index
    """
    max_size = 32  # max length for a sentence
    ts = [w for i, w in enumerate(s) if i < max_size] 
    ids = [word_set[w] if w in word_set.keys() else word_set['[UNK]'] for w in ts] # word to index
    ids += [word_set['[UNK]'] for _ in range(max_size-len(ts))] # ensure all sentences have the same length
    return ids

def load_data(filename):
    """
    get encoded data and label
    """
    labels = []
    data = []
    count = [0] * 10
    with open(filename, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Loading data', colour="green"):
            fields  = line.strip().split('\t')
            if len(fields) != 2 :
                continue
            labels.append(LABEL2ID[fields[0]])
            count[LABEL2ID[fields[0]]] += 1
            data.append(tokenize_textCNN(fields[1]))
    f.close()
    print('number of each class:', count)
    return data, labels

class MyData(Dataset):
    """
    dataset class
    """
    def __init__(self, data, labels):
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def getDataLoader(train_dataset, val_dataset):
    """
    dataloader for train and val
    """
    batch_size = 128
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    dev_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_dataloader, dev_dataloader


def my_metrics(confusion_matrix, eps=1e-6):
    """
    compute metrics from confusion matrix
    """
    category = len(confusion_matrix)
    results = torch.zeros(category, 4)
    for i in range(category):
        tp = confusion_matrix[i][i]
        fp = confusion_matrix[:i, i].sum() + confusion_matrix[i + 1 :, i].sum()
        fn = confusion_matrix[i, :i].sum() + confusion_matrix[i, i + 1:].sum()
        tn = confusion_matrix.sum() - tp - fp - fn

        acc = (tp + tn) / (tp + fp + tn + fn + eps)
        pre = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * pre * rec / (pre + rec + eps)
        results[i] = torch.tensor([acc, pre, rec, f1])

    return results
    