import os
import torch
import numpy as np
from models import RNNModel

from data_processing import MyData, getDataLoader, load_data, my_metrics, load_word_set
from data_processing import labels as LABELS
from train import train


def setSeed(seed):
    """
    set seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setSeed(seed=42)

def trainDNN():
    print("----------------------\n", RNNModel.__name__)
    filename = 'data/train.txt'
    load_word_set(filename=filename)
    dataset, labels = load_data(filename=filename)
    N = np.random.permutation(len(dataset))
    dataset = np.array(dataset)[N]
    labels = np.array(labels)[N]
    lr = 1e-3  # optimizer lr
    fold = 10 # ten fold cross validation
    category = len(LABELS) # num of category
    fold_size = int(1 / fold * len(dataset))
    confusion_matrix_total = torch.zeros(category, category)
    macro_metrics = torch.zeros(category, 4)
    micro_metrics = torch.zeros(category, 4)

    for i in range(fold):
        print(f'\n----------{i + 1} fold training start----------')
        train_data = np.concatenate((dataset[0 : i * fold_size], dataset[(i + 1) * fold_size :]), 0)
        train_label = np.concatenate((labels[0 : i * fold_size], labels[(i + 1) * fold_size :]), 0)
        val_data = dataset[i * fold_size : (i + 1) * fold_size]
        val_label = labels[i * fold_size : (i + 1) * fold_size]
        train_dataset = MyData(train_data, train_label)
        val_dataset = MyData(val_data, val_label)
        train_loader, val_loader = getDataLoader(train_dataset, val_dataset)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device
        model = RNNModel().to(device)
        confusion_matrix = train(model, device, RNNModel.__name__, lr, train_loader, val_loader)  # start training
        confusion_matrix_total += confusion_matrix
        macro_metrics += my_metrics(confusion_matrix) / fold
    micro_metrics = my_metrics(confusion_matrix_total)

    title = '      {:>12}{:>12}{:>12}{:>12}'.format('accuracy', 'precision', 'recall', 'f1')
    fmt = '{:>}  {:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}'
    print('macro metrics:')
    print(title)
    for i in range(category):
        metric = macro_metrics[i]
        print(fmt.format(LABELS[i], metric[0].item(), metric[1].item(), metric[2].item(), metric[3].item()))

    print('\nmicro_metrics:')
    print(title)
    for i in range(category):
        metric = micro_metrics[i]
        print(fmt.format(LABELS[i], metric[0].item(), metric[1].item(), metric[2].item(), metric[3].item()))

    print('\nconfusion matrix:\n', confusion_matrix.numpy())

if __name__ == '__main__':
    trainDNN()