import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from data_processing import labels as LABELS
import time
from datetime import timedelta

def get_time_dif(start_time):
    """
    get used time
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(model, device, model_name, lr, train_dataloader, val_dataloader):
    """
    train data
    """
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr ) 
    total_batch = 0
    val_best_acc = float('-inf')
    num_epochs = 3
    checkpoint_path = f"./checkpoint/{model_name}_best_model.pt"
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for _, (trains, labels) in enumerate(train_dataloader):
            outputs = model(trains.to(device))
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels.to(device))  # cross entropy loss
            loss.backward()
            optimizer.step()

            if total_batch % 100  == 0:
                # print results each 100 batchs
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                # get validation accuracy and loss
                val_acc, val_loss = evaluate(model, device, val_dataloader)
                # save best model
                if val_acc > val_best_acc:
                    val_best_acc = val_acc
                    torch.save(model, checkpoint_path)
                    improve = '*'  # set saving label
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter:{0:>4}, Tr-Loss:{1:>6.4}, Tr-Acc:{2:>6.2%}, Va-Loss:{3:>6.4}, Va-Acc:{4:>6.2%}, Time:{5}{6}'
                # print message
                print(msg.format(total_batch, loss.item(), train_acc, val_loss, val_acc, time_dif, improve))
            total_batch += 1
    
    model = torch.load(checkpoint_path)
    confusion_matrix = validate(model, device, val_dataloader)
    return confusion_matrix

def evaluate(model, device, dataload):
    """
    evaluate data
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in dataload:
            outputs = model(texts.to(device))
            loss = F.cross_entropy(outputs, labels.to(device))
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    model.train()
    return acc, loss_total / len(dataload)


def validate(model, device, dataload):
    """
    get confusion matrix
    """
    model.eval()
    loss_total = 0
    # predicted label columns, actual label rows
    confusion_matrix = torch.zeros(len(LABELS), len(LABELS), dtype=torch.long)
    with torch.no_grad():
        for texts, labels in dataload:
            outputs = model(texts.to(device))
            loss = F.cross_entropy(outputs, labels.to(device))
            loss_total += loss
            predict = torch.max(outputs.data, 1)[1]
            for i in range(len(labels)):
                confusion_matrix[labels[i], predict[i]] += 1
    
    return confusion_matrix