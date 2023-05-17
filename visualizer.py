import torch
import torch.nn as nn
import torchvision
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time

def getMetrics(model: nn.Module, data_loader, dir, criterion=nn.CrossEntropyLoss(), device=torch.device("cuda:0")):
    # this should load the train dataset since it would not calculate the gradients
    # load best model in dir, the name is best.pth
    model.load_state_dict(torch.load(os.path.join(dir, "best.pth")))
    model = model.to(device)
    model.eval()

    losses = 0.0

    y_true = []
    y_pred = []

    # start testing timer
    start = time.time()

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses += loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()

    # end testing timer
    end = time.time()
    test_time = end - start
    losses = losses/len(data_loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    precision_score_val = precision_score(y_true, y_pred, average='macro')
    recall_score_val = recall_score(y_true, y_pred, average='macro')
    f1_score_val = f1_score(y_true, y_pred, average='macro')
    y_true_bin = label_binarize(y_true, classes=list(range(len(data_loader.dataset.classes))))
    y_pred_bin = label_binarize(y_pred, classes=list(range(len(data_loader.dataset.classes))))
    auc_score = roc_auc_score(y_true_bin, y_pred_bin, average='macro')
    
    print("accuracy: ", accuracy)
    print("losses: ", losses)
    print("testing time: ", test_time)
    print("precision_score: ", precision_score_val)
    print("recall_score: ", recall_score_val)
    print("f1_score: ", f1_score_val)
    print("auc_score: ", auc_score)

    with open(os.path.join(dir, "test.json"), "w") as f:
        json.dump({"accuracy": accuracy, "losses": losses, "testing time": test_time,
                   "precision_score": precision_score_val.tolist(), "recall_score": recall_score_val.tolist(),
                   "f1_score": f1_score_val.tolist(), "auc_score": auc_score,
                   "y_true": y_true, "y_pred": y_pred}, f)

    return


def load_and_test(model: nn.Module, path, test_loader, device=torch.device("cuda:0")):
    model.load_state_dict(torch.load(path))
    return getMetrics(model, test_loader, device)


def save_gridify(images, save_dir):
    img = torchvision.utils.make_grid(images)
    img = img/2+0.5
    img.to("cpu")
    npimg = img.numpy()
    result = np.transpose(npimg, (1, 2, 0))
    plt.imshow(result)
    plt.savefig(os.path.join(save_dir, 'test.png'))


def testBatch(model, test_loader, save_dir, device=torch.device("cuda:0")):
    classes = test_loader.dataset.classes
    batch_size = test_loader.batch_size

    images, labels = next(iter(test_loader))

    save_gridify(images, save_dir)
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                                    for j in range(batch_size)))
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))

def save_loss_fig(save_dir):
    data = None
    with open(os.path.join(save_dir, "train.json"), "r") as f:
        data = json.load(f)

    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")

    plt.legend()

    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.clf()


def save_accu_fig(save_dir):
    data = None
    with open(os.path.join(save_dir, "train.json"), "r") as f:
        data = json.load(f)

    train_accuracies = data["train_accuracies"]
    val_accuracies = data["val_accuracies"]
    epochs = range(1, len(train_accuracies)+1)

    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")

    plt.legend()

    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.clf()

def save_confusion_mtx_fig(save_dir):
    data = None
    with open(os.path.join(save_dir, "test.json"), "r") as f:
        data = json.load(f)

    y_true = data["y_true"]
    y_pred = data["y_pred"]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap="Spectral")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix")

    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.clf()

def save_result_fig(save_dir):
    save_accu_fig(save_dir)
    save_loss_fig(save_dir)
    save_confusion_mtx_fig(save_dir)