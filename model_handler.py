import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import json

# for training

def train(num_epochs, model, train_loader, val_loader,
          optimizer, save_dir,
          save_every_epochs=5, loss_fn=nn.CrossEntropyLoss(),
          device=torch.device("cuda:0")):

    best_val_accuracy = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_time = 0.0

    print("Training on: ", device)
    model = model.to(device)

    # start training timer
    start = time.time()

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_total = 0.0
        train_correct = 0.0
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss/len(train_loader.dataset)
        train_accuracy = (100*train_correct)/train_total

        model.eval()
        val_loss = 0.0
        val_total = 0.0
        val_correct = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()*images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss/len(val_loader.dataset)
        val_accuracy = (100*val_correct)/val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1, train_loss))
        print('Epoch: {} \tValidation Loss: {:.6f}'.format(
            epoch+1, val_loss))
        print('Epoch: {} \tTraining Accuracy: {:.6f}'.format(
            epoch+1, train_accuracy))
        print('Epoch: {} \tValidation Accuracy: {:.6f}'.format(
            epoch+1, val_accuracy))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if (val_accuracy > best_val_accuracy):
            best_epoch = epoch+1
            torch.save(model.state_dict(), os.path.join(
                save_dir, 'best.pth'.format(best_epoch)))
            best_val_accuracy = val_accuracy

        if (epoch % save_every_epochs == save_every_epochs-1):
            torch.save(model.state_dict(), os.path.join(
                save_dir, 'epoch-{}.pth'.format(epoch+1)))

    # end training timer
    end = time.time()
    train_time = end-start

    print("Best accuracy: ", best_val_accuracy, " at epoch: ", best_epoch)
    print("Training time: ", train_time)

    # write losses, accuracies and training time to json file
    with open(os.path.join(save_dir, "train.json"), "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses,
                   "train_accuracies": train_accuracies, "val_accuracies": val_accuracies,
                   "best_val_accuracy": best_val_accuracy, "best_epoch": best_epoch,
                   "train_time": train_time}, f)
    return
