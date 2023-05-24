import torch.optim as optim
import base_cnn
import visualizer
import data_process
import model_handler
import datetime

cifar_10_dir = "./data/cifar_10"
cifar_100_dir = "./data/cifar_100"

ex_2_10_dir = "./ex_2/cifar_10/" + \
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ex_2_100_dir = "./ex_2/cifar_100/" + \
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def train_cifar_100():
    cnn = base_cnn.Large_CNN(100)
    adam = optim.AdamW(cnn.parameters(), lr=0.001, weight_decay=0.0001)
    train_loader, val_loader, test_loader = data_process.load_cifar_aug(
        cifar_100_dir, False, batch_size=128, 
        pre_process=data_process.enlarge_transformation(), aug=data_process.aug_transformations())

    model_handler.train(50, cnn, train_loader,
                        val_loader, adam, ex_2_100_dir, 5)

    visualizer.getMetrics(cnn, test_loader, ex_2_100_dir)
    visualizer.save_result_fig(ex_2_100_dir)


def train_cifar_10():
    cnn = base_cnn.Large_CNN(10)
    adam = optim.AdamW(cnn.parameters(), lr=0.001, weight_decay=0.0001)

    train_loader, val_loader, test_loader = data_process.load_cifar_aug(
        cifar_10_dir, True, batch_size=128,
        pre_process=data_process.enlarge_transformation(), aug=data_process.aug_transformations())

    model_handler.train(50, cnn, train_loader, val_loader,
                        adam, ex_2_10_dir, 5)

    visualizer.getMetrics(cnn, test_loader, ex_2_10_dir)
    visualizer.save_result_fig(ex_2_10_dir)

if __name__ == '__main__':
    train_cifar_10()
    train_cifar_100()
