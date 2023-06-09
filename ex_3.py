import torch.optim as optim
import visualizer
import data_process
import model_handler
import datetime
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

cifar_10_dir = "./data/cifar_10"
cifar_100_dir = "./data/cifar_100"

ex_3_10_dir = "./ex_3/cifar_10/" + \
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ex_3_100_dir = "./ex_3/cifar_100/" + \
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def train_cifar_100():
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    cnn = efficientnet_v2_s(weights=weights)
    train_loader, val_loader, test_loader = data_process.load_cifar_100(
        cifar_100_dir, batch_size = 64, 
        aug=data_process.base_transformation(), pre_process=data_process.base_transformation())
    adam = optim.Adam(cnn.parameters(), lr=0.0001, weight_decay=0.0001)

    model_handler.train(40, cnn, train_loader,
                        val_loader, adam, ex_3_100_dir, 5)

    visualizer.getMetrics(cnn, test_loader, ex_3_100_dir)
    visualizer.save_result_fig(ex_3_100_dir)

def train_cifar_10():
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    cnn = efficientnet_v2_s(weights=weights)
    train_loader, val_loader, test_loader = data_process.load_cifar_10(
        cifar_10_dir, batch_size = 64, 
        aug=data_process.base_transformation(), pre_process=data_process.base_transformation())
    adam = optim.Adam(cnn.parameters(), lr=0.0001, weight_decay=0.0001)

    model_handler.train(40, cnn, train_loader, val_loader,
                        adam, ex_3_10_dir, 5)

    visualizer.getMetrics(cnn, test_loader, ex_3_10_dir)
    visualizer.save_result_fig(ex_3_10_dir)

if __name__ == '__main__':
    train_cifar_10()
    train_cifar_100()