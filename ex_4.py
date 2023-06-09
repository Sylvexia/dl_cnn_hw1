import torch.optim as optim
import visualizer
import data_process
import model_handler
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def train_cifar_100():
    cifar_100_dir = "./data/cifar_100"
    ex_4_100_dir = "./ex_4/cifar_100/" + \
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    num_epochs = 15
    
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    cnn = efficientnet_v2_s(weights=weights)
    train_loader, val_loader, test_loader = data_process.load_cifar_100(
        cifar_100_dir, batch_size=64, 
        aug=data_process.eff_aug(), pre_process=data_process.enlarge_transformation())
    
    adam = optim.Adam(cnn.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(adam, T_max=num_epochs)

    model_handler.train(num_epochs, cnn, train_loader,
                        val_loader, adam, ex_4_100_dir, 5, scheduler=scheduler)

    visualizer.getMetrics(cnn, test_loader, ex_4_100_dir)
    visualizer.save_result_fig(ex_4_100_dir)


def train_cifar_10():
    cifar_10_dir = "./data/cifar_10"
    ex_4_10_dir = "./ex_4/cifar_10/" + \
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    num_epochs = 15

    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    cnn = efficientnet_v2_s(weights=weights)
    train_loader, val_loader, test_loader = data_process.load_cifar_10(
        cifar_10_dir, batch_size=64,
        aug=data_process.eff_aug(), pre_process=data_process.enlarge_transformation())
    
    adam = optim.Adam(cnn.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(adam, T_max=num_epochs)

    model_handler.train(num_epochs, cnn, train_loader, val_loader,
                        adam, ex_4_10_dir, 5, scheduler=scheduler)

    visualizer.getMetrics(cnn, test_loader, ex_4_10_dir)
    visualizer.save_result_fig(ex_4_10_dir)

if __name__ == '__main__':
    train_cifar_10()
    train_cifar_100()