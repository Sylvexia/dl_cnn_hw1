import visualizer
import data_process
import model_handler
import datetime
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from lion_pytorch import Lion

def train_cifar_100():
    cifar_100_dir = "./data/cifar_100"
    ex_5_100_dir = "./ex_5/cifar_100/" + \
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    num_epochs = 30
    
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    cnn = efficientnet_v2_s(weights=weights)
    train_loader, val_loader, test_loader = data_process.load_cifar_100(
        cifar_100_dir, batch_size=64, 
        aug=data_process.eff_aug(), pre_process=data_process.enlarge_transformation())
    
    lion = Lion(cnn.parameters(), lr=1e-4*(1/5), weight_decay=1e-4*5)
    scheduler = CosineAnnealingLR(lion, T_max=num_epochs)
    
    model_handler.train(num_epochs, cnn, train_loader,
                        val_loader, lion, ex_5_100_dir, 5, scheduler=scheduler)

    visualizer.getMetrics(cnn, test_loader, ex_5_100_dir)
    visualizer.save_result_fig(ex_5_100_dir)


def train_cifar_10():
    cifar_10_dir = "./data/cifar_10"
    ex_5_10_dir = "./ex_5/cifar_10/" + \
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    num_epochs = 30

    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    cnn = efficientnet_v2_s(weights=weights)
    train_loader, val_loader, test_loader = data_process.load_cifar_10(
        cifar_10_dir, batch_size=64,
        aug=data_process.eff_aug(), pre_process=data_process.enlarge_transformation())
    
    lion = Lion(cnn.parameters(), lr=1e-4*(1/5), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(lion, T_max=num_epochs)

    model_handler.train(num_epochs, cnn, train_loader, val_loader,
                        lion, ex_5_10_dir, 5, scheduler=scheduler)

    visualizer.getMetrics(cnn, test_loader, ex_5_10_dir)
    visualizer.save_result_fig(ex_5_10_dir)

if __name__ == '__main__':
    train_cifar_10()
    train_cifar_100()