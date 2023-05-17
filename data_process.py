import torch
import torchvision.datasets
import torchvision.transforms as transforms

def base_transformation() -> transforms.Compose:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def load_cifar_10(dir, batch_size=64, transformations=base_transformation()) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_set = torchvision.datasets.CIFAR10(
        root=dir, train=True, download=True, transform=base_transformation())

    train_set_size = int(0.8 * len(train_set))
    val_set_size = len(train_set) - train_set_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_set_size, val_set_size])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(
        root=dir, train=False, download=True, transform=transformations)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    return (train_loader, val_loader, test_loader)


def load_cifar_100(dir, batch_size=64, transformations=base_transformation()) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_set = torchvision.datasets.CIFAR100(
        root=dir, train=True, download=True, transform=base_transformation())

    train_set_size = int(0.8 * len(train_set))
    val_set_size = len(train_set) - train_set_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_set_size, val_set_size])

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    test_set = torchvision.datasets.CIFAR100(
        root=dir, train=False, download=True, transform=transformations)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    return (train_loader, val_loader, test_loader)
