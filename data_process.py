import torch
import torchvision.datasets
import torchvision.transforms as transforms
import pickle
import os


def enlarge_transformation() -> transforms.Compose:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((128, 128), antialias=True),
    ])
    return transform


def aug_transformations() -> transforms.Compose:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(128, scale=(
            0.8, 1.0), ratio=(0.8, 1.2), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((128, 128), antialias=True),
    ])
    return transform


def base_transformation() -> transforms.Compose:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform


def load_cifar_aug(dir, is_10, num_iteration=3, batch_size=64, split_ratio=0.8, pre_process=enlarge_transformation(), aug=aug_transformations()):

    train_set = None
    test_set = None

    if (is_10 == True):
        train_set = torchvision.datasets.CIFAR10(
            root=dir, train=True, download=True, transform=pre_process)
        test_set = torchvision.datasets.CIFAR10(
            root=dir, train=False, download=True, transform=pre_process)
    else:
        train_set = torchvision.datasets.CIFAR100(
            root=dir, train=True, download=True, transform=pre_process)
        test_set = torchvision.datasets.CIFAR100(
            root=dir, train=False, download=True, transform=pre_process)

    train_set_size = int(split_ratio * len(train_set))
    val_set_size = len(train_set) - train_set_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_set_size, val_set_size])

    new_train_set = train_set
    for i in range(num_iteration):
        aug_set = torchvision.datasets.CIFAR10(root=dir, train=True, download=True, transform=aug) if (
            is_10 == True) else torchvision.datasets.CIFAR100(root=dir, train=True, download=True, transform=aug)
        new_train_set = torch.utils.data.ConcatDataset(
            [new_train_set, aug_set])

    cached_train_set = None
    with open(os.path.join(dir, 'new_train_set.pkl'), 'wb') as f:
        pickle.dump(new_train_set, f)

    with open(os.path.join(dir, 'new_train_set.pkl'), 'rb') as f:
        cached_train_set = pickle.load(f)

    train_loader = torch.utils.data.DataLoader(
        cached_train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    return (train_loader, val_loader, test_loader)


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
