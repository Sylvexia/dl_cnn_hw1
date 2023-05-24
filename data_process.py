import torch
import torchvision.datasets
import torchvision.transforms as transforms
import pickle
import os


def enlarge_transformation() -> transforms.Compose:
    transform = transforms.Compose([
        transforms.Resize((128, 128), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    return transform


def aug_transformations() -> transforms.Compose:
    transform = transforms.Compose([
        transforms.Resize((128, 128), antialias=True),
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    return transform


def eff_aug():
    transform = transforms.Compose([
        transforms.Resize((128, 128), antialias=True),
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    return transform


def base_transformation():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform


def load_cifar_aug(dir, is_10, batch_size=64, split_ratio=0.8, pre_process=enlarge_transformation(), 
                   aug=aug_transformations()):

    train_set = None
    test_set = None

    if (is_10 == True):
        train_set = torchvision.datasets.CIFAR10(
            root=dir, train=True, download=True)
        test_set = torchvision.datasets.CIFAR10(
            root=dir, train=False, download=True, transform=pre_process)
    else:
        train_set = torchvision.datasets.CIFAR100(
            root=dir, train=True, download=True)
        test_set = torchvision.datasets.CIFAR100(
            root=dir, train=False, download=True, transform=pre_process)

    train_set_size = int(split_ratio * len(train_set))
    val_set_size = len(train_set) - train_set_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_set_size, val_set_size])
    
    train_set.dataset.transform = aug
    val_set.dataset.transfrom = pre_process

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=16, 
        pin_memory=True, persistent_workers=True, prefetch_factor=8)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=16, 
        pin_memory=True, persistent_workers=True, prefetch_factor=8)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=16, 
        pin_memory=True, persistent_workers=True, prefetch_factor=8)

    return (train_loader, val_loader, test_loader)


def load_cifar_10(dir, batch_size=64, split_ratio=0.8, pre_process=base_transformation(), aug=aug_transformations()):

    train_set = torchvision.datasets.CIFAR10(
        root=dir, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(
        root=dir, train=False, download=True, transform=pre_process)

    train_set_size = int(split_ratio * len(train_set))
    val_set_size = len(train_set) - train_set_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_set_size, val_set_size])
    
    train_set.dataset.transform = aug
    val_set.dataset.transform = pre_process
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=16, 
        pin_memory=True, persistent_workers=True, prefetch_factor=8)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=16, 
        pin_memory=True, persistent_workers=True, prefetch_factor=8)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=16, 
        pin_memory=True, persistent_workers=True, prefetch_factor=8)

    return (train_loader, val_loader, test_loader)


def load_cifar_100(dir, batch_size=64, split_ratio=0.8, pre_process=base_transformation(), aug=aug_transformations()):

    train_set = torchvision.datasets.CIFAR100(
        root=dir, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(
        root=dir, train=False, download=True, transform=pre_process)

    train_set_size = int(split_ratio * len(train_set))
    val_set_size = len(train_set) - train_set_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_set_size, val_set_size])
    
    train_set.dataset.tranform = aug
    val_set.dataset.transform = pre_process

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
        persistent_workers=True, prefetch_factor=8)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=16, 
        pin_memory=True, persistent_workers=True, prefetch_factor=8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True,
        persistent_workers=True, prefetch_factor=8)

    return (train_loader, val_loader, test_loader)
