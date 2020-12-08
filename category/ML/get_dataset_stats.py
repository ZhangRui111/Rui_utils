"""
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # data normalization
])
"""
import numpy as np


def get_dataset_stats(dataset_name):
    if dataset_name == 'CIFAR10':
        # RGB
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    elif dataset_name == 'CIFAR100':
        # RGB
        mean = np.array([0.5071, 0.4865, 0.4409])
        std = np.array([0.2673, 0.2564, 0.2762])
    elif dataset_name == 'MNIST':
        mean = np.array([0.1307])
        std = np.array([0.3081])
    elif dataset_name == 'FashionMNIST':
        mean = np.array([0.2860])
        std = np.array([0.3530])
    elif dataset_name == 'KMNIST':
        mean = np.array([0.1904])
        std = np.array([0.3475])
    elif dataset_name == 'ImageNet':
        # RGB
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError()
    return mean, std
