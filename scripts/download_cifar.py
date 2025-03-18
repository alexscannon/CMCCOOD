import torchvision
import os
from pathlib import Path

def download_cifar100():
    # Get the data root from environment variable or use default
    data_root = os.getenv('DATA_ROOT', './data')
    cifar_path = Path(data_root) / 'cifar100'
    print(f"Downloading CIFAR-100 to {cifar_path}")

    # Create the directory if it doesn't exist
    cifar_path.mkdir(parents=True, exist_ok=True)

    # Download training set
    print("Downloading CIFAR-100 training set...")
    torchvision.datasets.CIFAR100(
        root=str(cifar_path),
        train=True,
        download=True
    )

    # Download test set
    print("Downloading CIFAR-100 test set...")
    torchvision.datasets.CIFAR100(
        root=str(cifar_path),
        train=False,
        download=True
    )

    print(f"CIFAR-100 dataset downloaded successfully to {cifar_path}")

if __name__ == "__main__":
    download_cifar100()