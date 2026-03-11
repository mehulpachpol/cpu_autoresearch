import torch
import torchvision
import torchvision.transforms as transforms
import os

# Constants
BATCH_SIZE = 64
EVAL_BATCHES = 20  # How many batches to evaluate to get a quick score

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Check if MNIST is already downloaded
    data_path = "./data/MNIST/processed"
    download_data = not os.path.exists(data_path)

    # Load datasets
    trainset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=download_data,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True
    )

    valset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=download_data,
        transform=transform
    )

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False
    )

    return trainloader, valloader


@torch.no_grad()
def evaluate_accuracy(model, valloader):
    """The immutable metric: Validation Accuracy (Higher is better)"""
    model.eval()
    correct = 0
    total = 0
    batches = 0

    for images, labels in valloader:
        # Flatten images for standard MLP
        images = images.view(images.shape[0], -1)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        batches += 1
        if batches >= EVAL_BATCHES:
            break

    model.train()
    return 100 * correct / total


# Run once to download data when script is executed directly
if __name__ == "__main__":
    print("Checking dataset...")

    if os.path.exists("./data/MNIST/processed"):
        print("MNIST dataset already present. Skipping download.")
    else:
        print("Downloading MNIST dataset...")

    get_dataloaders()
    print("Infrastructure ready.")