import argparse
import datetime
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from model import ResNet18, ResNet20, ResNet34, ResNet50, ResNet56, ResNet101, ResNet110
from utils import EarlyStopper, FilenameClassDataset, ResizePad

LossFunctionType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

MODELS = {
    "resnet18": ResNet18,
    "resnet20": ResNet20,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet56": ResNet56,
    "resnet101": ResNet101,
    "resnet110": ResNet110,
}

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-m",
    "--model",
    choices=list(MODELS.keys()),
    metavar="MODEL",
    default=list(MODELS.keys())[0],
    help="Models: {%(choices)s}",
)
parser.add_argument(
    "-e",
    "--epochs",
    default=200,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)

parser.add_argument(
    "-b", "--batch-size", default=128, type=int, metavar="N", help="Batch size"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="Learning rate",
    dest="lr",
)

parser.add_argument(
    "--early-stop",
    default=5,
    type=int,
    metavar="N",
    help="Number of epochs of no improvement to early stop. If 0 then early-stop is not activated.",
    dest="early_stop",
)

args, _ = parser.parse_known_args()


def _train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: LossFunctionType,
    metrics_fn: Metric,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()
    num_batches = len(dataloader)
    train_loss = 0.0
    metrics_fn.reset()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        train_loss += loss.item()
        metrics_fn.update(outputs, labels)

        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / num_batches
    avg_train_accuracy = metrics_fn.compute()

    return avg_train_loss, avg_train_accuracy


def _evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: LossFunctionType,
    metrics_fn: Metric,
    device: torch.device,
):
    model.eval()
    num_batches = len(dataloader)
    validation_loss = 0.0
    metrics_fn.reset()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            validation_loss += loss.item()
            metrics_fn.update(outputs, labels)

    avg_validation_loss = validation_loss / num_batches
    avg_validation_accuracy = metrics_fn.compute()

    return avg_validation_loss, avg_validation_accuracy


def predict(
    model: nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(images.to(device))
        _, predicted = torch.max(output, 1)
    return predicted


def plot_custom_test(
    model: nn.Module,
    images: torch.Tensor,
    dataset_custom_test: FilenameClassDataset,
    epoch,
    writer: SummaryWriter,
    device: torch.device,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    axes = axes.ravel()
    predicted = predict(model, images, device)
    for ax, (image, _class), _class_pred in zip(axes, dataset_custom_test, predicted):
        if _class_pred == _class:
            color = "blue"
        else:
            color = "red"
        ax.imshow(image)
        ax.set_title(dataset_custom_test.classes[_class_pred], color=color)
        ax.axis("off")
    fig.tight_layout()
    writer.add_figure("custom dataset", fig, epoch)


def train(
    model: nn.Module,
    batch_size: int = 128,
    epochs: int = 10,
    learning_rate: float = 0.001,
    patience: int = 10,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
):
    # Mean and STD values calculated this way:
    # Stack all images into a single tensor
    # imgs = torch.stack([img for img, _ in dataset])  # Shape: (50000, 3, 32, 32)
    # Compute mean and std
    # mean = imgs.mean(dim=[0, 2, 3])  # Mean over (batch, height, width)
    # std = imgs.std(dim=[0, 2, 3])    # Std over (batch, height, width)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    transform_custom = transforms.Compose(
        [
            ResizePad((256, 256)),  # Your custom resize + pad transform
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats (adjust for your data)
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset_train = datasets.CIFAR10(
        root="./datasets/", train=True, download=True, transform=transform_train
    )

    # train_size = int(0.8 * len(dataset_train))  # 80% train
    # val_size = len(dataset_train) - train_size  # 20% validation
    # dataset_train, dataset_val = random_split(dataset_train, [train_size, val_size])

    dataset_test = datasets.CIFAR10(
        root="./datasets/", train=False, download=True, transform=transform
    )

    custom_test = True
    try:
        dataset_custom_test = FilenameClassDataset(root="./datasets/custom/")
        custom_test_images_tensor = torch.stack(
            [transform_custom(i[0]) for i in dataset_custom_test]
        )
    except FileNotFoundError:
        custom_test = False

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=200)
    metric_fn = MulticlassAccuracy(num_classes=10).to(device)

    hyperparams = {
        "model": model.__class__.__name__,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": epochs,
        "optimizer": scheduler.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
    }

    best_val_accuracy = float("-inf")
    best_model_path = "best_model.pth"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/cifar10_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model, torch.randn(1, 3, 32, 32).to(device))
    early_stopper = EarlyStopper(patience=patience, mode="max")
    for epoch in trange(epochs):
        avg_train_loss, avg_train_accuracy = _train(
            loader_train, model, criterion, metric_fn, optimizer, device
        )
        avg_val_loss, avg_val_accuracy = _evaluate(
            loader_test, model, criterion, metric_fn, device
        )

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", avg_train_accuracy, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", avg_val_accuracy, epoch)
        writer.add_scalar("Learning rate", current_lr, epoch)

        if custom_test:
            plot_custom_test(
                model,
                custom_test_images_tensor,
                dataset_custom_test,
                epoch,
                writer,
                device,
            )

        tqdm.write(
            f"Epoch {epoch + 1}/{epochs} | train loss: {avg_train_loss:.6f} | train acc: {avg_train_accuracy:.6f} | val loss: {avg_val_loss:.6f} | val acc: {avg_val_accuracy:.6f}"
        )

        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), best_model_path)

        if early_stopper(avg_val_accuracy):
            print(f"Early stop at epoch {epoch + 1}")
            break

    # Loading best weights for test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    avg_test_loss, avg_test_accuracy = _evaluate(
        loader_test, model, criterion, metric_fn, device
    )
    final_metrics = {"accuracy": avg_test_accuracy, "loss": avg_test_loss}
    if custom_test:
        # Plotting with best weights
        plot_custom_test(
            model,
            custom_test_images_tensor,
            dataset_custom_test,
            epochs,
            writer,
            device,
        )

    writer.add_hparams(hyperparams, final_metrics, run_name=f"exp_{timestamp}")
    tqdm.write(
        f"Best weight: loss {avg_test_loss:.6f}, accuracy{avg_test_accuracy:.6f}"
    )



if __name__ == "__main__":
    model = MODELS[args.model]()
    train(
        model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.early_stop,
    )
