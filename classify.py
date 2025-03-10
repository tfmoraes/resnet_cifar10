import argparse
import pathlib

import torch
from PIL import Image
from torchvision import transforms

from model import (
    ResNet18,
    ResNet20,
    ResNet34,
    ResNet50,
    ResNet56,
    ResNet101,
    ResNet110,
    ResnetModel,
)
from utils import ResizePad

MODELS = {
    "resnet18": ResNet18,
    "resnet20": ResNet20,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet56": ResNet56,
    "resnet101": ResNet101,
    "resnet110": ResNet110,
}

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-m",
    "--model",
    choices=list(MODELS.keys()),
    metavar="MODEL",
    default=list(MODELS.keys())[0],
    help="Models: {%(choices)s}",
)
parser.add_argument("-i", "--input", type=pathlib.Path, required=True)

parser.add_argument(
    "-b", "--batch-size", default=128, type=int, metavar="N", help="Batch size"
)

parser.add_argument(
    "-w", "--weights", type=pathlib.Path, default=pathlib.Path("best_model.pth")
)

args, _ = parser.parse_known_args()


def classify(
    model: ResnetModel,
    image_file: pathlib.Path,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
):
    model.to(device)

    print(image_file)

    image = Image.open(image_file)

    transform = transforms.Compose(
        [
            ResizePad((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    transformed_image = transform(image)
    output = model(transformed_image.unsqueeze(0).to(device))
    _, predicted = torch.max(output, 1)
    print(CLASSES[predicted])


def main():
    model = MODELS[args.model]()
    model.load_state_dict(torch.load(args.weights, map_location=torch.device("cpu")))
    model.eval()

    classify(model, args.input)


if __name__ == "__main__":
    main()
