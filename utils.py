import os
from typing import Literal

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


class EarlyStopper:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
    ):
        """
        patience: number of epochs to wait before stopping. if 0 deactivate early stop
        min_delta: minimun change to qualify as improvement
        mode: 'min' (for loss) or 'max' (for accuracy)
        """

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric = None

    def __call__(self, metric: float) -> bool:
        if self.patience == 0:
            return False

        if self.best_metric is None:
            self.best_metric = metric
            return False

        if self.mode == "min":
            improved = metric < self.best_metric - self.min_delta
        else:
            improved = metric > self.best_metric + self.min_delta

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ResizePad:
    def __init__(self, target_size, interpolation=InterpolationMode.BILINEAR):
        self.target_size = target_size  # (target_width, target_height)
        self.interpolation = interpolation

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            orig_height, orig_width = image.shape[-2:]
        else:
            orig_width, orig_height = F.get_image_size(image)
        target_width, target_height = self.target_size

        original_max = max(orig_width, orig_height)
        target_max = max(target_width, target_height)

        scaling_factor = target_max / original_max
        new_width = int(orig_width * scaling_factor)
        new_height = int(orig_height * scaling_factor)

        resized_image = F.resize(
            image, [new_height, new_width], interpolation=self.interpolation
        )

        pad_left = (target_width - new_width) // 2
        pad_right = (target_width - new_width) - pad_left
        pad_top = (target_height - new_height) // 2
        pad_bottom = (target_height - new_height) - pad_top

        padded_image = F.pad(
            resized_image, [pad_left, pad_top, pad_right, pad_bottom], fill=0
        )
        return padded_image


class FilenameClassDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform

        # Collect all image paths and extract class names from filenames
        self.images = []
        self.classes = []
        self.class_to_idx = {}  # Maps class name to integer label

        # Get all image files
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        for filename in sorted(os.listdir(root)):
            if filename.lower().endswith(image_extensions):
                # Extract class name from filename (adjust parsing logic as needed)
                class_name = filename.split(".")[
                    0
                ]  # Split on underscore; modify for your case

                # Add to paths and labels
                self.images.append(Image.open(os.path.join(root, filename)))
                self.classes.append(class_name)

        # Create class-to-index mapping
        unique_classes = sorted(list(set(self.classes)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.class_to_idx[self.classes[idx]]  # Convert to integer label

        if self.transform:
            image = self.transform(image)

        return image, label
