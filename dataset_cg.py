import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import random

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, split='train', split_ratio=0.9, img_size=256):
        """
        root_dir: data/
        split: train or val
        split_ratio: 90% train / 10% val
        """

        self.grey_paths = []
        self.color_paths = []

        # Collect Cars
        self.grey_paths += sorted(glob.glob(os.path.join(root_dir, "Cars/cars_grey/*")))
        self.color_paths += sorted(glob.glob(os.path.join(root_dir, "Cars/cars_colour/*")))

        # Collect Flowers
        self.grey_paths += sorted(glob.glob(os.path.join(root_dir, "Flowers/flowers_grey/*")))
        self.color_paths += sorted(glob.glob(os.path.join(root_dir, "Flowers/flowers_colour/*")))

        # Ensure grayscale & color counts match
        assert len(self.grey_paths) == len(self.color_paths), "Mismatch between grey & color images"

        # Create index split
        total = len(self.grey_paths)
        train_len = int(total * split_ratio)

        self.indices = list(range(total))
        random.shuffle(self.indices)

        self.indices = self.indices[:train_len] if split == 'train' else self.indices[train_len:]

        # Transforms
        self.transform_grey = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        self.transform_color = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_index = self.indices[idx]

        grey_img = Image.open(self.grey_paths[real_index]).convert("L")
        color_img = Image.open(self.color_paths[real_index]).convert("RGB")

        grey = self.transform_grey(grey_img)
        color = self.transform_color(color_img)

        return grey, color
