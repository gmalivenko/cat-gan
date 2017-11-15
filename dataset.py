from PIL import Image
from glob import glob

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class CatsDataset(Dataset):
    """Cats dataset."""

    def __init__(self, params):
        """
        Args:
            params (attr_dict): Parameters
        """
        self.images = glob(params.root_dir)
        print(transforms)
        self.transform = transforms.Compose([
            transforms.Scale(params.image_size),
            transforms.RandomCrop(params.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=params.brightness, contrast=params.contrast,
                saturation=params.saturation, hue=params.hue
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        sample = self.transform(image)
        return sample
