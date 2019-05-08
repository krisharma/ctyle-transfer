import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transformations=None, unaligned=False, mode='train'):
        self.transform = transformations
        self.files_ = sorted(os.listdir(root))
        self.files_ = [os.path.join(root, f) for f in self.files_]

    def __getitem__(self, index):
        item = self.transform(Image.open(self.files_[index % len(self.files_)]))
        return item

    def __len__(self):
        return len(self.files_)
