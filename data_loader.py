import os

# Torch imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

"""Creates training and test data loaders and pipeline."""
def get_data_loader(opts, image_type):
    transform = transforms.Compose([
                    transforms.Resize(opts.image_size), #resize 512x512 images to 256x256
                    transforms.RandomHorizontalFlip(), #new addition as a data augmentation tactic
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_path = os.path.join(opts.data_dir, 'Train_' + image_type)
    test_path = os.path.join(opts.data_dir, 'Test_' + image_type)

    print("TRAIN PATH: ", train_path, "TEST PATH: ", test_path)

    train_dloader = DataLoader(ImageDataset(train_path, transforms=transform), batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_dloader = DataLoader(ImageDataset(test_path, transforms=transform), batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader
