import os

# Torch imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from MRITransform import Compose, Normalize, Scale

from MRIDataset import MRIDataset

"""Creates training and test data loaders and pipeline."""
def get_data_loader2d(img_type, opts):
    transform = transforms.Compose([
                    transforms.Scale(opts.image_size),
                    transforms.RandomHorizontalFlip(), #new addition as a data augmentation tactic
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_path = os.path.join('./MRI_Data_2d', img_type)
    test_path = os.path.join('./MRI_Data_2d', 'Test_{}'.format(img_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader


def get_data_loader3d(img_type, opts):

    """
    Original:

    transform_combo = transforms.Compose([
                    transforms.ToPILImage(), --> fook that we dont even need to scale it we know every input finna b 256x256x3 (for nowww)
                    transforms.Scale(opts.image_size),  --> this will now be Resize
                    transforms.ToTensor(), --> taken care of in MRIDataset.read_and_interpolate
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    """

    train_path = os.path.join('./MRI_Data', img_type)
    test_path = os.path.join('./MRI_Data', 'Test_{}'.format(img_type))


    transformed_train_dataset = MRIDataset(root_dir=train_path)
    transformed_test_dataset = MRIDataset(root_dir=test_path)

    train_dloader = DataLoader(dataset=transformed_train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_dloader = DataLoader(dataset=transformed_test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader
