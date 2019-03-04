import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from MRITransform import Normalize

class MRIDataset(Dataset):
    """MRI dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Parent directory containing all the image directories
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def read_and_interpolate(self, dir):
        #N x (1 x depth X 256 x 256)
        final_image = np.zeros([0,0,0,0], dtype=np.float)

        slice_files = os.listdir(dir)
        slice_files = [file for file in slice_files if 'mask' not in file] #get rid of all mask files

        #for file in slice_files:
        for i in range (20):
            index = i * len(slice_files)//20

            file = slice_files[index]
            slice = Image.open(os.path.join(dir, file))

            #256x256x1 needs to be transformed into 1x1x256x256
            slice = np.expand_dims(np.array(slice), axis=2)
            slice = slice/255.0

            slice = slice.transpose((2, 0, 1))
            slice = np.expand_dims(slice, axis=1)

            if final_image.size == 0:
                final_image = slice
            else:
                final_image = np.concatenate((final_image, slice), axis=1)

        return torch.from_numpy(final_image).type(torch.FloatTensor)


    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_list = os.listdir(self.root_dir)
        img_list.sort()
        img_dir = img_list[idx]

        img_dir = os.path.join(self.root_dir, img_dir)
        image = self.read_and_interpolate(img_dir)

        #normalize pixels to -1, 1 range
        image.sub_(0.5)
        image.div_(0.5)

        #label = 0 for 'pre', 1 for 'post'
        label = 0
        if("post" in self.root_dir): label = 1

        return image, label
#MRI_Dataset = MRIDataset(root_dir=os.path.join('./MRI_Data', 'flair'))
#for i in range(len(MRI_Dataset)):
    #sample = MRI_Dataset[i]
    #print(i, sample.shape)


#read_and_interpolate(os.path.join('/home', 'adi', 'Downloads', 'TCGA_HT_A61A_20000127'))
