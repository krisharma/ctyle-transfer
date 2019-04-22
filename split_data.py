import os
import random
import numpy as np
from PIL import Image
from distutils.dir_util import copy_tree

#go through all images, remove masks, and split each tif into individual channels, then save them appropriately
#
def split_data(dir):
    all_images = os.listdir(dir)
    all_images = [image for image in all_images if os.path.isdir(os.path.join(dir, image))]

    random.shuffle(all_images)

    cutoff_index = 3 * len(all_images) // 4
    train_images = all_images[:cutoff_index]
    test_images = all_images[cutoff_index:]


    for img in train_images:
        slice_files = os.listdir(os.path.join(dir, img))
        slice_files = [file for file in slice_files if 'mask' not in file]

        for file in slice_files:
            slice = Image.open(os.path.join(dir, img, file))

            pre_c, flair, post_c = slice.split()

            if not os.path.exists(os.path.join('./MRI_Data', 'pre_contrast', img)):
                os.mkdir(os.path.join('./MRI_Data', 'pre_contrast', img))

            pre_c.save(os.path.join('./MRI_Data', 'pre_contrast', img, file[:-4] + '.png'))



    for img in test_images:
        slice_files = os.listdir(os.path.join(dir, img))
        slice_files = [file for file in slice_files if 'mask' not in file]

        for file in slice_files:
            slice = Image.open(os.path.join(dir, img, file))

            pre_c, flair, post_c = slice.split()

            if not os.path.exists(os.path.join('./MRI_Data', 'Test_pre_contrast', img)):
                os.mkdir(os.path.join('./MRI_Data', 'Test_pre_contrast', img))

            pre_c.save(os.path.join('./MRI_Data', 'Test_pre_contrast', img, file[:-4] + '.png'))


            if not os.path.exists(os.path.join('./MRI_Data', 'Test_flair', img)):
                os.mkdir(os.path.join('./MRI_Data', 'Test_flair', img))

            flair.save(os.path.join('./MRI_Data', 'Test_flair', img, file[:-4] + '.png'))


            if not os.path.exists(os.path.join('./MRI_Data', 'Test_post_contrast', img)):
                os.mkdir(os.path.join('./MRI_Data', 'Test_post_contrast', img))

            post_c.save(os.path.join('./MRI_Data', 'Test_post_contrast', img, file[:-4] + '.png'))



def aggregate_2d_patches(parent_dir):
    all_subdirs = os.listdir(parent_dir)
    for subdir in all_subdirs:
        all_images = os.listdir(os.path.join(parent_dir, subdir))
        for image_dir in all_images:
            copy_tree(os.path.join(parent_dir, subdir, image_dir), os.path.join('/home', 'adi', 'hdd1', 'MRI_Data_2d', subdir))



aggregate_2d_patches('/home/adi/hdd1/mazurowski_research/3d_cycle_gan_code/MRI_Data')

#aggregate_2d_patches(os.path.join('/home', 'adi', 'hdd1', 'mazurowksi_research', '3d_cycle_gan_code', 'MRI_Data'))
#split_data(os.path.join('./LGG-segmentation'))
