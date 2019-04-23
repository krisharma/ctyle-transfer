import math
import numpy as np
import os
import pandas as pd
import csv
import shutil
import errno
from PIL import Image
from distutils.dir_util import copy_tree
from scipy.io import loadmat
from scipy.misc import imsave
import random
import cv2


def convert_mat_to_png(mat_dir, dataset_dir, param_csv, param_prefs_dict):
    img_list = os.listdir(mat_dir)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)


    param_df = pd.read_csv(param_csv)

    patient_id_list = list(param_df['Patient ID'])
    manufacturer_list = list(param_df['Manufacturer'])
    manufacturer_dict = {0: 'GE', 2: 'Siemens'}
    field_strength_list = list(param_df['Field strength (tesla)'])
    contrast_agent_list = list(param_df['Contrast agent'])
    slice_thickness_list = list(param_df['Slice Thickness'])

    #for now, will only train/test on 219/469 images that have field strength of 3 and contrast agent 1
    img_list = [img for img in img_list if field_strength_list[patient_id_list.index(img)] == 1 and contrast_agent_list[patient_id_list.index(img)] == 1.0]

    ge_img_list = list()
    siemens_img_list = list()

    for img in img_list:
        if(manufacturer_dict[manufacturer_list[patient_id_list.index(img)]] == 'GE'):
            ge_img_list.append(img)
        else:
            siemens_img_list.append(img)

    random.shuffle(ge_img_list)
    ge_train_list = ge_img_list[:int(3*len(ge_img_list)/4)]
    ge_test_list = ge_img_list[int(3*len(ge_img_list)/4):]


    random.shuffle(siemens_img_list)
    siemens_train_list = siemens_img_list[:int(3*len(siemens_img_list)/4)]
    siemens_test_list = siemens_img_list[int(3*len(siemens_img_list)/4):]

    print(len(ge_train_list), len(ge_test_list), len(siemens_train_list), len(siemens_test_list))
    #aggregate_and_save_slices(mat_dir, dataset_dir, ge_train_list, 'Train_GE')
    #aggregate_and_save_slices(mat_dir, dataset_dir, ge_test_list, 'Test_GE')
    #aggregate_and_save_slices(mat_dir, dataset_dir, siemens_train_list, 'Train_Siemens')
    #aggregate_and_save_slices(mat_dir, dataset_dir, siemens_test_list, 'Test_Siemens')



def aggregate_and_save_slices(mat_dir, dataset_dir, img_list, sub_dir):
        for img in img_list:
            current_img = os.path.join(mat_dir, img, 'pre_img.mat') #only looking at pre contrast images right now

            if not os.path.exists(os.path.join(dataset_dir, sub_dir)):
                os.mkdir(os.path.join(dataset_dir, sub_dir))

            img_array = loadmat(current_img)['dcmat']

            #key step --> only take middle 50% of slices in each MRI
            num_slices = img_array.shape[2]
            img_array = img_array[:, :, int(num_slices/4): int(3*num_slices/4)]


            for i in range (img_array.shape[2]):
                imsave(os.path.join(dataset_dir, sub_dir, img + '_slice_' + str(i) + '.png'), img_array[:,:,i])

            print("saved img: " , os.path.join(dataset_dir, sub_dir, img + '.png'), " " + str(img_array.shape[2]))


def flip_images(dir):
    image_list = os.listdir(dir)
    for image in image_list:
        img = cv2.imread(os.path.join(dir, image))
        (h, w) = img.shape[:2]
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        flipped = cv2.warpAffine(img, M, (h, w))
        imsave(os.path.join(dir, image), flipped)


#mat_dir = os.path.join('/home', 'adithya', 'Train_Subtype', 'Images')
#dataset_dir = os.path.join('/home', 'adithya', 'MRI_Dataset')
#param_csv = os.path.join('/home', 'adithya', 'Breast_Style_Transfer', 'ctyle-transfer', 'scanner_params.csv')
#param_dict = dict()

#convert_mat_to_png(mat_dir, dataset_dir, param_csv, param_dict)


flip_images(os.path.join('/home', 'adithya', 'MRI_Dataset', 'Test_Siemens', 'Test_Siemens'))
