import math
import numpy as np
import os
import pandas as pd
from mat4py import loadmat
from shutil import copyfile

def read_mat_header(header_dir):

        thickness_df = df_empty(['Img', 'Thickness'], dtypes=[np.str, np.float64])
        i = 0

        for header in os.listdir(header_dir):
                header_data = loadmat(os.path.join(header_dir, header))
                thickness = float(header_data['shortHeader']['SliceThickness'])
                thickness = int(thickness*10)/10.0
                thickness_df.loc[i] = [header[:-11], thickness]
                i += 1

        thickness_df.to_csv(os.path.join('/home', 'adithya', 'Desktop', 'Adithya_Breast_Style_Transfer', 'thickness_labels.csv'))


def copy_img_headers(img_dir):
    img_list = os.listdir(img_dir)
    new_png_dir = os.path.join(os.path.dirname(img_dir), 'pngs')
    if not os.path.exists(new_png_dir):
        os.mkdir(new_png_dir)

    for img in img_list:
        current_img = os.path.join(img_dir, img)
        files = os.listdir(os.path.join(img_dir, img))
        header_file = [f for f in files if '.mat' in f]
 
        copyfile(os.path.join(img_dir, img, header_file[0]), os.path.join('/home', 'adithya', 'Desktop', 'Adithya_Breast_Style_Transfer', 'headers',  img + '_header.mat')) 
           

def df_empty(columns, dtypes, index=None):
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)

    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)

    return df


#img_path = os.path.join('/root', '.local', 'share', 'Cryptomator', 'mnt', 'RBcKw0nRRJns_0', 'Train_Subtype', 'Images')
#copy_img_headers(img_path)



header_test_path = os.path.join('/home', 'adithya', 'Desktop', 'Adithya_Breast_Style_Transfer', 'headers')
read_mat_header(header_test_path)
