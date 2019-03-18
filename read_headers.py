import math
import numpy as np
import os
import pandas as pd
import csv
import shutil
import errno
#from mat4py import loadmat


def aggregate_train_and_test_patches(parent_dir):

    imgs = os.listdir(parent_dir)
    train_imgs = imgs[:8*len(imgs)/10]
    test_imgs = imgs[8*len(imgs)/10:]

    print(len(train_imgs), len(test_imgs))

    for sub_dir in train_imgs:
             for patch in os.listdir(os.path.join(parent_dir, sub_dir, 'T1_img')):	
                 os.rename(os.path.join(parent_dir, sub_dir, 'T1_img', patch), os.path.join(os.path.dirname(parent_dir), 'Dataset', parent_dir[-3:], parent_dir[-3:], sub_dir + "_"  +  patch))
             print("img done train: ", sub_dir)

   
    for sub_dir in test_imgs:
             for patch in os.listdir(os.path.join(parent_dir, sub_dir, 'T1_img')):      
                 os.rename(os.path.join(parent_dir, sub_dir, 'T1_img', patch), os.path.join(os.path.dirname(parent_dir), 'Dataset', 'Test_'+parent_dir[-3:], 'Test_'+parent_dir[-3:], sub_dir + "_"  +  patch))
             print("img done test: ", sub_dir)



			
def organize_imgs_by_header(img_dir, thickness_csv):
        thickness_df = pd.read_csv(thickness_csv)
       
	imgs = list(thickness_df['Img'])
        thicknesses = thickness_df['Thickness']
        
        
        for img in os.listdir(img_dir):	
		index = imgs.index(img)
                thickness = thicknesses[index]
	                
		img_folder = os.path.join(os.path.dirname(img_dir), 'Images_' + str(thickness))
                #img_folder = os.path.join('/home', 'adithya', 'Desktop', 'Adithya_Breast_Style_Transfer', 'Breast_MRI_Data', 'Images_' + str(thickness))                 
   
		if(os.path.isdir(img_folder)):
	             print("")	
                else:
                     os.mkdir(img_folder)
                print("Copying ", os.path.join(img_dir, img), " to ", os.path.join(img_folder, img))
 

                if(os.path.isdir(os.path.join(img_folder, img))):
                     print("image already copied over: ", os.path.join(img_folder, img))
	        else:
                     copy_dir(os.path.join(img_dir, img), os.path.join(img_folder, img))    


		print("Copied ", img, " to ", os.path.join(img_folder, img))
			

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
 
        shutil.copyfile(os.path.join(img_dir, img, header_file[0]), os.path.join('/home', 'adithya', 'Desktop', 'Adithya_Breast_Style_Transfer', 'headers',  img + '_header.mat')) 
           

def df_empty(columns, dtypes, index=None):
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)

    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)

    return df


def copy_dir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
		 print("yuh: ", src, dst)
		 raise



img_path = os.path.join('/root', '.local', 'share', 'Cryptomator', 'mnt', 'RBcKw0nRRJns_0', 'Train_Subtype', 'pngs')
csv_path = os.path.join('/home', 'adithya', 'Desktop', 'Adithya_Breast_Style_Transfer', 'thickness_labels.csv')
#organize_imgs_by_header(img_path, csv_path)

#/root/.local/share/Cryptomator/mnt/RBcKw0nRRJns_0/Train_Subtype/Images_1.0/

slice_subtype_dir = os.path.join('/root', '.local', 'share', 'Cryptomator', 'mnt', 'RBcKw0nRRJns_0', 'Train_Subtype', 'Images_1.1')
aggregate_train_and_test_patches(slice_subtype_dir)




#copy_img_headers(img_path)

#header_test_path = os.path.join('/home', 'adithya', 'Desktop', 'Adithya_Breast_Style_Transfer', 'headers')
#read_mat_header(header_test_path)



