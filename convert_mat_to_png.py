import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave
import os
import shutil

def convert_mat_to_png(img_dir):
    img_list = os.listdir(img_dir)
    new_png_dir = os.path.join(os.path.dirname(img_dir), 'pngs')
    if not os.path.exists(new_png_dir):
        os.mkdir(new_png_dir)

    for img in img_list:
        current_img = os.path.join(img_dir, img)

        if not os.path.exists(os.path.join(new_png_dir, img)):
            os.mkdir(os.path.join(new_png_dir, img))

        sub_img_list = os.listdir(os.path.join(img_dir, img))
        sub_img_list = [sub_img for sub_img in sub_img_list if 'mat' in sub_img and 'header' not in sub_img]

        for sub_img in sub_img_list:
            new_sub_img_dir = os.path.join(new_png_dir, img, sub_img[:-4])

            if not os.path.exists(new_sub_img_dir):
                os.mkdir(new_sub_img_dir)

            img_array = loadmat(os.path.join(img_dir, img, sub_img))['dcmat']


            if(len(os.listdir(new_sub_img_dir)) == img_array.shape[2]):
                print("skipping because all images have been saved already: " + new_sub_img_dir)
                os.remove(os.path.join(img_dir, img, sub_img))
                continue

                
            #key step --> only take middle 50% of slices in each MRI
            num_slices = img_array.shape[2]
            
            print("og img array shape: ", img_array.shape)
            img_array = img_array[:, :, num_slices/4: 3*num_slices/4]
            print("new img array shape: ", img_array.shape)
            
            for i in range (img_array.shape[2]):
                    imsave(os.path.join(new_sub_img_dir, 'slice_' + str(i) + '.png'), img_array[:,:,i])


            print("saved img: " + os.path.join(new_sub_img_dir) + " " + str(img_array.shape[2]))
            os.remove(os.path.join(img_dir, img, sub_img))

    print "success"

#img_path = os.path.join('/root', '.local', 'share', 'Cryptomator', 'mnt', 'RBcKw0nRRJns_0', 'Train_Subtype', 'Images')
#convert_mat_to_png(img_path)
