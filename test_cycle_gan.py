import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import misc
from models import CycleGenerator2d

"""Loads the generator and discriminator models from checkpoints."""
def load_checkpoint(checkpoint_dir, iteration_num):
    G_YtoX_path = os.path.join(checkpoint_dir, 'G_YtoX_' + str(iteration_num) + '_.pkl')
    G_YtoX = CycleGenerator2d()
    G_YtoX.load_state_dict(torch.load(G_YtoX_path, map_location=lambda storage, loc: storage))
    return G_YtoX

"""Loads the real image found in img_dir and transfer it to the style of Van Gogh using the specified model iteration. Then, save the painting in output_dir."""
def test_image_to_painting(img_dir, output_dir, iteration):
        image = Image.open(img_dir)
        image = np.stack((image, image, image), axis=2)

        x = TF.to_tensor(image)
        x.unsqueeze_(0)

        G_YtoX = load_checkpoint(os.path.join('./checkpoints_cyclegan'), iteration)

        generated_van_gogh = G_YtoX(x)
        generated_van_gogh = torch.squeeze(generated_van_gogh)
        generated_van_gogh = generated_van_gogh.detach().numpy()
        generated_van_gogh = np.swapaxes(generated_van_gogh, 0, 2)
        generated_van_gogh = np.rot90(np.flip(generated_van_gogh, 0), k=1, axes=(1,0))

        misc.imsave(output_dir, generated_van_gogh)

def test_all_images_in_dir(img_dir, output_dir, iteration):
    all_test_images = os.listdir(img_dir)
    for img in all_test_images:
        test_image_to_painting(os.path.join(img_dir, img), os.path.join(output_dir, img), iteration)


#transfer the specified image to a van gogh style painting
test_all_images_in_dir(os.path.join('./MRI_Data_2d', 'Test_pre_contrast'), os.path.join('./MRI_Data_2d', 'pre_contrast_to_flair'), 37000)
#test_image_to_painting(os.path.join('./test_images', 'baldwin.jpg'), os.path.join('./test_images', 'baldwin_painting.jpg'), 37000)
