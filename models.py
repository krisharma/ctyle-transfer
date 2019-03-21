# Model architectures for generators and discriminators

# Torch imports
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) --> N x Cin x H x W
torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) --> N x Cin x D X H X W

torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) --> N x Cin x H x W
torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) --> N x Cin x D X H X W

# TODO:  look @ documentation carefully
torch.nn.ReflectionPad2d(padding)
torch.nn.functional.pad(input, pad, mode='constant', value = 0)

torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False) --> N x C x H x W
torch.nn.InstanceNorm3d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False) --> N x C x D x H x W
"""

#########################################
################2D MODELS###############
#########################################

"""Creates a transposed-convolutional layer, with optional batch normalization."""
def deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=0, instance_norm=True, reflect_pad=False):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False))

    if instance_norm:
       layers.append(nn.InstanceNorm2d(out_channels))

    if reflect_pad:
       layers.append(nn.ReflectionPad2d(3))

    return nn.Sequential(*layers)


"""Creates a convolutional layer, with optional batch normalization."""
def conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, instance_norm=True, init_zero_weights=False, reflect_pad=False):
    layers = []

    if reflect_pad:
       layers.append(nn.ReflectionPad2d(3))

    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
       conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001

    layers.append(conv_layer)

    if instance_norm:
       layers.append(nn.InstanceNorm2d(out_channels))

    return nn.Sequential(*layers)


"""Implementation of ResNet code for preserving input through deep architecture."""
class ResnetBlock2d(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock2d, self).__init__()
        self.conv_layer = conv2d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out

"""Defines the architecture of the generator network (both generators G_XtoY an G_YtoX have the same architecture)."""
class CycleGenerator2d(nn.Module):
    def __init__(self, init_zero_weights=False):
        super(CycleGenerator2d, self).__init__()

        ####   GENERATOR ARCHITECTURE   ####

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=1, padding=0, reflect_pad=True)
        self.conv2 = conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=2, padding=1)

        # 2. Define the transformation part of the generator
        self.resnet_block1 = ResnetBlock2d(conv_dim=256)
        self.resnet_block2 = ResnetBlock2d(conv_dim=256)
        self.resnet_block3 = ResnetBlock2d(conv_dim=256)
        self.resnet_block4 = ResnetBlock2d(conv_dim=256)
        self.resnet_block5 = ResnetBlock2d(conv_dim=256)
        self.resnet_block6 = ResnetBlock2d(conv_dim=256)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv2d_1 = deconv2d(in_channels=256, out_channels=192, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2d_2 = deconv2d(in_channels=192, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = conv2d(in_channels=128, out_channels=3, kernel_size=7, stride=1, padding=0, reflect_pad=True, instance_norm=False)

    def forward(self, x):
        """Generates an image conditioned
           on an input image.

            Input
            -----
                x: batch_size x 1 x N x N

            Output
            ------
                out: batch_size x 1 x N x N
        """

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = F.relu(self.resnet_block1(out))
        out = F.relu(self.resnet_block2(out))
        out = F.relu(self.resnet_block3(out))
        out = F.relu(self.resnet_block4(out))
        out = F.relu(self.resnet_block5(out))
        out = F.relu(self.resnet_block6(out))

        out = F.relu(self.deconv2d_1(out))
        out = F.relu(self.deconv2d_2(out))
        out = F.tanh(self.conv4(out))

        return out

"""Defines the architecture of the discriminator network (both discriminators D_X and D_Y have the same architecture)."""
class PatchGANDiscriminator2d(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator2d, self).__init__()

        #### ARCHITECTURE ####
        self.conv1 = conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, instance_norm=False)
        self.conv2 = conv2d(in_channels=128, out_channels=192, kernel_size=4, stride=2, padding=1)
        self.conv3 = conv2d(in_channels=192, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv5 = conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, instance_norm=False)


    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2, inplace=True)
        out = F.sigmoid(self.conv5(out))

        return out


#########################################
################3D MODELS###############
#########################################

"""Creates a transposed-convolutional layer, with optional batch normalization."""
def deconv3d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=0, instance_norm=True, reflect_pad=False):
    layers = []
    layers.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False))

    if instance_norm:
       layers.append(nn.InstanceNorm3d(out_channels))

    #if reflect_pad:
       #layers.append(F.pad(pad=(3,3,3,3,3,3), mode='reflect'))

    return nn.Sequential(*layers)


"""Creates a convolutional layer, with optional batch normalization."""
def conv3d(in_channels, out_channels, kernel_size, stride=2, padding=1, instance_norm=True, init_zero_weights=False, reflect_pad=False):
    layers = []

    #if reflect_pad:
      # layers.append(F.pad(pad=(3,3,3,3,3,3), mode='reflect'))

    conv_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
       conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001

    layers.append(conv_layer)

    if instance_norm:
       layers.append(nn.InstanceNorm3d(out_channels))

    return nn.Sequential(*layers)


"""Implementation of ResNet code for preserving input through deep architecture."""
class ResnetBlock3d(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock3d, self).__init__()
        self.conv_layer = conv3d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out

"""Defines the architecture of the generator network (both generators G_XtoY an G_YtoX have the same architecture)."""
class CycleGenerator3d(nn.Module):
    def __init__(self, init_zero_weights=False):
        super(CycleGenerator3d, self).__init__()

        ####   GENERATOR ARCHITECTURE   ####

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv3d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0, reflect_pad=True)
        self.conv2 = conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        # 2. Define the transformation part of the generator
        self.resnet_block1 = ResnetBlock3d(conv_dim=256)
        self.resnet_block2 = ResnetBlock3d(conv_dim=256)
        self.resnet_block3 = ResnetBlock3d(conv_dim=256)
        self.resnet_block4 = ResnetBlock3d(conv_dim=256)
        self.resnet_block5 = ResnetBlock3d(conv_dim=256)
        self.resnet_block6 = ResnetBlock3d(conv_dim=256)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv3d_1 = deconv3d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3d_2 = deconv3d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv4 = conv3d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0, reflect_pad=True, instance_norm=False)

    def forward(self, x):
        """Generates an image conditioned
           on an input image.

            Input
            -----
                x: batch_size x 1 x depth x height x width

            Output
            ------
                out: batch_size x 1 x depth x height x width
        """

        out = F.relu(F.pad(self.conv1(x), pad=(3,3,3,3,3,3), mode='reflect')) #gotta do 3d padding here cuz only supported by F
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = F.relu(self.resnet_block1(out))
        out = F.relu(self.resnet_block2(out))
        out = F.relu(self.resnet_block3(out))
        out = F.relu(self.resnet_block4(out))
        out = F.relu(self.resnet_block5(out))
        out = F.relu(self.resnet_block6(out))

        out = F.relu(self.deconv3d_1(out))
        out = F.relu(self.deconv3d_2(out))
        out = F.tanh(F.pad(self.conv4(out), pad=(3,3,3,3,3,3), mode='reflect')) #do 3d padding again

        return out


"""Defines the architecture of the discriminator network (both discriminators D_X and D_Y have the same architecture)."""
class PatchGANDiscriminator3d(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator3d, self).__init__()

        #### ARCHITECTURE ####
        self.conv1 = conv3d(in_channels=3, out_channels=64, kernel_size=4, stride=(2,2,1), padding=1, instance_norm=False)
        self.conv2 = conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=(2,2,1), padding=1)
        self.conv3 = conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=(2,2,1), padding=1)
        self.conv4 = conv3d(in_channels=256, out_channels=512, kernel_size=(4,4,2), stride=(2,2,1), padding=1)
        self.conv5 = conv3d(in_channels=512, out_channels=1, kernel_size=(4,4,1), stride=1, padding=1, instance_norm=False)


    def forward(self, x):

        print("X: ", x.shape)
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        print("X: ", out.shape)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2, inplace=True)
        print("X: ", out.shape)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2, inplace=True)
        print("X: ", out.shape)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2, inplace=True)

        print("yih: ", out.shape)

        out = F.sigmoid(self.conv5(out))

        return out
