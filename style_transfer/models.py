# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator       --> Used in the vanilla GAN in Part 1
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)
#
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ methods in the DCGenerator, CycleGenerator, and DCDiscriminator classes.
# Note that the forward passes of these models are provided for you, so the only part you need to
# fill in is __init__.

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=0, instance_norm=True, reflect_pad=False):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False))
   
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
 
    if reflect_pad:
	layers.append(nn.ReflectionPad2d(3)) 
  
   return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, instance_norm=True, init_zero_weights=False, reflect_pad=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    
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




class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, init_zero_weights=False):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0, reflect_pad=True)  
        self.conv2 = conv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) 
        self.conv3 = conv(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1) 

        # 2. Define the transformation part of the generator
        self.resnet_block1 = ResnetBlock(conv_dim=256) 
        self.resnet_block2 = ResnetBlock(conv_dim=256)
        self.resnet_block3 = ResnetBlock(conv_dim=256)
        self.resnet_block4 = ResnetBlock(conv_dim=256)
        self.resnet_block5 = ResnetBlock(conv_dim=256)
        self.resnet_block6 = ResnetBlock(conv_dim=256) 

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv1 = deconv(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.deconv2 = deconv(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) 
    
        self.conv4 = conv(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0, reflect_pad=True, instance_norm=False)

    def forward(self, x):
        """Generates an image conditioned
           on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
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

        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.tanh(self.deconv3(out))

        return out




#only here for transcription purposes

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)



class PatchGANDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self):
        super(DCDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        #original architecture
        #self.conv1 = conv(in_channels=3, out_channels=32, kernel_size=4)
        #self.conv2 = conv(in_channels=32, out_channels=64, kernel_size=4)        
        #self.conv3 = conv(in_channels=64, out_channels=128, kernel_size=4)
        #self.conv4 = conv(in_channels=128, out_channels=1, kernel_size=4, padding=0, batch_norm=False)


        
        self.conv1 = conv(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, instance_norm=False)    
        self.conv2 = conv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  
        self.conv3 = conv(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) 
        self.conv4 = conv(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)        
        self.conv5 = conv(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, instance_norm=False)     

       
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2, inplace=True)
        out = self.conv5(out)

        return out
