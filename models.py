# Model architectures for generators and discriminators

# Torch imports
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

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
class CycleGenerator(nn.Module):
    def __init__(self, init_zero_weights=False):
        super(CycleGenerator, self).__init__()

        ####   GENERATOR ARCHITECTURE   ####

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0, reflect_pad=True)
        self.conv2 = conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2) #prev kernel_size = 3, padding = 1
        self.conv3 = conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        # 2. Define the transformation part of the generator
        self.resnet_block1 = ResnetBlock2d(conv_dim=256)
        self.resnet_block2 = ResnetBlock2d(conv_dim=256)
        self.resnet_block3 = ResnetBlock2d(conv_dim=256)
        self.resnet_block4 = ResnetBlock2d(conv_dim=256)
        self.resnet_block5 = ResnetBlock2d(conv_dim=256)
        self.resnet_block6 = ResnetBlock2d(conv_dim=256)
        self.resnet_block7 = ResnetBlock2d(conv_dim=256)
        self.resnet_block8 = ResnetBlock2d(conv_dim=256)
        self.resnet_block9 = ResnetBlock2d(conv_dim=256)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv2d_1 = deconv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2d_2 = deconv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1) #prev kernel_size = 3, padding = 1
        self.conv4 = conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0, reflect_pad=True, instance_norm=False)

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
        out = F.relu(self.resnet_block7(out))
        out = F.relu(self.resnet_block8(out))
        out = F.relu(self.resnet_block9(out))

        out = F.relu(self.deconv2d_1(out))
        out = F.relu(self.deconv2d_2(out))
        out = F.tanh(self.conv4(out))

        return out


#XNet encoder
class XNetEncoder(nn.Module):
    def __init__(self, init_zero_weights=False):
        super(XNetEncoder2d, self).__init__()

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0, reflect_pad=True)
        self.conv2 = conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        # 2. Define the transformation part of the generator
        self.resnet_block1 = ResnetBlock2d(conv_dim=256)
        self.resnet_block2 = ResnetBlock2d(conv_dim=256)
        self.resnet_block3 = ResnetBlock2d(conv_dim=256)
        self.resnet_block4 = ResnetBlock2d(conv_dim=256)
        self.resnet_block5 = ResnetBlock2d(conv_dim=256)
        self.resnet_block6 = ResnetBlock2d(conv_dim=256)
        self.resnet_block7 = ResnetBlock2d(conv_dim=256)
        self.resnet_block8 = ResnetBlock2d(conv_dim=256)
        self.resnet_block9 = ResnetBlock2d(conv_dim=256)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = F.relu(self.resnet_block1(out))
        out = F.relu(self.resnet_block2(out))
        out = F.relu(self.resnet_block3(out))
        out = F.relu(self.resnet_block4(out))
        out = F.relu(self.resnet_block5(out))
        out = F.relu(self.resnet_block6(out))
        out = F.relu(self.resnet_block7(out))
        out = F.relu(self.resnet_block8(out))
        out = F.relu(self.resnet_block9(out))

        return out


#XNet decoder
class XNetDecoder(nn.Module):
    def __init__(self, init_zero_weights=False):
        super(XNetDecoder2d, self).__init__()

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv2d_1 = deconv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2d_2 = deconv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=0, reflect_pad=True, instance_norm=False)

    def forward(self, x):
        out = F.relu(self.deconv2d_1(x))
        out = F.relu(self.deconv2d_2(out))
        out = F.tanh(self.conv4(out))

        return out

#XNet translator
class XNetTranslator(nn.Module):
    def __init__(self, init_zero_weights=False):
        super(XNetTranslator2d, self).__init__()

        # 2. Define the transformation part of the generator
        self.resnet_block1 = ResnetBlock2d(conv_dim=256)
        self.resnet_block2 = ResnetBlock2d(conv_dim=256)
        self.resnet_block3 = ResnetBlock2d(conv_dim=256)
        self.resnet_block4 = ResnetBlock2d(conv_dim=256)
        self.resnet_block5 = ResnetBlock2d(conv_dim=256)
        self.resnet_block6 = ResnetBlock2d(conv_dim=256)
        self.resnet_block7 = ResnetBlock2d(conv_dim=256)
        self.resnet_block8 = ResnetBlock2d(conv_dim=256)
        self.resnet_block9 = ResnetBlock2d(conv_dim=256)

    def forward(self, x):
        out = F.relu(self.resnet_block1(x))
        out = F.relu(self.resnet_block2(out))
        out = F.relu(self.resnet_block3(out))
        out = F.relu(self.resnet_block4(out))
        out = F.relu(self.resnet_block5(out))
        out = F.relu(self.resnet_block6(out))
        out = F.relu(self.resnet_block7(out))
        out = F.relu(self.resnet_block8(out))
        out = F.relu(self.resnet_block9(out))

        return out

"""Defines the architecture of the discriminator network (both discriminators D_X and D_Y have the same architecture)."""
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator2d, self).__init__()

        #### ARCHITECTURE ####
        self.conv1 = conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, instance_norm=False)
        self.conv2 = conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv5 = conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, instance_norm=False)


    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2, inplace=True)
        out = F.sigmoid(self.conv5(out))

        return out
