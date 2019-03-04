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
                x: batch_size x 3 x depth x height x width

            Output
            ------
                out: batch_size x 3 x depth x height x width
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

        out = F.relu(self.deconv3d_1(out))
        out = F.relu(self.deconv3d_2(out))
        out = F.tanh(self.conv4(out))

        return out


"""Defines the architecture of the discriminator network (both discriminators D_X and D_Y have the same architecture)."""
class PatchGANDiscriminator3d(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        #### ARCHITECTURE ####
        self.conv1 = conv3d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, instance_norm=False)
        self.conv2 = conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv5 = conv3d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, instance_norm=False)


    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2, inplace=True)
        out = F.sigmoid(self.conv5(out))

        return out
