import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks
        self.cover_dependent = config.cover_dependent
        # self.residual = config.residual

        if self.cover_dependent == 1:
            layers = [ConvBNRelu(3, self.conv_channels)]
        else:
            layers = [ConvBNRelu(config.message_length, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        if self.cover_dependent == 1:
            self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)
        else:
            self.after_concat_layer = ConvBNRelu(config.message_length,
                                             self.conv_channels)            

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)
        self.final_tanh = nn.Tanh()
        self.factor = 10/255

    def forward(self, image, message):

        # # First, add two dummy dimensions in the end of the message.
        # # This is required for the .expand to work correctly
        # expanded_message = message.unsqueeze(-1)
        # expanded_message.unsqueeze_(-1)

        # expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        # encoded_image = self.conv_layers(image)
        # # concatenate expanded message and image
        # concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        # im_w = self.after_concat_layer(concat)
        # im_w = self.final_layer(im_w)

        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        if self.cover_dependent:
            encoded_image = self.conv_layers(image)
            # concatenate expanded message and image
            concat = torch.cat([expanded_message, encoded_image, image], dim=1)
            im_w = self.after_concat_layer(concat)
            im_w = self.final_layer(im_w)
            # if self.residual:
            #     im_w = self.factor * self.final_tanh(im_w) + image
        else:
            #import pdb; pdb.set_trace()
            #encoded_message = self.conv_layers(expanded_message)
            # concatenate expanded message and image
            #concat = encoded_message # torch.cat([expanded_message, encoded_image, image], dim=1)
            im_w = self.after_concat_layer(expanded_message)
            im_w = self.final_layer(im_w) + image
        return im_w
