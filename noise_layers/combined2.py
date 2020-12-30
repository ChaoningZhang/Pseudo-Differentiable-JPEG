import torch
import torch.nn as nn
import numpy as np
import math
from noise_layers.identity import Identity
from noise_layers.jpeg_compression2 import JpegCompression2
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.dropout import Dropout
from noise_layers.crop2 import Crop2
from noise_layers.cropout import Cropout
from noise_layers.gaussian import Gaussian

class Combined2(nn.Module):
    """
    Combined noise
    """
    def __init__(self, device, jpeg_type='jpeg2'):
        super(Combined2, self).__init__()
        self.identity = Identity()
        if jpeg_type == 'jpeg2':
            self.jpeg = JpegCompression2()
        elif jpeg_type == 'jpeg':
            self.jpeg = JpegCompression(device)
        elif jpeg_type == 'diff_jpeg':
            self.jpeg = DiffJPEG()
        else:
            self.jpeg = Quantization()
        self.dropout = Dropout([0.3, 0.3])
        self.gaussian = Gaussian()
        self.crop2 = Crop2([0.187, 0.187], [0.187, 0.187]) # Crop2([0.547, 0.547], [0.547, 0.547])
        self.cropout = Cropout([0.547, 0.547], [0.547, 0.547])

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]

        N, _, _, _ = noised_image.shape

        n0, c0 = noised_and_cover[0][:N//6], noised_and_cover[1][:N//6]
        n1, c1 = noised_and_cover[0][N//6:2*N//6], noised_and_cover[1][N//6:2*N//6]
        n2, c2 = noised_and_cover[0][2*N//6:3*N//6], noised_and_cover[1][2*N//6:3*N//6]
        n3, c3 = noised_and_cover[0][3*N//6:4*N//6], noised_and_cover[1][3*N//6:4*N//6]
        n4, c4 = noised_and_cover[0][4*N//6:5*N//6], noised_and_cover[1][4*N//6:5*N//6]
        n5, c5 = noised_and_cover[0][5*N//6:], noised_and_cover[1][5*N//6:]

        nc0 = self.identity([n0, c0])
        nc1 = self.jpeg([n1, c1])
        nc2 = self.dropout([n2, c2])
        nc3 = self.gaussian([n3, c3])
        nc4 = self.crop2([n4, c4])
        nc5 = self.cropout([n5, c5])
                
        # noised_new = torch.cat((nc0[0], nc1[0], nc2[0], nc3[0]), 0)
        # cover_new = torch.cat((nc0[1], nc1[1], nc2[1], nc3[1]), 0)
        noised_new = torch.cat((nc0[0], nc1[0], nc2[0], nc3[0], nc4[0], nc5[0]), 0)
        cover_new = torch.cat((nc0[1], nc1[1], nc2[1], nc3[1], nc4[1], nc5[1]), 0)

        return [noised_new, cover_new]