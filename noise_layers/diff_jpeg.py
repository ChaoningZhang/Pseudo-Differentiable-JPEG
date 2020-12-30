import torch
import torch.nn as nn
# Local
from modules.compression import compress_jpeg
from modules.decompression import decompress_jpeg

def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3

def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.

class DiffJPEG(nn.Module):
    def __init__(self, height=128, width=128, differentiable=True, quality=50):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image height
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        '''
        '''
        x0, x1 = x
        y, cb, cr = self.compress(x0)
        recovered = self.decompress(y, cb, cr)
        return [recovered, x1]