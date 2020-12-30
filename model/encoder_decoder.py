import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration

from noise_layers.noiser import Noiser
from noise_layers.identity import Identity
from noise_layers.jpeg_compression2000 import JpegCompression2000
from noise_layers.jpeg_compression2 import JpegCompression2
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.quantization import Quantization
from noise_layers.dropout import Dropout
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.gaussian import Gaussian
from noise_layers.webp import WebP
from noise_layers.mpeg4_compression import MPEG4
from noise_layers.h264_compression import H264
from noise_layers.xvid_compression import XVID
from noise_layers.diff_quality_jpeg_compression2 import DiffQFJpegCompression2
from noise_layers.diff_corruptions import DiffCorruptions


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser

        self.decoder = Decoder(config)

        self.identity = Identity()
        self.dropout = Dropout([0.3, 0.3])
        self.cropout = Cropout([0.547, 0.547], [0.547, 0.547])
        self.crop = Crop([0.187, 0.187], [0.187, 0.187])
        self.gaussian = Gaussian()
        self.jpeg = JpegCompression2()

        self.jpeg_10 = JpegCompression2(quality=10)
        self.jpeg_25 = JpegCompression2(quality=25)
        self.jpeg_50 = JpegCompression2(quality=50)
        self.jpeg_75 = JpegCompression2(quality=75)
        self.jpeg_90 = JpegCompression2(quality=90)

        self.jpeg2000_100 = JpegCompression2000(quality=100)
        self.jpeg2000_250 = JpegCompression2000(quality=250)
        self.jpeg2000_500 = JpegCompression2000(quality=500)
        self.jpeg2000_750 = JpegCompression2000(quality=750)
        self.jpeg2000_900 = JpegCompression2000(quality=900)

        self.webp_10 = WebP(quality=10)
        self.webp_25 = WebP(quality=25)
        self.webp_50 = WebP(quality=50)
        self.webp_75 = WebP(quality=75)
        self.webp_90 = WebP(quality=90)

        self.jpeg2000 = JpegCompression2000()
        self.webp = WebP()
        self.mpeg4 = MPEG4()
        self.h264 = H264()
        self.xvid = XVID()
        self.diff_qf_jpeg2 = DiffQFJpegCompression2()
        self.diff_corruptions = DiffCorruptions()

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        # noised_and_cover = self.noiser([torch.clamp(encoded_image, 0., 1.), image]) # noised_and_cover = self.noiser([encoded_image, image]) # changed by chaoning
        noised_and_cover = self.noiser([encoded_image, image]) # changed by chaoning, to the original (quant does not work anyway)
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message

    def forward_specific_noiser(self, image, message, noiser='identity'):
        encoded_image = self.encoder(image, message)
        # encoded_image = torch.clamp(encoded_image, 0., 1.) # changed by chaoning
        if noiser == 'identity':
            noised_and_cover = self.identity([encoded_image, image])
        elif noiser == 'dropout':
            noised_and_cover = self.dropout([encoded_image, image])
        elif noiser == 'cropout':
            noised_and_cover = self.cropout([encoded_image, image])
        elif noiser == 'crop':
            noised_and_cover = self.crop([encoded_image, image])
        elif noiser == 'gaussian':
            noised_and_cover = self.gaussian([encoded_image, image])
        elif noiser == 'jpeg':
            noised_and_cover = self.jpeg([encoded_image, image])
        elif noiser == 'jpeg_10':
            noised_and_cover = self.jpeg_10([encoded_image, image])
        elif noiser == 'jpeg_25':
            noised_and_cover = self.jpeg_25([encoded_image, image])
        elif noiser == 'jpeg_50':
            noised_and_cover = self.jpeg_50([encoded_image, image])
        elif noiser == 'jpeg_75':
            noised_and_cover = self.jpeg_75([encoded_image, image])
        elif noiser == 'jpeg_90':
            noised_and_cover = self.jpeg_90([encoded_image, image])
        elif noiser == 'jpeg2000':
            noised_and_cover = self.jpeg2000([encoded_image, image])
        elif noiser == 'jpeg2000_100':
            noised_and_cover = self.jpeg2000_100([encoded_image, image])
        elif noiser == 'jpeg2000_250':
            noised_and_cover = self.jpeg2000_250([encoded_image, image])
        elif noiser == 'jpeg2000_500':
            noised_and_cover = self.jpeg2000_500([encoded_image, image])
        elif noiser == 'jpeg2000_750':
            noised_and_cover = self.jpeg2000_750([encoded_image, image])
        elif noiser == 'jpeg2000_900':
            noised_and_cover = self.jpeg2000_900([encoded_image, image])
        elif noiser == 'webp':
            noised_and_cover = self.webp([encoded_image, image])
        elif noiser == 'webp_10':
            noised_and_cover = self.webp_10([encoded_image, image])
        elif noiser == 'webp_25':
            noised_and_cover = self.webp_25([encoded_image, image])
        elif noiser == 'webp_50':
            noised_and_cover = self.webp_50([encoded_image, image])
        elif noiser == 'webp_75':
            noised_and_cover = self.webp_75([encoded_image, image])
        elif noiser == 'webp_90':
            noised_and_cover = self.webp_90([encoded_image, image])
        elif noiser == 'mpeg4':
            noised_and_cover = self.mpeg4([encoded_image, image])
        elif noiser == 'h264':
            noised_and_cover = self.h264([encoded_image, image])
        elif noiser == 'xvid':
            noised_and_cover = self.xvid([encoded_image, image])
        elif noiser == 'diff_qf_jpeg2':
            noised_and_cover = self.diff_qf_jpeg2([encoded_image, image])
        elif noiser == 'diff_corruptions':
            noised_and_cover = self.diff_corruptions([encoded_image, image])

        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message