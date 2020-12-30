import argparse
import re
from noise_layers.cropout import Cropout
from noise_layers.crop import Crop
from noise_layers.identity import Identity
from noise_layers.dropout import Dropout
from noise_layers.resize import Resize
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.jpeg_compression2 import JpegCompression2
from noise_layers.jpeg_compression2000 import JpegCompression2000
from noise_layers.gaussian import Gaussian
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.webp import WebP
from noise_layers.mpeg4_compression import MPEG4
from noise_layers.h264_compression import H264
from noise_layers.xvid_compression import XVID
from noise_layers.diff_quality_jpeg_compression2 import DiffQFJpegCompression2
from noise_layers.diff_corruptions import DiffCorruptions


def parse_pair(match_groups):
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)

def parse_crop(crop_command):
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))

def parse_cropout(cropout_command):
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))


def parse_dropout(dropout_command):
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))

def parse_resize(resize_command):
    matches = re.match(r'resize\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))


class NoiseArgParser(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )

    @staticmethod
    def parse_cropout_args(cropout_args):
        pass

    @staticmethod
    def parse_dropout_args(dropout_args):
        pass

    def __call__(self, parser, namespace, values,
                 option_string=None):

        layers = []
        split_commands = values[0].split('+')

        for command in split_commands:
            # remove all whitespace
            command = command.replace(' ', '')
            if command[:len('cropout')] == 'cropout':
                layers.append(parse_cropout(command))
            elif command[:len('diff_qf_jpeg2')] == 'diff_qf_jpeg2':
                layers.append(DiffQFJpegCompression2())
            elif command[:len('diff_corruptions')] == 'diff_corruptions':
                layers.append(DiffCorruptions())
            elif command[:len('crop')] == 'crop':
                layers.append(parse_crop(command))
            elif command[:len('dropout')] == 'dropout':
                layers.append(parse_dropout(command))
            elif command[:len('resize')] == 'resize':
                layers.append(parse_resize(command))
            elif command[:len('jpeg2000')] == 'jpeg2000':
                layers.append(JpegCompression2000())
            elif command[:len('jpeg2')] == 'jpeg2':
                layers.append(JpegCompression2())
            elif command[:len('jpeg')] == 'jpeg':
                layers.append('JpegPlaceholder')
            elif command[:len('quant')] == 'quant':
                layers.append('QuantizationPlaceholder')
            elif command[:len('combined2')] == 'combined2':
                layers.append('Combined2Placeholder')
            elif command[:len('identity')] == 'identity':
                layers.append(Identity())
            elif command[:len('gaussian4')] == 'gaussian4':
                layers.append(Gaussian(3,4,3))
            elif command[:len('gaussian')] == 'gaussian':
                layers.append(Gaussian())
            elif command[:len('diff_jpeg')] == 'diff_jpeg':
                layers.append(DiffJPEG(128,128))
            elif command[:len('webp')] == 'webp':
                layers.append(WebP())
            elif command[:len('mpeg4')] == 'mpeg4':
                layers.append(MPEG4())
            elif command[:len('h264')] == 'h264':
                layers.append(H264())
            elif command[:len('xvid')] == 'xvid':
                layers.append(XVID())
            else:
                raise ValueError('Command not recognized: \n{}'.format(command))
        setattr(namespace, self.dest, layers)
