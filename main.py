import os
import pprint
import argparse
import torch
import pickle
import utils
import logging
import sys
import socket
import numpy as np

from options import *
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from noise_argparser import NoiseArgParser

from train import train, train_other_noises, train_own_noise


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')
    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--data-dir', '-d', required=True, type=str,
                                help='The directory where the data is stored.')
    new_run_parser.add_argument('--batch-size', '-b', required=True, type=int, help='The batch size.')
    new_run_parser.add_argument('--epochs', '-e', default=300, type=int, help='Number of epochs to run the simulation.')
    new_run_parser.add_argument('--name', required=True, type=str, help='The name of the experiment.')
    new_run_parser.add_argument('--adv_loss', default=0, required=False, type=float, help='Coefficient of the adversarial loss.')
    new_run_parser.add_argument('--residual', default=0, required=False, type=int, help='If to use residual or not.')
    new_run_parser.add_argument('--video_dataset', default=0, required=False, type=int, help='If to use video dataset or not.')
    new_run_parser.add_argument('--save-dir', '-sd', default='runs', required=True, type=str,
                                help='The save directory where the result is stored.')

    new_run_parser.add_argument('--size', '-s', default=128, type=int,
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are starting a new experiment.')
    # parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
    #                     help='If specified, use adds a Tensorboard log. On by default')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')

    new_run_parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                                help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")
    new_run_parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
    new_run_parser.add_argument('--cover-dependent', default=1, required=False, type=int, help='If to use cover dependent architecture or not.')
    new_run_parser.add_argument('--jpeg_type', '-j', required=False, type=str, default='jpeg',
                                 help='Jpeg type used in the combined2 noise.')

    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--folder', '-f', required=True, type=str,
                                 help='Continue from the last checkpoint in this folder.')
    continue_parser.add_argument('--data-dir', '-d', required=False, type=str,
                                 help='The directory where the data is stored. Specify a value only if you want to override the previous value.')
    continue_parser.add_argument('--epochs', '-e', required=False, type=int,
                                help='Number of epochs to run the simulation. Specify a value only if you want to override the previous value.')

    # continue_parser.add_argument('--tensorboard', action='store_true',
    #                             help='Override the previous setting regarding tensorboard logging.')
    
    # Setting up a seed for debug
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)

    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None
    print(args.cover_dependent)

    if not args.video_dataset:
        if args.hostname == 'ee898-System-Product-Name':
            args.data_dir = '/home/ee898/Desktop/chaoning/ImageNet'
            args.hostname = 'ee898'
        elif args.hostname == 'DL178':
            args.data_dir = '/media/user/SSD1TB-2/ImageNet' 
        else:
            args.data_dir = '/workspace/data_local/imagenet_pytorch' 
    else:
        if args.hostname == 'ee898-System-Product-Name':
            args.data_dir = '/home/ee898/Desktop/chaoning/ImageNet'
            args.hostname = 'ee898'
        elif args.hostname == 'DL178':
            args.data_dir = '/media/user/SSD1TB-2/ImageNet' 
        else:
            args.data_dir = './oops_dataset/oops_video' 
    assert args.data_dir

    if args.command == 'continue':
        this_run_folder = args.folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
        train_options.start_epoch = checkpoint['epoch'] + 1
        if args.data_dir is not None:
            train_options.train_folder = os.path.join(args.data_dir, 'train')
            train_options.validation_folder = os.path.join(args.data_dir, 'val')
        if args.epochs is not None:
            if train_options.start_epoch < args.epochs:
                train_options.number_of_epochs = args.epochs
            else:
                print(f'Command-line specifies of number of epochs = {args.epochs}, but folder={args.folder} '
                      f'already contains checkpoint for epoch = {train_options.start_epoch}.')
                exit(1)

    else:
        assert args.command == 'new'
        start_epoch = 1
        train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', args.save_dir),
            start_epoch=start_epoch,
            experiment_name=args.name,
            video_dataset=args.video_dataset)


        noise_config = args.noise if args.noise is not None else []
        hidden_config = HiDDenConfiguration(H=args.size, W=args.size,
                                            message_length=args.message,
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            use_discriminator=True,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,
                                            encoder_loss=0.7,
                                            adversarial_loss=args.adv_loss,
                                            cover_dependent=args.cover_dependent,
                                            residual=args.residual,
                                            enable_fp16=args.enable_fp16
                                            )

        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(hidden_config, f)


    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    if (args.command == 'new' and args.tensorboard) or \
            (args.command == 'continue' and os.path.isdir(os.path.join(this_run_folder, 'tb-logs'))):
        logging.info('Tensorboard is enabled. Creating logger.')
        from tensorboard_logger import TensorBoardLogger
        tb_logger = TensorBoardLogger(os.path.join(this_run_folder, 'tb-logs'))
    else:
        tb_logger = None


    noiser = Noiser(noise_config, device, args.jpeg_type)
    model = Hidden(hidden_config, device, noiser, tb_logger)

    if args.command == 'continue':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    logging.info('HiDDeN model: {}\n'.format(model.to_stirng()))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(hidden_config)))
    logging.info('\nNoise configuration:\n')
    logging.info(pprint.pformat(str(noise_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    # train(model, device, hidden_config, train_options, this_run_folder, tb_logger)
    # train_other_noises(model, device, hidden_config, train_options, this_run_folder, tb_logger)
    if str(args.noise[0]) == "WebP()":
        noise = 'webp'
    elif str(args.noise[0]) == "JpegCompression2000()":
        noise = 'jpeg2000'
    elif str(args.noise[0]) == "MPEG4()":
        noise = 'mpeg4'
    elif str(args.noise[0]) == "H264()":
        noise = 'h264'
    elif str(args.noise[0]) == "XVID()":
        noise = 'xvid'
    elif str(args.noise[0]) == "DiffQFJpegCompression2()":
        noise = 'diff_qf_jpeg2'
    elif str(args.noise[0]) == "DiffCorruptions()":
        noise = 'diff_corruptions'
    else:
        noise = 'jpeg'
    train_own_noise(model, device, hidden_config, train_options, this_run_folder, tb_logger, noise)

if __name__ == '__main__':
    main()