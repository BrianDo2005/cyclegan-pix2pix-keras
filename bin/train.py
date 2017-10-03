import os
import argparse
import datetime

from cyclegan_keras.cyclegan import CycleGAN
from cyclegan_keras.generators import ImageFileGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required')
    required.add_argument('--dataroot', required=True,
                          help='path to images (should have subfolders trainA, trainB)')
    required.add_argument('--model-dir', type=str, required=True,
                          help=('Directory where model definition files are saved during training. Models will be '
                                'saved as `model_dir`/`experiment_name`_{G_A,G_B,D_A,D_B}_epoch##.h5'))
    required.add_argument('--experiment-name', type=str, 
                          default='cyclegan_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                          help=('Experiment name for use in naming model definition files. Models will be saved '
                                'as `model_dir`/`experiment_name`_{G_A,G_B,D_A,D_B}_epoch##.h5. Defaults to '
                                '"cyclegan_{CURRENT_DATE_TIME}".'))
    
    images = parser.add_argument_group('images')
    images.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    images.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    images.add_argument('--image-size', type=int, default=128,
                        help='size of input images (not used if `crop_images` or resize_images` is set)')
    images.add_argument('--resize-images', action='store_true', default=False,
                        help='resize images to the `scale_size` option')
    images.add_argument('--crop-images', action='store_true', default=False,
                        help='randomly crop images to the `crop_size` option')
    images.add_argument('--flip-images', action='store_true', default=False,
                        help='randomly flip images in the x direction')
    images.add_argument('--scale-size', type=int, default=286, help='scale images to this size')
    images.add_argument('--crop-size', type=int, default=256, help='then crop to this size')
    
    model = parser.add_argument_group('model')
    model.add_argument('--init-filters-gen', type=int, default=64, help='# of gen filters in first conv layer')
    model.add_argument('--init-filters-dis', type=int, default=64, help='# of discrim filters in first conv layer')
    model.add_argument('--generator-name', type=str, default='resnet_9blocks',
                       choices=['resnet_6blocks', 'resnet_9blocks', 'unet128', 'unet256'],
                       help='selects model to use for generator network')
    model.add_argument('--num-layers-dis', type=int, default=3,
                       help='number of layers in discriminator network (PatchGAN)')
    model.add_argument('--norm-method', type=str, choices=['instance', 'batch'], default='instance',
                       help='normalization method (instance or batch normalization)')
    model.add_argument('--deconv-method', type=str, choices=['deconv', 'upsample'], default='upsample',
                       help='method to use for deconvolution. "deconv" uses the Conv2DTranspose layer, '
                            'while "upsample" uses Upsampling2D followed by Conv2D')
    model.add_argument('--no-dropout', action='store_true', default=False, help='no dropout for the generator')
    
    training = parser.add_argument_group('training')
    training.add_argument('--n-epochs', type=int, default=100, help='# of epochs at starting learning rate')
    training.add_argument('--n-epochs-decay', type=int, default=100,
                          help='# of epochs to linearly decay learning rate to zero')
    training.add_argument('--steps-per-epoch', type=int, default=10000,
                          help='number of steps (batches) to run for each epoch')
    training.add_argument('--batch-size', type=int, default=1, help='number of images used in each step (batch)')
    training.add_argument('--starting-epoch', type=int, default=1,
                          help='(the starting epoch count, we save the model by <starting_epoch>, '
                               '<starting_epoch>+<save_latest_freq>, ...')
    training.add_argument('--pool_size', type=int, default=50,
                          help=('the size of image buffer that stores previously generated images '
                                'for discrimnator training'))
    training.add_argument('--pretrain-iter', type=int, default=1000,
                          help=('the number of pretraining batches to run on the adversarial model'
                                ' before starting GAN'))
    training.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    training.add_argument('--learning-rate', type=float, default=0.0002, help='initial learning rate for adam')
    training.add_argument('--no-lsgan', action='store_true', default=False,
                          help='do *not* use least square GAN, if selected, use vanilla GAN')
    training.add_argument('--lambda-a', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    training.add_argument('--lambda-b', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    training.add_argument('--use-identity-loss', action='store_true', default=False,
                          help='Use identity mapping to ensure that G_A(B) produces B.')
    training.add_argument('--lambda-id', type=float, default=0.0,
                          help=('Identity mapping weight. Setting lambda_id other than 1 has an effect of scaling '
                                'the weight of the identity mapping loss. This weight is combined with lambda_a and '
                                'lambda_b to create the total identity weight for each direction '
                                '(lambda_id_a = lambda_a * lambda_id). For example, if the weight of '
                                'the identity loss should be 10 times smaller than the weight of the '
                                'reconstruction loss, please set `lambda_id` = 0.1'))
    training.add_argument('--stacked-training', action='store_true', default=False,
                          help='use stacked training, freezing the adversarial model during joint model training')
    training.add_argument('--continue-training', action='store_true', default=False,
                          help=('Continue training? Loads models from `model_dir` to continue training. '
                                'Loading an existing model overwrites any options in the `model` group.'))
    training.add_argument('--experiment-to-load', type=str,
                          help='Experiment name to load (if different from `experiment_name`)')
    training.add_argument('--which-epoch', type=str, default='latest',
                          help='Which epoch to load? Set to "latest" to use latest saved model')
    training.add_argument('--print-freq', type=int, default=100,
                          help='frequency of showing training results on console (in steps/batches)')
    training.add_argument('--save-epoch-freq', type=int, default=5,
                          help='frequency of saving models at the end of epochs')
    
    args = parser.parse_args()

    args.image_size = (args.image_size, args.image_size)
    if args.resize_images:
        args.image_size = (args.scale_size, args.scale_size)
        args.scale_size = (args.scale_size, args.scale_size)
    else:
        args.scale_size = None
    if args.crop_images:
        args.image_size = (args.crop_size, args.crop_size)
        args.crop_size = (args.crop_size, args.crop_size)
    else:
        args.crop_size = None
        
    if args.experiment_to_load is None:
        args.experiment_to_load = args.experiment_name
        
    img_generator_a = ImageFileGenerator(os.path.join(args.dataroot, 'trainA'), args.scale_size, args.crop_size,
                                         args.flip_images)
    img_generator_b = ImageFileGenerator(os.path.join(args.dataroot, 'trainB'), args.scale_size, args.crop_size,
                                         args.flip_images)
        
    cyclegan_model = CycleGAN(args.image_size, args.input_nc, args.output_nc, args.generator_name, args.num_layers_dis,
                              args.init_filters_gen, args.init_filters_dis, not args.no_lsgan, not args.no_dropout,
                              args.norm_method, args.deconv_method, args.learning_rate, args.beta1, args.lambda_a,
                              args.lambda_b, args.use_identity_loss, args.lambda_id, args.stacked_training,
                              args.continue_training, args.model_dir, args.experiment_to_load, args.which_epoch)
    cyclegan_model.connect_inputs(img_generator_a, img_generator_b)
    cyclegan_model.fit(args.model_dir, args.experiment_name, args.batch_size, args.pool_size, args.n_epochs,
                       args.n_epochs_decay, args.steps_per_epoch, args.pretrain_iter, args.save_epoch_freq,
                       args.print_freq, args.starting_epoch)
