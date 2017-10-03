from __future__ import print_function, division

import math

import numpy as np

from keras.models import Model, Input
from keras.layers import (Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, concatenate,
                          Activation, Add, UpSampling2D)
from keras.initializers import Zeros, RandomNormal

from .utils import get_channel_axis, get_input_shape


def get_norm_layer(layer_name):
    if layer_name == 'batch':
        return BatchNormalization(axis=get_channel_axis(), gamma_initializer=RandomNormal(1.0, 0.02),
                                  beta_initializer=Zeros(), moving_mean_initializer=RandomNormal(1.0, 0.02),
                                  moving_variance_initializer=Zeros())
    elif layer_name == 'instance':
        try:
            from keras_contrib.layers import InstanceNormalization
        except ImportError:
            raise ImportError('keras_contrib is required to use InstanceNormalization layers. Install keras_contrib or '
                              'switch normalization to "batch".')
        return InstanceNormalization()
    else:
        return NotImplementedError('Normalization layer name [%s] is not recognized.' % layer_name)


def get_conv_initialiers():
    return RandomNormal(0.0, 0.2), Zeros()


def build_generator_model(image_size, input_nc, output_nc, init_num_filters, model_name, norm_layer='batch',
                          use_dropout=False):
    
    if model_name == 'unet_128':
        gen_model = build_unet(image_size, input_nc, output_nc, init_num_filters, norm_layer=norm_layer,
                               n_levels=7, use_dropout=use_dropout)
    elif model_name == 'unet_256':
        gen_model = build_unet(image_size, input_nc, output_nc, init_num_filters, norm_layer=norm_layer,
                               n_levels=8, use_dropout=use_dropout)
    elif model_name == 'resnet_6blocks':
        gen_model = build_resnet(image_size, input_nc, output_nc, init_num_filters, norm_layer=norm_layer,
                                 use_dropout=use_dropout, n_blocks=6)
    elif model_name == 'resnet_9blocks':
        gen_model = build_resnet(image_size, input_nc, output_nc, init_num_filters, norm_layer=norm_layer,
                                 use_dropout=use_dropout, n_blocks=9)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % model_name)

    return gen_model


def build_discriminator_model(image_size, input_nc, init_num_filters, n_layers=3, norm_layer='batch',
                              use_sigmoid=False):
    
    dis_model = build_nlayer_discriminator(image_size, input_nc, init_num_filters, n_layers, norm_layer,
                                           use_sigmoid)

    return dis_model


def build_unet(image_size, input_nc, output_nc, init_num_filters, norm_layer='batch', n_levels=7, use_dropout=False,
               dropout_rate=0.5, autoencoder=True):

    kernel_size = (4, 4)
    conv_kernel_init, conv_bias_init = get_conv_initialiers()
    input_shape = get_input_shape(image_size, input_nc)
    use_bias = norm_layer == 'instance'
    nodes_for_concat = []

    # construct unet structure from bottom up
    input_img = Input(shape=input_shape, name='input1')
    prev = Conv2D(init_num_filters, kernel_size, strides=(2, 2), padding='same', use_bias=use_bias,
                  kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(input_img)
    nodes_for_concat.append(prev)
    
    for i in range(1, n_levels - 1):
        relu = LeakyReLU(0.2)(prev)
        conv = Conv2D(init_num_filters * max(2 ** i, 8), kernel_size, strides=(2, 2), padding='same',
                      use_bias=use_bias, kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(relu)
        prev = get_norm_layer(norm_layer)(conv)
        nodes_for_concat.append(prev)
        
    relu = LeakyReLU(0.2)(prev)
    conv = Conv2D(init_num_filters * max(2 ** n_levels - 1, 8), kernel_size, strides=(2, 2), padding='same',
                  use_bias=use_bias, kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(relu)
    relu = Activation('relu')(conv)
    # deconv = Conv2DTranspose(init_num_filters * max(2 ** n_levels - 2, 8), kernel_size, strides=(2, 2),
    #                          padding='same', use_bias=use_bias, kernel_initializer=conv_kernel_init,
    #                          bias_initializer=conv_bias_init)(relu)
    up = UpSampling2D(relu)
    deconv = Conv2D(init_num_filters * max(2 ** n_levels - 2, 8), kernel_size, padding='same',
                    use_bias=use_bias, kernel_initializer=conv_kernel_init,
                    bias_initializer=conv_bias_init)(up)

    for i in reversed(range(1, n_levels - 1)):
        norm = get_norm_layer(norm_layer)(deconv)
        if use_dropout:
            norm = Dropout(dropout_rate)(norm)
        if not autoencoder:
            norm = concatenate(norm, [nodes_for_concat[i]], axis=get_channel_axis())
        relu = Activation('relu')(norm)
        # deconv = Conv2DTranspose(init_num_filters * max(2 ** i, 8), kernel_size, strides=(2, 2),
        #                          padding='same', use_bias=use_bias, kernel_initializer=conv_kernel_init,
        #                          bias_initializer=conv_bias_init)(relu)
        up = UpSampling2D(relu)
        deconv = Conv2D(init_num_filters * max(2 ** i, 8), kernel_size, padding='same',
                        use_bias=use_bias, kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(up)
            
    norm = get_norm_layer(norm_layer)(deconv)
    if not autoencoder:
        norm = concatenate([norm, nodes_for_concat[0]], axis=get_channel_axis())
    relu = Activation('relu')(norm)
    conv = Conv2D(output_nc, kernel_size, strides=(2, 2), padding='same', use_bias=use_bias,
                  kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(relu)
    act = Activation('tanh')(conv)
    model = Model(inputs=input_img, outputs=act)

    return model


def build_resnet(image_size, input_nc, output_nc, init_num_filters, norm_layer='batch', padding_layer='zero',
                 n_blocks=6, use_dropout=False, dropout_rate=0.5):
    outer_kernel_size = (7, 7)
    outer_padding_size = ((int(math.ceil((outer_kernel_size[0] - 1) / 2)),
                           int(math.floor((outer_kernel_size[0] - 1) / 2))),
                          (int(math.ceil((outer_kernel_size[1] - 1) / 2)),
                           int(math.floor((outer_kernel_size[1] - 1) / 2))))
    kernel_size = (3, 3)
    conv_kernel_init, conv_bias_init = get_conv_initialiers()
    input_shape = get_input_shape(image_size, input_nc)
    use_bias = norm_layer == 'instance'

    # TODO Implement 2DReflectionPadding (CycleGAN-keras) has partial code, use tf.pad
    if padding_layer == 'zero':
        padding_layer = ZeroPadding2D
    else:
        return NotImplementedError('Only zero padding is currently supported.')
    
    input_img = Input(shape=input_shape, name='input1')
    pad = padding_layer(outer_padding_size)(input_img)
    conv = Conv2D(init_num_filters, outer_kernel_size, use_bias=use_bias, kernel_initializer=conv_kernel_init,
                  bias_initializer=conv_bias_init)(pad)
    norm = get_norm_layer(norm_layer)(conv)
    act = Activation('relu')(norm)
    
    n_downsamples = 2
    for i in range(n_downsamples):
        conv = Conv2D(init_num_filters * (2**(i+1)), kernel_size, strides=(2, 2), padding='same',
                      kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(act)
        norm = get_norm_layer(norm_layer)(conv)
        act = Activation('relu')(norm)
    
    for i in range(n_blocks):
        act = build_conv_block(act, init_num_filters * (2**n_downsamples), kernel_size, norm_layer, padding_layer,
                               use_dropout, dropout_rate, use_bias)
    
    for i in reversed(range(n_downsamples)):
        # deconv = Conv2DTranspose(init_num_filters * (2**i), kernel_size, strides=(2, 2), padding='same',
        #                          kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(act)
        up = UpSampling2D()(act)
        deconv = Conv2D(init_num_filters * (2 ** i), kernel_size, padding='same',
                        kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(up)
        norm = get_norm_layer(norm_layer)(deconv)
        act = Activation('relu')(norm)

    pad = padding_layer(outer_padding_size)(act)
    conv = Conv2D(output_nc, outer_kernel_size, use_bias=use_bias, kernel_initializer=conv_kernel_init,
                  bias_initializer=conv_bias_init)(pad)
    act = Activation('tanh')(conv)

    model = Model(inputs=input_img, outputs=act)

    return model
    
    
def build_conv_block(previous, num_filters, kernel_size, norm_layer, padding_layer, use_dropout,
                     dropout_rate, use_bias):
    conv_kernel_init, conv_bias_init = get_conv_initialiers()
    pad = padding_layer(((1, 1), (1, 1)))(previous)
    conv = Conv2D(num_filters, kernel_size, use_bias=use_bias, kernel_initializer=conv_kernel_init,
                  bias_initializer=conv_bias_init)(pad)
    norm = get_norm_layer(norm_layer)(conv)
    act = Activation('relu')(norm)

    if use_dropout:
        act = Dropout(dropout_rate)

    pad = padding_layer(((1, 1), (1, 1)))(act)
    conv = Conv2D(num_filters, kernel_size, use_bias=use_bias, kernel_initializer=conv_kernel_init,
                  bias_initializer=conv_bias_init)(pad)
    norm = get_norm_layer(norm_layer)(conv)
    
    return Add()([previous, norm])


# TODO Add ImageGAN
def build_pixel_gan(input_shape, init_num_filters, norm_layer, use_bias, final_activation):
    conv_kernel_init, conv_bias_init = get_conv_initialiers()
    input_img = Input(shape=input_shape)
    conv = Conv2D(init_num_filters, (1, 1), padding='same', use_bias=use_bias, kernel_initializer=conv_kernel_init,
                  bias_initializer=conv_bias_init, input_shape=input_shape)(input_img)
    act = LeakyReLU(0.2)(conv)
    conv = Conv2D(init_num_filters * 2, (1, 1), padding='same', use_bias=use_bias, kernel_initializer=conv_kernel_init,
                  bias_initializer=conv_bias_init)(act)
    norm = get_norm_layer(norm_layer)(conv)
    act = LeakyReLU(0.2)(norm)
    conv = Conv2D(init_num_filters * 2, (1, 1), padding='same', use_bias=use_bias, activation=final_activation,
                  kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(act)
    
    model = Model(inputs=input_img, outputs=conv)
    
    return model


def build_nlayer_discriminator(image_size, input_nc, init_num_filters=64, n_layers=3, norm_layer='batch',
                               use_sigmoid=False, use_dropout=False, dropout_rate=0.5):
    kernel_size = (4, 4)
    conv_kernel_init, conv_bias_init = get_conv_initialiers()
    input_shape = get_input_shape(image_size, input_nc)
    use_bias = norm_layer == 'instance'
    final_activation = 'sigmoid' if use_sigmoid else None
    
    if n_layers == 0:  # Pixel-wise GAN Discriminator
        return build_pixel_gan(input_shape, init_num_filters, norm_layer, use_bias, use_sigmoid)
    elif n_layers == -1:  # Image-wise GAN DIscriminator
        n_layers = np.log2(image_size[0])
    
    input_img = Input(shape=input_shape)
    conv = Conv2D(init_num_filters, kernel_size, strides=(2, 2), padding='same', use_bias=use_bias,
                  kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init,
                  input_shape=input_shape)(input_img)
    act = LeakyReLU(0.2)(conv)

    for n in range(1, n_layers):
        conv = Conv2D(init_num_filters * min(2 ** n, 8), kernel_size, strides=(2, 2), padding='same', use_bias=use_bias,
                      kernel_initializer=conv_kernel_init, bias_initializer=conv_bias_init)(act)
        norm = get_norm_layer(norm_layer)(conv)
        if use_dropout:
            norm = Dropout(dropout_rate)(norm)
        act = LeakyReLU(0.2)(norm)

    conv = Conv2D(init_num_filters * min(2 ** n_layers, 8), kernel_size, padding='same', use_bias=use_bias)(act)
    norm = get_norm_layer(norm_layer)(conv)
    act = LeakyReLU(0.2)(norm)
    conv = Conv2D(1, kernel_size, padding='same', activation=final_activation, kernel_initializer=conv_kernel_init,
                  bias_initializer=conv_bias_init)(act)
    
    model = Model(inputs=input_img, outputs=conv)

    return model
