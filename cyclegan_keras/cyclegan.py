from __future__ import print_function, division

import os
import random
import time
from glob import glob

import numpy as np
from scipy.misc import imread, imresize, imsave

from keras import backend
from keras.models import Input, Model, load_model
from keras.optimizers import Adam

from .networks import build_generator_model, build_discriminator_model, get_input_shape, get_channel_axis


class CycleGAN(object):

    def __init__(self, image_size, input_nc, output_nc, generator_name, discriminator_layers, init_num_filters_gen,
                 init_num_filters_dis, use_lsgan, use_dropout, norm_layer, learning_rate, beta1, lambda_a,
                 lambda_b, id_bool, id_lambda, continue_training, model_dir, exp_to_load, which_epoch):
        self.image_size = image_size
        self.id_bool = id_bool
        self.lr = learning_rate
        self.beta1 = beta1
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.input_a = None
        self.input_b = None
        self.adversarial_model = None
        self.discriminator_model_a = None
        self.discriminator_model_b = None
        
        if self.id_bool and self.input_nc != self.output_nc:
            raise ValueError('Identity mapping is not supported with unequal channels between inputs and outputs.')
        
        if continue_training:
            self.load_models(model_dir, exp_to_load, which_epoch)
        else:
            self.gen_a = build_generator_model(image_size, input_nc, output_nc, init_num_filters_gen, generator_name,
                                               norm_layer, use_dropout)
            self.gen_b = build_generator_model(image_size, output_nc, input_nc, init_num_filters_gen, generator_name,
                                               norm_layer, use_dropout)
            self.dis_a = build_discriminator_model(image_size, input_nc, init_num_filters_dis, discriminator_layers,
                                                   norm_layer, not use_lsgan)
            self.dis_b = build_discriminator_model(image_size, output_nc, init_num_filters_dis, discriminator_layers,
                                                   norm_layer, not use_lsgan)

        self.gan_loss = 'MSE' if use_lsgan else 'binary_crossentropy'
        if self.id_bool:
            self.gen_loss_weights = [1.0, 1.0, lambda_a, lambda_b, id_lambda * lambda_a, id_lambda * lambda_b]
            self.gen_loss_functions = [self.gan_loss, self.gan_loss, 'MAE', 'MAE', 'MAE', 'MAE']
            self.loss_names = ['G_A', 'G_B', 'Cyc_A', 'Cyc_B', 'Id_A', 'Id_B']
        else:
            self.gen_loss_weights = [1.0, 1.0, lambda_a, lambda_b]
            self.gen_loss_functions = [self.gan_loss, self.gan_loss, 'MAE', 'MAE']
            self.loss_names = ['G_A', 'G_B', 'Cyc_A', 'Cyc_B']
        self.loss_names += ['Real_A', 'Fake_A', 'Real_B', 'Fake_B']
        
        self.compile()
        
    def load_models(self, model_dir, exp_to_load, which_epoch):
        if which_epoch == 'latest':
            g_a_models = sorted(glob(os.path.join(model_dir, exp_to_load + '_G_A_epoch*.h5')))
            epoch_number = int(os.path.basename(g_a_models[-1]).split('_')[-1].split('.')[0].replace('epoch', ''))
        else:
            epoch_number = int(which_epoch)
        self.gen_a = load_model(os.path.join(model_dir, exp_to_load + '_G_A_epoch%03d.h5' % epoch_number))
        self.gen_b = load_model(os.path.join(model_dir, exp_to_load + '_G_B_epoch%03d.h5' % epoch_number))
        self.dis_a = load_model(os.path.join(model_dir, exp_to_load + '_D_A_epoch%03d.h5' % epoch_number))
        self.dis_b = load_model(os.path.join(model_dir, exp_to_load + '_D_B_epoch%03d.h5' % epoch_number))
        
    def compile(self):
        # Build adversarial model
        image_size_a = get_input_shape(self.image_size, self.input_nc)
        real_a = Input(shape=image_size_a)  # A
        fake_b = self.gen_a(real_a)  # B' = G_A(A)
        recon_a = self.gen_b(fake_b)  # A'' = G_B(G_A(A))
        dis_fake_b = self.dis_b(fake_b)  # D_A(G_A(A))
    
        image_size_b = get_input_shape(self.image_size, self.output_nc)
        real_b = Input(shape=image_size_b)  # B
        fake_a = self.gen_b(real_b)  # A' = G_B(B)
        recon_b = self.gen_a(fake_a)  # B'' = G_A(G_B(B))
        dis_fake_a = self.dis_a(fake_a)  # D_B(G_B(B))
    
        if self.id_bool:
            id_a = self.gen_b(real_a)  # I' = G_B(A)
            id_b = self.gen_a(real_b)  # I' = G_A(B)
            self.adversarial_model = Model([real_a, real_b], [dis_fake_b, dis_fake_a, recon_a, recon_b, id_a, id_b])
            self.adversarial_model.compile(optimizer=Adam(self.lr, self.beta1), loss=self.gen_loss_functions,
                                           loss_weights=self.gen_loss_weights)
        else:
            self.adversarial_model = Model([real_a, real_b], [dis_fake_b, dis_fake_a, recon_a, recon_b])
            self.adversarial_model.compile(optimizer=Adam(self.lr, self.beta1), loss=self.gen_loss_functions,
                                           loss_weights=self.gen_loss_weights)

        # Build discriminators model
        real_a = Input(shape=image_size_a)
        fake_a = Input(shape=image_size_a)

        dis_real_a = self.dis_a(real_a)
        dis_fake_a = self.dis_a(fake_a)

        self.discriminator_model_a = Model([real_a, fake_a],
                                           [dis_real_a, dis_fake_a])
        self.discriminator_model_a.compile(optimizer=Adam(self.lr, self.beta1), loss=self.gan_loss)

        real_b = Input(shape=image_size_b)
        fake_b = Input(shape=image_size_b)

        dis_real_b = self.dis_b(real_b)
        dis_fake_b = self.dis_b(fake_b)

        self.discriminator_model_b = Model([real_b, fake_b],
                                           [dis_real_b, dis_fake_b])
        self.discriminator_model_b.compile(optimizer=Adam(self.lr, self.beta1), loss=self.gan_loss)
    
    def connect_inputs(self, input_generator_a, input_generator_b):
        self.input_a = input_generator_a
        self.input_b = input_generator_b
        
    def decay_learning_rate(self, epoch, total_epochs):
        current_lr = self.lr - (epoch * self.lr / total_epochs)
        backend.set_value(self.adversarial_model.optimizer.lr, current_lr)
        backend.set_value(self.discriminator_model_a.optimizer.lr, current_lr)
        backend.set_value(self.discriminator_model_b.optimizer.lr, current_lr)
        
    def fit(self, model_dir, experiment_name, batch_size, pool_size, n_epochs, n_epochs_decay, steps_per_epoch,
            save_freq, print_freq, starting_epoch):
        real_labels = np.zeros((batch_size,) + self.dis_a.output_shape[1:])
        fake_labels = np.ones((batch_size,) + self.dis_a.output_shape[1:])
        
        total_steps = 0
        for epoch in range(starting_epoch, n_epochs + n_epochs_decay + 1):
            epoch_start_time = time.time()
            if epoch > n_epochs + 1:
                self.decay_learning_rate(epoch - (n_epochs + 1), n_epochs_decay)
            
            pool_a = ImagePool(pool_size)
            pool_b = ImagePool(pool_size)
            
            for i in range(steps_per_epoch):
                iter_start_time = time.time()
                total_steps += 1
                
                real_a, _ = self.input_a(batch_size)
                real_b, _ = self.input_b(batch_size)
                
                if self.id_bool:
                    g_loss = self.adversarial_model.train_on_batch([real_a, real_b],
                                                                   [fake_labels, fake_labels, real_a, real_b,
                                                                    real_a, real_b])
                else:
                    g_loss = self.adversarial_model.train_on_batch([real_a, real_b],
                                                                   [fake_labels, fake_labels, real_a, real_b])

                pool_a.add_to_pool(self.gen_b.predict(real_b))
                pool_b.add_to_pool(self.gen_a.predict(real_a))

                d_a_loss = self.discriminator_model_a.train_on_batch([real_a, pool_a.generate_batch(batch_size)],
                                                                     [real_labels, fake_labels])
                d_b_loss = self.discriminator_model_b.train_on_batch([real_b, pool_b.generate_batch(batch_size)],
                                                                     [real_labels, fake_labels])
                
                if total_steps % print_freq == 0:
                    time_per_img = (time.time() - iter_start_time) / batch_size
                    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, time_per_img)
                    for name, loss in zip(self.loss_names, g_loss + d_a_loss + d_b_loss):
                        message += '%s: %.3f ' % (name, loss)
                    print(message)

            print('End of epoch %d / %d \t Time Elapsed: %d sec' % (epoch, n_epochs + n_epochs_decay,
                                                                    time.time() - epoch_start_time))
            
            if epoch % save_freq == 0:
                self.gen_a.save(os.path.join(model_dir, '%s_G_A_epoch%03d.h5' % (experiment_name, epoch)))
                self.gen_b.save(os.path.join(model_dir, '%s_G_B_epoch%03d.h5' % (experiment_name, epoch)))
                self.dis_a.save(os.path.join(model_dir, '%s_D_A_epoch%03d.h5' % (experiment_name, epoch)))
                self.dis_b.save(os.path.join(model_dir, '%s_D_B_epoch%03d.h5' % (experiment_name, epoch)))


class PredictionModel(object):
    
    def __init__(self, model_dir, experiment_name, which_epoch, which_direction):
        if which_epoch == 'latest':
            g_a_models = sorted(glob(os.path.join(model_dir, experiment_name + '_G_A_epoch*.h5')))
            epoch_number = int(os.path.basename(g_a_models[-1]).split('_')[-1].split('.')[0].replace('epoch', ''))
        else:
            epoch_number = int(which_epoch)
        model_name = 'G_A' if which_direction == 'AtoB' else 'G_B'
        self.model = load_model(os.path.join(model_dir, experiment_name + '_' + model_name +
                                             '_epoch%03d.h5' % epoch_number))
    
    def predict(self, input_generator, output_sink, batch_size):
        result_images = []
        continuing = True
        while continuing:
            batch, continuing = input_generator(batch_size)
            result_images.extend(self.model.predict_on_batch(batch))
        output_sink(result_images)
        

class InputGenerator(object):
    
    def __init__(self):
        pass
    
    def __call__(self, batch_size):
        raise NotImplementedError


class ImageFileGenerator(InputGenerator):
    
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp']
    
    def __init__(self, root_dir, resize=None, crop_size=None, flip=False, serial_access=False):
        super(ImageFileGenerator, self).__init__()
        self.root_dir = root_dir
        self.resize = resize
        self.crop_size = crop_size
        self.flip = flip
        self.serial_access = serial_access
        self.position = 0
        
        self.images = []
        for root, _, fnames in sorted(os.walk(self.root_dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    self.images.append(path)

    @staticmethod
    def is_image_file(filename):
        return any(filename.lower().endswith(extension) for extension in ImageFileGenerator.IMG_EXTENSIONS)
    
    def random_crop(self, img):
        startx = random.randint(0, img.shape[0]-self.crop_size[0])
        starty = random.randint(0, img.shape[1]-self.crop_size[1])
        return img[startx:startx+self.crop_size[0], starty:starty+self.crop_size[1]]
    
    def __call__(self, batch_size):
        if self.serial_access:
            if self.position + batch_size <= len(self.images):
                end = self.position + batch_size
                continuing = True
            else:
                end = len(self.images)
                continuing = False
            indexes = range(self.position, end)
            
        else:
            indexes = np.random.choice(len(self.images), batch_size)
            continuing = True
        batch = []
        for idx in indexes:
            img = imread(self.images[idx]).astype(np.float32)
            if self.resize is not None:
                img = imresize(img, self.resize, mode='F')
            if self.crop_size is not None:
                img = self.random_crop(img)
            if len(img.shape) < 3:
                img = np.reshape(img, img.shape + (1,))
            if self.flip:
                img = img[:, ::-1, :]
            batch.append(img)
        batch = np.array(batch)
        if get_channel_axis() > 0:
            batch = np.transpose(batch, [0, 3, 1, 2])
        return batch, continuing
    
    
class OutputSink(object):
    
    def __init__(self):
        pass
    
    def __call__(self, result_images):
        raise NotImplementedError


class ImageFileOutputSink(OutputSink):
    
    def __init__(self, image_filenames, output_dir, output_suffix):
        super(ImageFileOutputSink, self).__init__()
        self.image_filenames = image_filenames
        self.output_dir = output_dir
        self.output_suffix = output_suffix
    
    def __call__(self, result_images):
        if get_channel_axis() > 0:
            result_images = np.transpose(result_images, [0, 2, 3, 1])
        for i, img in enumerate(self.image_filenames):
            base, ext = os.path.splitext(os.path.basename(img))
            imsave(os.path.join(self.output_dir, base + self.output_suffix + ext),
                   np.squeeze(result_images[i, :, :, :]))


class ImagePool(object):
    
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []
        
    def add_to_pool(self, image_list):
        self.images.extend(image_list)
        self.images = self.images[-self.pool_size:]
        
    def generate_batch(self, batch_size):
        return np.random.permutation(self.images)[:batch_size]
