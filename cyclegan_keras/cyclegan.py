from __future__ import print_function, division

import os
import random
import time

import numpy as np
from scipy.misc import imread, imresize

from keras import backend
from keras.models import Input, Model
from keras.optimizers import Adam

from .networks import build_generator_model, build_discriminator_model, get_input_shape, get_channel_axis


class CycleGAN(object):

    def __init__(self, image_size, input_nc, output_nc, generator_name, discriminator_layers, init_num_filters,
                 use_lsgan, use_dropout, norm_layer, learning_rate, beta1, lambda_a, lambda_b, id_bool, id_lambda):
        self.image_size = image_size
        self.id_bool = id_bool
        self.gen_a = build_generator_model(image_size, input_nc, output_nc, init_num_filters, generator_name,
                                           norm_layer, use_dropout)
        self.gen_b = build_generator_model(image_size, output_nc, input_nc, init_num_filters, generator_name,
                                           norm_layer, use_dropout)
        self.dis_a = build_discriminator_model(image_size, input_nc, init_num_filters, discriminator_layers,
                                               norm_layer, not use_lsgan)
        self.dis_b = build_discriminator_model(image_size, output_nc, init_num_filters, discriminator_layers,
                                               norm_layer, not use_lsgan)
        
        # Build adversarial model
        image_size_a = get_input_shape(self.image_size, input_nc)
        real_a = Input(shape=image_size_a)  # A
        fake_b = self.gen_a(real_a)  # B' = G_A(A)
        recon_a = self.gen_b(fake_b)  # A'' = G_B(G_A(A))
        dis_fake_b = self.dis_b(fake_b)  # D_A(G_A(A))

        image_size_b = get_input_shape(self.image_size, output_nc)
        real_b = Input(shape=image_size_b)  # B
        fake_a = self.gen_b(real_b)  # A' = G_B(B)
        recon_b = self.gen_a(fake_a)  # B'' = G_A(G_B(B))
        dis_fake_a = self.dis_a(fake_a)  # D_B(G_B(B))
        
        gan_loss = 'MSE' if use_lsgan else 'binary_crossentropy'
        if id_bool:
            id_a = self.gen_b(real_a)  # I' = G_B(A)
            id_b = self.gen_a(real_b)  # I' = G_A(B)
            self.adversarial_model = Model([real_a, real_b], [dis_fake_b, dis_fake_a, recon_a, recon_b, id_a, id_b])
            self.adversarial_model.compile(optimizer=Adam(learning_rate, beta1), loss=[gan_loss, gan_loss, 'MAE', 'MAE',
                                                                                       'MAE', 'MAE'],
                                           loss_weights=[1.0, 1.0, lambda_a, lambda_b, id_lambda*lambda_a, 
                                                         id_lambda*lambda_b])
            self.loss_names = ['G_A', 'G_B', 'Cyc_A', 'Cyc_B', 'Id_A', 'Id_B']
        else:
            self.adversarial_model = Model([real_a, real_b], [dis_fake_b, dis_fake_a, recon_a, recon_b])
            self.adversarial_model.compile(optimizer=Adam(learning_rate, beta1),
                                           loss=[gan_loss, gan_loss, 'MAE', 'MAE'],
                                           loss_weights=[1.0, 1.0, lambda_a, lambda_b])
        self.loss_names = ['G_A', 'G_B', 'Cyc_A', 'Cyc_B']
        
        # Build discriminators model
        real_a = Input(shape=image_size_a)
        fake_a = Input(shape=image_size_a)
        
        dis_real_a = self.dis_a(real_a)
        dis_fake_a = self.dis_a(fake_a)
        
        self.discriminator_model_a = Model([real_a, fake_a],
                                           [dis_real_a, dis_fake_a])
        self.discriminator_model_a.compile(optimizer=Adam(learning_rate, beta1), loss=gan_loss)
        self.loss_names += ['Real_A', 'Fake_A']

        real_b = Input(shape=image_size_b)
        fake_b = Input(shape=image_size_b)

        dis_real_b = self.dis_b(real_b)
        dis_fake_b = self.dis_b(fake_b)
        
        self.discriminator_model_b = Model([real_b, fake_b],
                                           [dis_real_b, dis_fake_b])
        self.discriminator_model_b.compile(optimizer=Adam(learning_rate, beta1), loss=gan_loss)
        self.loss_names += ['Real_B', 'Fake_B']
        
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.input_a = None
        self.input_b = None
        
    def load_input_images(self, root_dir_a, root_dir_b, resize, crop_size):
        self.input_a = ImageGenerator(root_dir_a, resize, crop_size)
        self.input_b = ImageGenerator(root_dir_b, resize, crop_size)
        
    def decay_learning_rate(self, decay_rate):
        self.current_lr = self.current_lr - decay_rate
        backend.set_value(self.adversarial_model.optimizer.lr, self.current_lr)
        backend.set_value(self.discriminator_model_a.optimizer.lr, self.current_lr)
        backend.set_value(self.discriminator_model_b.optimizer.lr, self.current_lr)
        
    def fit(self, output_prefix, batch_size, pool_size, n_epoch, n_epoch_delay, steps_per_epoch, save_freq, print_freq,
            starting_epoch=1):
        decay_rate = self.lr / n_epoch_delay
        real_labels = np.zeros((batch_size,) + self.dis_a.output_shape[1:])
        fake_labels = np.ones((batch_size,) + self.dis_a.output_shape[1:])
        
        total_steps = 0
        for epoch in range(starting_epoch, n_epoch + n_epoch_delay + 1):
            epoch_start_time = time.time()
            if epoch > n_epoch + 1:
                self.decay_learning_rate(decay_rate)
            
            pool_a = ImagePool(pool_size)
            pool_b = ImagePool(pool_size)
            
            for i in range(steps_per_epoch):
                iter_start_time = time.time()
                total_steps += 1
                
                real_a = self.input_a(batch_size)
                real_b = self.input_b(batch_size)
                
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

            print('End of epoch %d / %d \t Time Elapsed: %d sec' % (epoch, n_epoch + n_epoch_delay,
                                                                    time.time() - epoch_start_time))
            
            if epoch % save_freq == 0:
                self.gen_a.save(output_prefix + ('_G_A_epoch%02d.h5' % epoch))
                self.gen_b.save(output_prefix + ('_G_B_epoch%02d.h5' % epoch))
                self.dis_a.save(output_prefix + ('_D_A_epoch%02d.h5' % epoch))
                self.dis_b.save(output_prefix + ('_D_B_epoch%02d.h5' % epoch))


class ImageGenerator(object):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp']
    
    def __init__(self, root_dir, resize=None, crop_size=None):
        self.root_dir = root_dir
        self.resize = resize
        self.crop_size = crop_size
        
        self.images = []
        for root, _, fnames in sorted(os.walk(self.root_dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    self.images.append(path)

    @staticmethod
    def is_image_file(filename):
        return any(filename.lower().endswith(extension) for extension in ImageGenerator.IMG_EXTENSIONS)
    
    def random_crop(self, img):
        startx = random.randint(0, img.shape[0]-self.crop_size[0])
        starty = random.randint(0, img.shape[1]-self.crop_size[1])
        return img[startx:startx+self.crop_size[0], starty:starty+self.crop_size[1]]
    
    def __call__(self, batch_size):
        indexes = np.random.choice(len(self.images), batch_size)
        batch = []
        for idx in indexes:
            img = imread(self.images[idx]).astype(np.float32)
            if self.resize is not None:
                img = imresize(img, self.resize, mode='F')
            if self.crop_size is not None:
                img = self.random_crop(img)
            if len(img.shape) < 3:
                img = np.reshape(img, img.shape + (1,))
            batch.append(img)
        batch = np.array(batch)
        if get_channel_axis() > 0:
            batch = np.transpose(batch, [0, 3, 1, 2])
        return batch


class ImagePool(object):
    
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []
        
    def add_to_pool(self, image_list):
        self.images.extend(image_list)
        self.images = self.images[-self.pool_size:]
        
    def generate_batch(self, batch_size):
        return np.random.permutation(self.images)[:batch_size]
