from __future__ import print_function, division

import os
import sys
import time
from glob import glob
from collections import OrderedDict

import numpy as np

from keras import backend
from keras.models import Input, Model, load_model
from keras.optimizers import Adam
from keras_contrib.layers.normalization import InstanceNormalization

from .networks import build_generator_model, build_discriminator_model
from .visualization import save_training_page
from .utils import get_input_shape


class CycleGAN(object):

    def __init__(self, image_size, input_nc, output_nc, generator_name, discriminator_layers, init_num_filters_gen,
                 init_num_filters_dis, use_lsgan, use_dropout, norm_layer, deconv_type, learning_rate, beta1, lambda_a,
                 lambda_b, id_bool, id_lambda, stacked_training, continue_training, model_dir, exp_to_load,
                 which_epoch):
        self.id_bool = id_bool
        self.lr = learning_rate
        self.beta1 = beta1
        self.stacked_training = stacked_training
        self.input_a = None
        self.input_b = None
        self.generative_model = None
        self.adversarial_model = None
        
        self.image_size_a = get_input_shape(image_size, input_nc)
        self.image_size_b = get_input_shape(image_size, output_nc)
        
        if self.id_bool and input_nc != output_nc:
            raise ValueError('Identity mapping is not supported with unequal channels between inputs and outputs.')
        
        if continue_training:
            self.load_models(model_dir, exp_to_load, which_epoch)
        else:
            self.gen_a = build_generator_model(image_size, input_nc, output_nc, init_num_filters_gen, generator_name,
                                               norm_layer, deconv_type, use_dropout)
            self.gen_b = build_generator_model(image_size, output_nc, input_nc, init_num_filters_gen, generator_name,
                                               norm_layer, deconv_type, use_dropout)
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
        
    @staticmethod
    def make_trainable(net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val
        
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
        # Build adversarial models
        real_a = Input(shape=self.image_size_a)
        fake_a = Input(shape=self.image_size_a)
        real_b = Input(shape=self.image_size_b)
        fake_b = Input(shape=self.image_size_b)
    
        dis_real_a = self.dis_a(real_a)
        dis_fake_a = self.dis_a(fake_a)
        dis_real_b = self.dis_b(real_b)
        dis_fake_b = self.dis_b(fake_b)
    
        self.adversarial_model = Model([real_a, fake_a, real_b, fake_b],
                                       [dis_real_a, dis_fake_a, dis_real_b, dis_fake_b])
        self.adversarial_model.compile(optimizer=Adam(self.lr, self.beta1), loss=self.gan_loss)
        
        self.make_trainable(self.adversarial_model, not self.stacked_training)
        
        # Build generative model
        real_a = Input(shape=self.image_size_a)  # A
        fake_b = self.gen_a(real_a)  # B' = G_A(A)
        recon_a = self.gen_b(fake_b)  # A'' = G_B(G_A(A))
        dis_fake_b = self.dis_b(fake_b)  # D_A(G_A(A))
        
        real_b = Input(shape=self.image_size_b)  # B
        fake_a = self.gen_b(real_b)  # A' = G_B(B)
        recon_b = self.gen_a(fake_a)  # B'' = G_A(G_B(B))
        dis_fake_a = self.dis_a(fake_a)  # D_B(G_B(B))
    
        if self.id_bool:
            id_a = self.gen_b(real_a)  # I' = G_B(A)
            id_b = self.gen_a(real_b)  # I' = G_A(B)
            self.generative_model = Model([real_a, real_b], [dis_fake_b, dis_fake_a, recon_a, recon_b, id_a, id_b])
            self.generative_model.compile(optimizer=Adam(self.lr, self.beta1), loss=self.gen_loss_functions,
                                          loss_weights=self.gen_loss_weights)
        else:
            self.generative_model = Model([real_a, real_b], [dis_fake_b, dis_fake_a, recon_a, recon_b])
            self.generative_model.compile(optimizer=Adam(self.lr, self.beta1), loss=self.gen_loss_functions,
                                          loss_weights=self.gen_loss_weights)
    
    def connect_inputs(self, input_generator_a, input_generator_b):
        self.input_a = input_generator_a
        self.input_b = input_generator_b
        
    def decay_learning_rate(self, epoch, total_epochs):
        current_lr = self.lr - (epoch * self.lr / total_epochs)
        backend.set_value(self.generative_model.optimizer.lr, current_lr)
        backend.set_value(self.adversarial_model.optimizer.lr, current_lr)
        
    def fit(self, model_dir, experiment_name, batch_size, pool_size, n_epochs, n_epochs_decay, steps_per_epoch,
            pretrain_iters, save_freq, print_freq, starting_epoch):
        real_labels = np.ones((batch_size,) + self.dis_a.output_shape[1:])
        fake_labels = np.zeros((batch_size,) + self.dis_a.output_shape[1:])
        
        if pretrain_iters > 0:
            print('Beginning Pre-Training of Adversarial Model...')
            sys.stdout.flush()
            self.make_trainable(self.adversarial_model, True)
            iter_losses = []
            losses = []
            for step in range(1, pretrain_iters + 1):
                iter_start_time = time.time()
                real_a, _ = self.input_a(batch_size)
                real_b, _ = self.input_b(batch_size)
                fake_a = self.gen_b.predict(real_b)
                fake_b = self.gen_a.predict(real_a)

                _, d_a_loss_real, d_a_loss_fake, d_b_loss_real, d_b_loss_fake = \
                    self.adversarial_model.train_on_batch([real_a, fake_a, real_b, fake_b],
                                                          [real_labels, fake_labels, real_labels, fake_labels])
                d_loss = [d_a_loss_real, d_a_loss_fake, d_b_loss_real, d_b_loss_fake]
                iter_losses.append(d_loss)
                
                if step % print_freq == 0:
                    mean_iter_loss = []
                    for loss in zip(*iter_losses):
                        mean_iter_loss.append(np.mean(loss))
                    losses.extend(iter_losses)
                    iter_losses = []
                    time_per_img = (time.time() - iter_start_time) / batch_size
                    message = '(Pre-Train, iter: %d, time: %.3f) ' % (step+1, time_per_img)
                    for name, loss in zip(self.loss_names, mean_iter_loss):
                        message += '%s: %.3f ' % (name, loss)
                    print(message)
                    sys.stdout.flush()
                
            self.make_trainable(self.adversarial_model, not self.stacked_training)
            mean_loss = []
            for loss in zip(*losses):
                mean_loss.append(np.mean(loss))
            print('(Discriminator Pre-Training) Real_A: %.3f, Fake_A: %.3f, Real_B: %.3f, Fake_B: %.3f' %
                  tuple(mean_loss))
            sys.stdout.flush()
            
        total_steps = 0
        for epoch in range(starting_epoch, n_epochs + n_epochs_decay + 1):
            epoch_start_time = time.time()
            if epoch > n_epochs + 1:
                self.decay_learning_rate(epoch - (n_epochs + 1), n_epochs_decay)
            
            pool_a = ImagePool(pool_size)
            pool_b = ImagePool(pool_size)
            
            iter_losses = []
            losses = []
            for i in range(steps_per_epoch):
                iter_start_time = time.time()
                total_steps += 1
                
                real_a, continuing_a = self.input_a(batch_size)
                real_b, continuing_b = self.input_b(batch_size)

                pool_a.add_to_pool(self.gen_b.predict(real_b))
                pool_b.add_to_pool(self.gen_a.predict(real_a))
                
                self.make_trainable(self.adversarial_model, True)

                _, d_a_loss_real, d_a_loss_fake, d_b_loss_real, d_b_loss_fake = \
                    self.adversarial_model.train_on_batch([real_a, pool_a.generate_batch(batch_size),
                                                           real_b, pool_b.generate_batch(batch_size)],
                                                          [real_labels, fake_labels, real_labels, fake_labels])
                d_loss = [d_a_loss_real, d_a_loss_fake, d_b_loss_real, d_b_loss_fake]
                
                self.make_trainable(self.adversarial_model, not self.stacked_training)
                
                if self.id_bool:
                    _, g_loss_dis_b, g_loss_dis_a, g_loss_rec_a, g_loss_rec_b, g_loss_id_a, g_loss_id_b = \
                        self.generative_model.train_on_batch([real_a, real_b],
                                                             [real_labels, real_labels, real_a, real_b,
                                                              real_a, real_b])
                    g_loss = [g_loss_dis_b, g_loss_dis_a, g_loss_rec_a, g_loss_rec_b, g_loss_id_a, g_loss_id_b]
                else:
                    _, g_loss_dis_b, g_loss_dis_a, g_loss_rec_a, g_loss_rec_b = \
                        self.generative_model.train_on_batch([real_a, real_b],
                                                             [real_labels, real_labels, real_a, real_b])
                    g_loss = [g_loss_dis_b, g_loss_dis_a, g_loss_rec_a, g_loss_rec_b]
                
                iter_losses.append(g_loss + d_loss)
                
                if total_steps % print_freq == 0:
                    mean_iter_loss = []
                    for loss in zip(*iter_losses):
                        mean_iter_loss.append(np.mean(loss))
                    losses.extend(iter_losses)
                    iter_losses = []
                    time_per_img = (time.time() - iter_start_time) / batch_size
                    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i+1, time_per_img)
                    for name, loss in zip(self.loss_names, mean_iter_loss):
                        message += '%s: %.3f ' % (name, loss)
                    print(message)
                    sys.stdout.flush()

                    fake_a = self.gen_b.predict(real_b)
                    fake_b = self.gen_a.predict(real_a)
                    rec_a = self.gen_b.predict(fake_b)
                    rec_b = self.gen_a.predict(fake_a)
                    if self.id_bool:
                        id_a = self.gen_a.predict(real_b)
                        id_b = self.gen_b.predict(real_a)
                        visuals = OrderedDict([('real_A', real_a[0, ...]), ('fake_B', fake_b[0, ...]),
                                               ('rec_A', rec_a[0, ...]), ('idt_B', id_b[0, ...]),
                                               ('real_B', real_b[0, ...]), ('fake_A', fake_a[0, ...]),
                                               ('rec_B', rec_b[0, ...]), ('idt_A', id_a[0, ...])])
                    else:
                        visuals = OrderedDict([('real_A', real_a[0, ...]), ('fake_B', fake_b[0, ...]),
                                               ('rec_A', rec_a[0, ...]), ('real_B', real_b[0, ...]),
                                               ('fake_A', fake_a[0, ...]), ('rec_B', rec_b[0, ...])])
                    save_training_page(os.path.join(model_dir, 'web'), experiment_name, visuals, epoch)
                    
                if not (continuing_a and continuing_b):
                    break
            
            losses.extend(iter_losses)
            mean_loss = []
            for loss in zip(*losses):
                mean_loss.append(np.mean(loss))
            
            print('End of epoch %d / %d \t Time Elapsed: %d sec' % (epoch, n_epochs + n_epochs_decay,
                                                                    time.time() - epoch_start_time))
            message = 'Epoch %d Losses: ' % epoch
            for name, loss in zip(self.loss_names, mean_loss):
                message += '%s: %.3f ' % (name, loss)
            print(message)
            sys.stdout.flush()
            
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
                                             '_epoch%03d.h5' % epoch_number),
                                custom_objects={'InstanceNormalization': InstanceNormalization})
    
    def predict(self, input_generator, output_sink, batch_size):
        result_images = []
        continuing = True
        batch_num = 1
        while continuing:
            print('Starting Batch: %d' % batch_num)
            sys.stdout.flush()
            batch, continuing = input_generator(batch_size)
            result_images.extend(self.model.predict_on_batch(batch))
            batch_num += 1
        output_sink(result_images)


class ImagePool(object):
    
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []
        
    def add_to_pool(self, image_list):
        self.images.extend(image_list)
        self.images = self.images[-self.pool_size:]
        
    def generate_batch(self, batch_size):
        return np.random.permutation(self.images)[:batch_size]
