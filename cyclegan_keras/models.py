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
from keras.layers import concatenate
from keras_contrib.layers.normalization import InstanceNormalization

from .generators import InputGenerator, PairedInputGenerator
from .networks import build_generator_model, build_discriminator_model
from .visualization import save_training_page
from .utils import get_input_shape, get_channel_axis


class BaseModel(object):
    input_class = InputGenerator
    inputs = None
    base_models = None
    outer_models = None
    
    def __init__(self, image_size, input_nc, output_nc, generator_name, init_num_filters_gen, use_dropout,
                 norm_layer, deconv_type, learning_rate, beta1, continue_training, model_dir, exp_to_load,
                 which_epoch):
        
        self.image_size = image_size
        self.input_nc = input_nc
        self.input_image_size = get_input_shape(image_size, input_nc)
        self.output_nc = output_nc
        self.output_image_size = get_input_shape(image_size, output_nc)
        
        self.generator_name = generator_name
        self.init_num_filters_gen = init_num_filters_gen
        self.use_dropout = use_dropout
        self.norm_layer = norm_layer
        self.deconv_type = deconv_type
        self.lr = learning_rate
        self.beta1 = beta1
        
        if continue_training:
            self.load_models(model_dir, exp_to_load, which_epoch)
        else:
            self.build_base_models()

        self.build_outer_models()

        self.compile()
    
    @staticmethod
    def make_trainable(net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val
            
    def load_models(self, model_dir, exp_to_load, which_epoch):
        if which_epoch == 'latest':
            g_a_models = sorted(glob(os.path.join(model_dir, exp_to_load + '_' + self.base_models[0][1] +
                                                  '_epoch*.h5')))
            epoch_number = int(os.path.basename(g_a_models[-1]).split('_')[-1].split('.')[0].replace('epoch', ''))
        else:
            epoch_number = int(which_epoch)
        for model_arg, model_name, _ in self.base_models:
            setattr(self, model_arg, load_model(os.path.join(model_dir, exp_to_load + '_' +
                                                             model_name + '_epoch%03d.h5' % epoch_number)))
    
    def build_base_model(self, model_arg, model_type):
        if model_type == 'gen':
            setattr(self, model_arg, build_generator_model(self.image_size, self.input_nc, self.output_nc,
                                                           self.init_num_filters_gen, self.generator_name,
                                                           self.norm_layer, self.deconv_type, self.use_dropout))
        else:
            raise ValueError('Model type %s is not valid.' % model_type)
    
    def build_base_models(self):
        for model_arg, _, model_type in self.base_models:
            self.build_base_model(model_arg, model_type)
            
    def decay_learning_rate(self, epoch, total_epochs):
        current_lr = self.lr - (epoch * self.lr / total_epochs)
        for model_arg, _, _, _ in self.outer_models:
            backend.set_value(getattr(self, model_arg).optimizer.lr, current_lr)

    def connect_inputs(self, **kwargs):
        for input_name, input_generator in kwargs.items():
            if input_name not in self.inputs:
                raise ValueError('Input %s not valid for this class. [%s]' % (input_name, str(self.inputs)))
            # if not isinstance(input_generator, self.input_class):
            #     raise ValueError('Input generator is not an instance of %s' % self.input_class.__name__)
            setattr(self, input_name, input_generator)
            
    def loss_names(self):
        return [item for model_arr in self.outer_models for item in model_arr[3]]
            
    def compile(self):
        # Compile models
        for model_arg, loss_functions, loss_weights, _ in self.outer_models:
            getattr(self, model_arg).compile(optimizer=Adam(self.lr, self.beta1), loss=loss_functions,
                                             loss_weights=loss_weights)
    
    def save_models(self, model_dir, experiment_name, epoch):
        for model_arg, model_name, _ in self.base_models:
            getattr(self, model_arg).save(os.path.join(model_dir, '%s_%s_epoch%03d.h5' %
                                                       (experiment_name, model_name, epoch)))
            
    def build_outer_models(self):
        raise NotImplementedError
    
    def fit(self, model_dir, experiment_name, batch_size, n_epochs, n_epochs_decay, steps_per_epoch, save_freq,
            print_freq, starting_epochs):
        raise NotImplementedError
        

class SingleCovNet(BaseModel):
    input_class = PairedInputGenerator
    
    def __init__(self, image_size, input_nc, output_nc, generator_name, init_num_filters_gen, use_dropout, norm_layer,
                 deconv_type, learning_rate, beta1, continue_training, model_dir, exp_to_load, which_epoch):
        
        self.input = None
        self.model = None

        self.inputs = ['input']
        self.base_models = [('model', 'generator', 'gen')]
        self.outer_models = [('model', ['MSE'], [1.0], ['Loss'])]

        super(SingleCovNet, self).__init__(image_size, input_nc, output_nc, generator_name,
                                           init_num_filters_gen, use_dropout, norm_layer, deconv_type, learning_rate,
                                           beta1, continue_training, model_dir, exp_to_load, which_epoch)
        
    def build_outer_models(self):
        pass

    def fit(self, model_dir, experiment_name, batch_size, n_epochs, n_epochs_decay, steps_per_epoch,
            save_freq, print_freq, starting_epoch):
    
        total_steps = 0
        for epoch in range(starting_epoch, n_epochs + n_epochs_decay + 1):
            epoch_start_time = time.time()
            if epoch > n_epochs:
                self.decay_learning_rate(epoch - n_epochs, n_epochs_decay)
        
            iter_losses = []
            losses = []
            for i in range(steps_per_epoch):
                iter_start_time = time.time()
                total_steps += 1
            
                real_a, real_b, continuing = self.input(batch_size)
                model_loss = self.model.train_on_batch([real_a], [real_b])
            
                iter_losses.append([model_loss])
            
                if total_steps % print_freq == 0:
                    mean_iter_loss = []
                    for loss in zip(*iter_losses):
                        mean_iter_loss.append(np.mean(loss))
                    losses.extend(iter_losses)
                    iter_losses = []
                    time_per_img = (time.time() - iter_start_time) / batch_size
                    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i + 1, time_per_img)
                    for name, loss in zip(self.loss_names(), mean_iter_loss):
                        message += '%s: %.3f ' % (name, loss)
                    print(message)
                    sys.stdout.flush()
                
                    fake_b = self.model.predict(real_a)
                    visuals = OrderedDict([('real_A', real_a[0, ...]), ('fake_B', fake_b[0, ...]),
                                           ('real_B', real_b[0, ...])])
                    save_training_page(os.path.join(model_dir, 'web'), experiment_name, visuals, epoch)
            
                if not continuing:
                    break
        
            losses.extend(iter_losses)
            mean_loss = []
            for loss in zip(*losses):
                mean_loss.append(np.mean(loss))
        
            print('End of epoch %d / %d \t Time Elapsed: %d sec' % (epoch, n_epochs + n_epochs_decay,
                                                                    time.time() - epoch_start_time))
            message = 'Epoch %d Losses: ' % epoch
            for name, loss in zip(self.loss_names(), mean_loss):
                message += '%s: %.3f ' % (name, loss)
            print(message)
            sys.stdout.flush()
        
            if epoch % save_freq == 0:
                self.save_models(model_dir, experiment_name, epoch)


class GANModel(BaseModel):
    
    def __init__(self, image_size, input_nc, output_nc, generator_name, discriminator_layers, init_num_filters_gen,
                 init_num_filters_dis, use_lsgan, use_dropout, norm_layer, deconv_type, learning_rate, beta1,
                 pretrain_iters, pool_size, stacked_training, continue_training, model_dir, exp_to_load,
                 which_epoch):
        
        self.stacked_training = stacked_training
        self.discriminator_layers = discriminator_layers
        self.init_num_filters_dis = init_num_filters_dis
        self.use_lsgan = use_lsgan
        self.pretrain_iters = pretrain_iters
        self.pool_size = pool_size

        super(GANModel, self).__init__(image_size, input_nc, output_nc, generator_name, init_num_filters_gen,
                                       use_dropout, norm_layer, deconv_type, learning_rate, beta1, continue_training,
                                       model_dir, exp_to_load, which_epoch)
        
    def build_base_model(self, model_arg, model_type):
        if model_type == 'dis':
            setattr(self, model_arg, build_discriminator_model(self.image_size, self.input_nc,
                                                               self.init_num_filters_dis,
                                                               self.discriminator_layers, self.norm_layer,
                                                               not self.use_lsgan))
        else:
            super(GANModel, self).build_base_model(model_arg, model_type)
            
    def build_outer_models(self):
        raise NotImplementedError

    def fit(self, model_dir, experiment_name, batch_size, n_epochs, n_epochs_decay, steps_per_epoch,
            save_freq, print_freq, starting_epoch):
        raise NotImplementedError
    

class Pix2Pix(GANModel):
    input_class = PairedInputGenerator
    
    def __init__(self, image_size, input_nc, output_nc, generator_name, discriminator_layers, init_num_filters_gen,
                 init_num_filters_dis, use_lsgan, use_dropout, norm_layer, deconv_type, learning_rate, beta1,
                 lambda_weight, pretrain_iters, pool_size, stacked_training, continue_training, model_dir, exp_to_load,
                 which_epoch):
        
        self.input = None
        self.gen = None
        self.dis = None
        self.generative_model = None
        self.adversarial_model = None
        
        gan_loss = 'MSE' if use_lsgan else 'binary_crossentropy'
        self.inputs = ['input']
        self.base_models = [('gen', 'G', 'gen'), ('dis', 'D', 'dis')]
        self.outer_models = [('adversarial_model', gan_loss, 1.0, ['D_Real', 'D_Fake']),
                             ('generative_model', [gan_loss, 'MAE'], [1.0, lambda_weight], ['G_GAN', 'G_L1'])]
        
        super(Pix2Pix, self).__init__(image_size, input_nc, output_nc, generator_name, discriminator_layers,
                                      init_num_filters_gen, init_num_filters_dis, use_lsgan, use_dropout, norm_layer,
                                      deconv_type, learning_rate, beta1, pretrain_iters, pool_size, stacked_training,
                                      continue_training, model_dir, exp_to_load, which_epoch)
        
    def build_outer_models(self):
        # Build adversarial models
        real_a = Input(shape=self.input_image_size)
        real_b = Input(shape=self.output_image_size)
        fake_b = Input(shape=self.output_image_size)
        
        real_ab = concatenate([real_a, real_b], axis=get_channel_axis())
        fake_ab = concatenate([real_a, fake_b], axis=get_channel_axis())
    
        dis_real_ab = self.dis(real_ab)
        dis_fake_ab = self.dis(fake_ab)
    
        self.adversarial_model = Model([real_a, real_b, fake_b], [dis_real_ab, dis_fake_ab])
        
        # Build generative model
        real_a = Input(shape=self.input_image_size)  # A
        real_b = Input(shape=self.output_image_size)  # B
        fake_b = self.gen(real_a)  # B' = G_A(A)
        fake_ab = concatenate([real_a, fake_b], axis=get_channel_axis())
        dis_fake_ab = self.dis(fake_ab)
    
        self.generative_model = Model([real_a, real_b], [dis_fake_ab, real_b])

    def fit(self, model_dir, experiment_name, batch_size, n_epochs, n_epochs_decay, steps_per_epoch, save_freq,
            print_freq, starting_epoch):
    
        real_labels = np.ones((batch_size,) + self.dis.output_shape[1:])
        fake_labels = np.zeros((batch_size,) + self.dis.output_shape[1:])
    
        if self.pretrain_iters > 0:
            print('Beginning Pre-Training of Adversarial Model...')
            sys.stdout.flush()
            self.make_trainable(self.adversarial_model, True)
            iter_losses = []
            losses = []
            for step in range(1, self.pretrain_iters + 1):
                iter_start_time = time.time()
                real_a, real_b, _ = self.input(batch_size)
                fake_b = self.gen.predict(real_a)
                
                real_ab = np.concatenate([real_a, real_b], axis=get_channel_axis())
                fake_ab = np.concatenate([real_a, fake_b], axis=get_channel_axis())
            
                _, d_loss_real, d_loss_fake = self.adversarial_model.train_on_batch([real_ab, fake_ab],
                                                                                    [real_labels, fake_labels])
                d_loss = [d_loss_real, d_loss_fake]
                iter_losses.append(d_loss)
            
                if step % print_freq == 0:
                    mean_iter_loss = []
                    for loss in zip(*iter_losses):
                        mean_iter_loss.append(np.mean(loss))
                    losses.extend(iter_losses)
                    iter_losses = []
                    time_per_img = (time.time() - iter_start_time) / batch_size
                    message = '(Pre-Train, iter: %d, time: %.3f) ' % (step, time_per_img)
                    for name, loss in zip(self.loss_names[-4:], mean_iter_loss):
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
            if epoch > n_epochs:
                self.decay_learning_rate(epoch - n_epochs, n_epochs_decay)
        
            pool = ImagePool(self.pool_size)
        
            iter_losses = []
            losses = []
            for i in range(steps_per_epoch):
                iter_start_time = time.time()
                total_steps += 1
            
                real_a, real_b, continuing = self.input(batch_size)
                pool.add_to_pool(self.gen.predict(real_a))
            
                self.make_trainable(self.adversarial_model, True)
                _, d_loss_real, d_loss_fake = self.adversarial_model.train_on_batch([real_a, real_b,
                                                                                     pool.generate_batch(batch_size)],
                                                                                    [real_labels, fake_labels])
                d_loss = [d_loss_real, d_loss_fake]
                self.make_trainable(self.adversarial_model, not self.stacked_training)
            
                _, g_loss_gan, g_loss_l1 = self.generative_model.train_on_batch([real_a, real_b],
                                                                                [real_labels, real_b])
                g_loss = [g_loss_gan, g_loss_l1]
            
                iter_losses.append(g_loss + d_loss)
            
                if total_steps % print_freq == 0:
                    mean_iter_loss = []
                    for loss in zip(*iter_losses):
                        mean_iter_loss.append(np.mean(loss))
                    losses.extend(iter_losses)
                    iter_losses = []
                    time_per_img = (time.time() - iter_start_time) / batch_size
                    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i + 1, time_per_img)
                    for name, loss in zip(self.loss_names(), mean_iter_loss):
                        message += '%s: %.3f ' % (name, loss)
                    print(message)
                    sys.stdout.flush()
                
                    fake_b = self.gen.predict(real_a)
                    visuals = OrderedDict([('real_A', real_a[0, ...]), ('fake_B', fake_b[0, ...]),
                                           ('real_B', real_b[0, ...])])
                    save_training_page(os.path.join(model_dir, 'web'), experiment_name, visuals, epoch)
            
                if not continuing:
                    break
        
            losses.extend(iter_losses)
            mean_loss = []
            for loss in zip(*losses):
                mean_loss.append(np.mean(loss))
        
            print('End of epoch %d / %d \t Time Elapsed: %d sec' % (epoch, n_epochs + n_epochs_decay,
                                                                    time.time() - epoch_start_time))
            message = 'Epoch %d Losses: ' % epoch
            for name, loss in zip(self.loss_names(), mean_loss):
                message += '%s: %.3f ' % (name, loss)
            print(message)
            sys.stdout.flush()
        
            if epoch % save_freq == 0:
                self.save_models(model_dir, experiment_name, epoch)


class CycleGAN(GANModel):

    def __init__(self, image_size, input_nc, output_nc, generator_name, discriminator_layers, init_num_filters_gen,
                 init_num_filters_dis, use_lsgan, use_dropout, norm_layer, deconv_type, learning_rate, beta1, lambda_a,
                 lambda_b, use_identity_loss, id_lambda, pretrain_iters, pool_size, stacked_training, continue_training,
                 model_dir, exp_to_load, which_epoch):
        
        self.use_identity_loss = use_identity_loss
        
        self.input_a = None
        self.input_b = None
        self.gen_a = None
        self.gen_b = None
        self.dis_a = None
        self.dis_b = None
        self.generative_model = None
        self.adversarial_model = None
        
        if self.use_identity_loss and input_nc != output_nc:
            raise ValueError('Identity mapping is not supported with unequal channels between inputs and outputs.')

        gan_loss = 'MSE' if use_lsgan else 'binary_crossentropy'
        if self.use_identity_loss:
            gen_loss_weights = [1.0, 1.0, lambda_a, lambda_b, id_lambda * lambda_a, id_lambda * lambda_b]
            gen_loss_functions = [gan_loss, gan_loss, 'MAE', 'MAE', 'MAE', 'MAE']
            gen_loss_names = ['G_A', 'G_B', 'Cyc_A', 'Cyc_B', 'Id_A', 'Id_B']
        else:
            gen_loss_weights = [1.0, 1.0, lambda_a, lambda_b]
            gen_loss_functions = [gan_loss, gan_loss, 'MAE', 'MAE']
            gen_loss_names = ['G_A', 'G_B', 'Cyc_A', 'Cyc_B']
        dis_loss_names = ['Real_A', 'Fake_A', 'Real_B', 'Fake_B']

        self.inputs = ['input_a', 'input_b']
        self.base_models = [('gen_a', 'G_A', 'gen'), ('gen_b', 'G_B', 'gen'), ('dis_a', 'D_A', 'dis'),
                            ('dis_b', 'D_B', 'dis')]
        self.models = [('adversarial_model', gen_loss_functions, gen_loss_weights, gen_loss_names),
                       ('generative_model', gan_loss, 1.0, dis_loss_names)]

        super(CycleGAN, self).__init__(image_size, input_nc, output_nc, generator_name, discriminator_layers,
                                       init_num_filters_gen, init_num_filters_dis, use_lsgan, use_dropout, norm_layer,
                                       deconv_type, learning_rate, beta1, pretrain_iters, pool_size, stacked_training,
                                       continue_training, model_dir, exp_to_load, which_epoch)
        
    def build_outer_models(self):
        # Build adversarial models
        real_a = Input(shape=self.input_image_size)
        fake_a = Input(shape=self.input_image_size)
        real_b = Input(shape=self.output_image_size)
        fake_b = Input(shape=self.output_image_size)
    
        dis_real_a = self.dis_a(real_a)
        dis_fake_a = self.dis_a(fake_a)
        dis_real_b = self.dis_b(real_b)
        dis_fake_b = self.dis_b(fake_b)
    
        self.adversarial_model = Model([real_a, fake_a, real_b, fake_b],
                                       [dis_real_a, dis_fake_a, dis_real_b, dis_fake_b])
        # Build generative model
        real_a = Input(shape=self.input_image_size)  # A
        fake_b = self.gen_a(real_a)  # B' = G_A(A)
        recon_a = self.gen_b(fake_b)  # A'' = G_B(G_A(A))
        dis_fake_b = self.dis_b(fake_b)  # D_A(G_A(A))
    
        real_b = Input(shape=self.output_image_size)  # B
        fake_a = self.gen_b(real_b)  # A' = G_B(B)
        recon_b = self.gen_a(fake_a)  # B'' = G_A(G_B(B))
        dis_fake_a = self.dis_a(fake_a)  # D_B(G_B(B))
    
        if self.use_identity_loss:
            id_a = self.gen_b(real_a)  # I' = G_B(A)
            id_b = self.gen_a(real_b)  # I' = G_A(B)
            self.generative_model = Model([real_a, real_b], [dis_fake_b, dis_fake_a, recon_a, recon_b, id_a, id_b])
        else:
            self.generative_model = Model([real_a, real_b], [dis_fake_b, dis_fake_a, recon_a, recon_b])
                    
    def fit(self, model_dir, experiment_name, batch_size, n_epochs, n_epochs_decay, steps_per_epoch, save_freq,
            print_freq, starting_epoch):
        
        real_labels = np.ones((batch_size,) + self.dis_a.output_shape[1:])
        fake_labels = np.zeros((batch_size,) + self.dis_a.output_shape[1:])
        
        if self.pretrain_iters > 0:
            print('Beginning Pre-Training of Adversarial Model...')
            sys.stdout.flush()
            self.make_trainable(self.adversarial_model, True)
            iter_losses = []
            losses = []
            for step in range(1, self.pretrain_iters + 1):
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
                    message = '(Pre-Train, iter: %d, time: %.3f) ' % (step, time_per_img)
                    for name, loss in zip(self.loss_names[-4:], mean_iter_loss):
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
            if epoch > n_epochs:
                self.decay_learning_rate(epoch - n_epochs, n_epochs_decay)
            
            pool_a = ImagePool(self.pool_size)
            pool_b = ImagePool(self.pool_size)
            
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
                
                if self.use_identity_loss:
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
                    for name, loss in zip(self.loss_names(), mean_iter_loss):
                        message += '%s: %.3f ' % (name, loss)
                    print(message)
                    sys.stdout.flush()

                    fake_a = self.gen_b.predict(real_b)
                    fake_b = self.gen_a.predict(real_a)
                    rec_a = self.gen_b.predict(fake_b)
                    rec_b = self.gen_a.predict(fake_a)
                    if self.use_identity_loss:
                        id_a = self.gen_b.predict(real_a)
                        id_b = self.gen_a.predict(real_b)
                        visuals = OrderedDict([('real_A', real_a[0, ...]), ('fake_B', fake_b[0, ...]),
                                               ('rec_A', rec_a[0, ...]), ('idt_A', id_a[0, ...]),
                                               ('real_B', real_b[0, ...]), ('fake_A', fake_a[0, ...]),
                                               ('rec_B', rec_b[0, ...]), ('idt_b', id_b[0, ...])])
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
            for name, loss in zip(self.loss_names(), mean_loss):
                message += '%s: %.3f ' % (name, loss)
            print(message)
            sys.stdout.flush()
            
            if epoch % save_freq == 0:
                self.save_models(model_dir, experiment_name, epoch)


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
