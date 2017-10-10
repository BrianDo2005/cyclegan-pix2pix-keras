import os
import random

import numpy as np
from scipy.misc import imread, imresize

from .utils import get_channel_axis


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
        startx = random.randint(0, img.shape[0] - self.crop_size[0])
        starty = random.randint(0, img.shape[1] - self.crop_size[1])
        return img[startx:startx + self.crop_size[0], starty:starty + self.crop_size[1]]
    
    def reset(self):
        self.position = 0
    
    def __call__(self, batch_size):
        if self.serial_access:
            if self.position + batch_size <= len(self.images):
                end = self.position + batch_size
                continuing = True
            else:
                end = len(self.images)
                continuing = False
            indexes = range(self.position, end)
            self.position = end
        
        else:
            indexes = np.random.choice(len(self.images), batch_size)
            continuing = True
        batch = []
        for idx in indexes:
            img = imread(self.images[idx]).astype(np.float32)
            if self.resize is not None:
                img = imresize(img, self.resize)
            if self.crop_size is not None:
                temp = np.zeros_like(img)
                while np.sum(temp) == 0:
                    temp = self.random_crop(img)
                img = temp
            if len(img.shape) < 3:
                img = np.reshape(img, img.shape + (1,))
            if self.flip:
                img = img[:, ::-1, :]
            # WARNING: This was copied from PyTorch version and may not be applicable to images not in [0,255]
            img = (img / (img.max() / 2)) - 1
            # END WARNING
            batch.append(img)
        batch = np.array(batch)
        if get_channel_axis() > 0:
            batch = np.transpose(batch, [0, 3, 1, 2])
        return batch, continuing
