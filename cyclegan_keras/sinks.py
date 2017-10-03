import os

import numpy as np
from scipy.misc import imsave

from .utils import get_channel_axis


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
                   np.squeeze(result_images[i]))