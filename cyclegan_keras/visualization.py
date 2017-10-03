import os

import numpy as np
from PIL import Image


class HTML:
    def __init__(self, web_dir, title):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        self.doc = '<!DOCTYPE html>\n<html>\n<head>\n<title>%s</title><body>\n' % self.title

    def add_header(self, string):
        self.doc += '<h3>' + string + '</h3>\n'

    def begin_table(self, border=1):
        self.doc += '<table border="%s" style="table-layout: fixed;">\n' % str(border)
    
    def end_table(self):
        self.doc += '</table>\n'
    
    def begin_row(self):
        self.doc += '<tr>\n'
    
    def end_row(self):
        self.doc += '</tr>\n'

    def begin_cell(self):
        self.doc += '<td style="word-wrap: break-word;" halign=center valign=top>\n'

    def end_cell(self):
        self.doc += '</td>\n'
        
    def br(self):
        self.doc += '<br />\n'

    def add_images(self, ims, txts, links):
        self.begin_table()
        self.begin_row()
        for im, txt, link in zip(ims, txts, links):
            self.begin_cell()
            self.doc += '<p><a href="%s"><img src="%s" /></a></p>\n' % (os.path.join('images', im),
                                                                        os.path.join('images', im))
            self.br()
            self.doc += '<p>%s</p>\n' % txt
            self.end_cell()
        self.end_row()
        self.end_table()

    def save(self):
        doc = self.doc + '</body>\n</html>'
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(doc)
        f.close()


def save_image(image_numpy, image_path):
    image_numpy = (np.squeeze(image_numpy) + 1) / 2.0 * 255.0
    image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    image_pil.save(image_path)


def save_training_page(web_dir, experiment_name, visuals, epoch):
    webpage = HTML(web_dir, 'CycleGAN Experiment = %s' % experiment_name)
    for label, image_numpy in visuals.items():
        img_path = os.path.join(web_dir, 'images', 'epoch%.3d_%s.png' % (epoch, label))
        save_image(image_numpy, img_path)
    for n in range(epoch, 0, -1):
        webpage.add_header('epoch [%d]' % n)
        ims = []
        txts = []
        links = []
        
        for label, image_numpy in visuals.items():
            img_path = 'epoch%.3d_%s.png' % (n, label)
            ims.append(img_path)
            txts.append(label)
            links.append(img_path)
        webpage.add_images(ims, txts, links)
    webpage.save()
