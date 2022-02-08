import os
import sys
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

from Tools import *

class IFREMER():
    def __init__(self, args):
        self.args = args
        self.classes = 8
        if self.args.class_grouping:
            self.classes = 7

        if self.args.phase == 'train':
            print('Coming soon!!!')
        elif self.args.phase == 'test':
            self.Testi_Paths = []
            self.Label_Paths = []
            images_counter = 0
            self.images_main_path = self.args.dataset_main_path + 'images/'
            self.labels_main_path = self.args.dataset_main_path + 'masks/'
            #Listing the images
            images_names = os.listdir(self.images_main_path)
            labels_names = os.listdir(self.labels_main_path)

            #Validating the path for images and labels is correct
            if len(images_names) > 0:
                for image_name in images_names:
                    if image_name[-4:] in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.Labels_Available = True
                        image_path = self.images_main_path + image_name
                        label_path = self.labels_main_path + image_name[:-4] + '.bmp'
                        self.Testi_Paths.append(image_path)
                        self.Label_Paths.append(label_path)
                if len(labels_names) == 0:
                    self.Labels_Available = False
            else:
                print("Please check the images' path provided. There is no files in the path provided.")
                sys.exit()
