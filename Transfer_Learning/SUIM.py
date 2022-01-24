import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

from Tools import *

class SUIM():
    def __init__(self, args):

        self.args = args
        self.classes = 8
        self.Train_Paths = []
        self.Label_Paths = []
        images_dimensions = 0
        labels_sum = tf.constant(0, shape = [1, self.classes], dtype = tf.float32)
        if self.args.phase == 'train':
            images_counter = 0
            self.images_main_path = self.args.dataset_main_path + 'images/'
            self.labels_main_path = self.args.dataset_main_path + 'masks/'
            #Listing the images
            images_names = os.listdir(self.images_main_path)
            for image_name in images_names:
                if image_name[-4:] in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_path = self.images_main_path + image_name
                    label_path = self.labels_main_path + image_name[:-4] + '.bmp'

                    #reading images and labels
                    image = mpimg.imread(image_path)
                    label = mpimg.imread(label_path)

                    if image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1]:

                        self.Train_Paths.append(image_path)
                        self.Label_Paths.append(label_path)

                        if self.args.classweight_type == 'global':
                            #Computing the global class weights
                            labels_ = tf.keras.utils.to_categorical(Label_Converter(label), self.classes)
                            #Computing class weights to mitigate the imabalance between classes
                            labels_sum += (tf.reduce_sum(labels_, [0, 1])/tf.constant(np.shape(image)[0] * np.shape(image)[1], dtype = tf.float32))
                            images_counter += 1
                    #    data = np.zeros((image.shape[0],image.shape[1],image.shape[2] + 1))

                        #Converting rgb labels to a int map
                    #    label = self.Label_Converter(label)
                    #    data[:,:,:3] = image.astype('float32')/255
                    #    data[:,:, 3] = label

                    #    self.data_list.append(data)
            if self.args.classweight_type == 'global':
                print(labels_sum/tf.constant(images_counter, dtype = tf.float32))
                print(1 - labels_sum/tf.constant(images_counter, dtype = tf.float32))
                self.class_weights = (1 - (labels_sum/tf.constant(images_counter, dtype = tf.float32)))*10
                print('Class weight used')
                print(self.class_weights)
            
            print("Splitting the data into Training and Validation sets")
            num_samples = len(self.Train_Paths)
            num_samples_val = int((num_samples * 10)/100)
            # Applying a shuffle aiming at avoid the network uses always the same samples
            index = np.arange(num_samples)
            np.random.shuffle(index)
            self.Train_Paths = np.asarray(self.Train_Paths)[index]
            self.Label_Paths = np.asarray(self.Label_Paths)[index]
            self.Valid_Paths = self.Train_Paths[:num_samples_val]
            self.Valid_Label_Paths = self.Label_Paths[:num_samples_val]
            self.Train_Paths = self.Train_Paths[num_samples_val:]
            self.Train_Label_Paths = self.Label_Paths[num_samples_val:]
