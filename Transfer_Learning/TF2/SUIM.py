import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

from Tools import *

class SUIM():
    def __init__(self, args):

        self.args = args
        self.class_number = 8
        if self.args.class_grouping:
            self.class_number = 7

        images_dimensions = 0
        labels_sum = tf.constant(0, shape = [1, self.classes], dtype = tf.float32)

        if self.args.phase == 'train':
            self.Train_Paths = []
            self.Label_Paths = []
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
                            labels_ = tf.keras.utils.to_categorical(self.Label_Converter(label), self.class_number)
                            #Computing class weights to mitigate the imabalance between classes
                            labels_sum += (tf.reduce_sum(labels_, [0, 1])/tf.constant(np.shape(image)[0] * np.shape(image)[1], dtype = tf.float32))
                            images_counter += 1
            if self.args.classweight_type == 'global':
                print(labels_sum/tf.constant(images_counter, dtype = tf.float32))
                print(1 - labels_sum/tf.constant(images_counter, dtype = tf.float32))
                self.class_weights = (1 - (labels_sum/tf.constant(images_counter, dtype = tf.float32)))*10
            elif self.args.classweight_type == 'None':
                self.class_weights = tf.constant(1, shape = [1, self.classes], dtype = tf.float32)
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

        if self.args.phase == 'test':
            self.Testi_Paths = []
            self.Label_Paths = []
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

                        self.Testi_Paths.append(image_path)
                        self.Label_Paths.append(label_path)

    def Label_Converter(self, rgb_label):

        Label_reshaped = rgb_label.reshape((rgb_label.shape[0] * rgb_label.shape[1], rgb_label.shape[2]))
        Label_reshaped_ = Label_reshaped.copy()
        Label_reshaped_[Label_reshaped>=200]=1
        Label_reshaped_[Label_reshaped <200]=0

        label = 4 * Label_reshaped_[:,0] + 2 * Label_reshaped_[:,1] + 1 * Label_reshaped_[:,2]
        if self.args.class_grouping:
            label[label == 2] = 0
            label[label == 7] = 0
        return label.reshape((rgb_label.shape[0] , rgb_label.shape[1]))

    def label_encode_sample(self, image, label):
        image = image.numpy()
        label = label.numpy()
        label = self.Label_Converter(label)
        return (image, label)

    def encode_single_sample_labels(self, image_path, label_path):
        # 1. Read image and labels
        img = tf.io.read_file(image_path)
        lbl = tf.io.read_file(label_path)
        # 2. Decode
        img = tf.io.decode_jpeg(img, channels=3)
        lbl = tf.io.decode_bmp(lbl, channels=3)
        # 3. Convert to float32 in [0,1] range
        img = tf.cast(img, tf.float32)

        return {"image": img, "label": lbl}
