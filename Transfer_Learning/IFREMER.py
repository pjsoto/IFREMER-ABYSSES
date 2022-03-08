import os
import sys
import numpy as np
import pandas as pd
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

    def encode_single_sample(image_path):
        # 1. Read image and labels
        img = tf.io.read_file(image_path)
        # 2. Decode
        img = tf.io.decode_jpeg(img, channels=3)
        # 3. Convert to float32 in [0,1] range
        img = tf.cast(img, tf.float32)

        return {"image": img}
class OTUSIFREMER_IMAGELABEL():
    def __init__(self, args):
        labels = []
        names = []
        self.args = args
        CSV_FILE_PATH = self.args.dataset_csv_main_path + self.args.csvfile_name

        datadf=pd.read_csv(CSV_FILE_PATH)
        datadf = datadf.sample(frac=1)
        print(datadf)
        #Computing the number of classes
        Labels = datadf['Labels_IDs'].values
        for l in Labels:
            l = np.array(l[1:-1].split(', '), dtype=np.int32)
            for i in range(len(l)):
                labels.append(l[i])
        self.class_ids = np.unique(labels)
        self.class_number = len(np.unique(labels))

        labels_names = datadf['Labels_Names'].values
        for l in labels_names:
            fields = l[1:-1].split("' '")
            for i in range(len(fields)):
                if "'" not in fields[i]:
                    names.append(fields[i])
        self.class_names = np.unique(names)

        #Computing the weights
        number_samples = np.zeros((1, self.class_number))
        for label in labels:
            number_samples[0, label] += 1
        samples_proportions = number_samples/len(Labels)
        self.class_weights = 1 - samples_proportions

        if self.args.phase == 'train':
            # Taking the image names
            train_files_names = datadf[datadf['Set'] == 1]['File_Names'].values
            train_labels_ids  = datadf[datadf['Set'] == 1]['Labels_IDs'].values
            num_samples = len(train_files_names)
            index = np.arange(num_samples)
            np.random.shuffle(index)
            #Computing the number of validation samples
            num_validation_samples = int((num_samples * 20)/100)
            #Rndom shuffle the samples
            train_files_names = train_files_names[index]
            train_labels_ids = train_labels_ids[index]
            # Spliting the set in train and validation
            valid_files_names = train_files_names[:num_validation_samples]
            valid_labels_ids = train_labels_ids[:num_validation_samples]

            train_files_names = train_files_names[num_validation_samples:]
            train_labels_ids = train_labels_ids[num_validation_samples:]

            self.tr_num_samples = len(train_files_names)
            self.vl_num_samples = len(valid_files_names)

            self.Train_Paths = []
            self.Train_Labels = []

            self.Valid_Paths = []
            self.Valid_Labels = []

            for i in range(len(train_files_names)):
                self.Train_Paths.append(self.args.dataset_main_path + train_files_names[i])
                self.Train_Labels.append(self.hot_encoding(np.array(train_labels_ids[i][1:-1].split(', '), dtype=np.int32)))

            for i in range(len(valid_files_names)):
                self.Valid_Paths.append(self.args.dataset_main_path + valid_files_names[i])
                self.Valid_Labels.append(self.hot_encoding(np.array(valid_labels_ids[i][1:-1].split(', '), dtype=np.int32)))

        if self.args.phase == 'test':
            test_files_names = datadf[datadf['Set'] == 2]['File_Names'].values
            test_labels_ids  = datadf[datadf['Set'] == 2]['Labels_IDs'].values
            self.ts_num_samples = len(test_files_names)

            self.Test_Paths = []
            self.Test_Labels = []

            for i in range(len(test_files_names)):
                self.Test_Paths.append(self.args.dataset_main_path + test_files_names[i])
                self.Test_Labels.append(self.hot_encoding(np.array(test_labels_ids[i][1:-1].split(', '), dtype=np.int32)))

    def hot_encoding(self, labels):
        hot_vector = np.zeros((1, self.class_number))
        for i in range(len(labels)):
            hot_vector[0, labels[i]] = 1
        return hot_vector

    def encode_single_sample_labels(self, image_path, labels):
        # 1. Read image and labels
        img = tf.io.read_file(image_path)
        # 2. Decode
        img = tf.io.decode_jpeg(img, channels=3)
        # 3. Convert to float32 in [0,1] range
        img = tf.cast(img, tf.float32)

        return {"image": img, "label": labels}
