import os
import sys
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from Tools import *

class OTUSIFREMER_IMAGELABEL():
    def __init__(self, args):
        labels = []
        names = []
        self.args = args
        CSV_FILE_PATH = self.args.dataset_csv_main_path + self.args.csvfile_name

        if 'Shells_White_fragments' in self.args.csvfile_name:
            self.class_names = ['0-10%','10-50%','50-100%']
        if 'Lithology' in self.args.csvfile_name:
            self.class_names = ['Slab','Sulfurs','Vocanoclastic']
        if 'Morphology' in self.args.csvfile_name:
            self.class_names = ['Fractured','Marbled','Scree/rubbles', 'Sedimented']


        datadf=pd.read_csv(CSV_FILE_PATH)
        datadf = datadf.sample(frac=1)
        #Computing the number of classes
        Labels = datadf['Labels_IDs'].values
        for l in Labels:
            l = np.array(l[1:-1].split(' '), dtype=np.int32)
            for i in range(len(l)):
                labels.append(l[i])

        self.class_ids = np.unique(labels)
        self.class_number = len(self.class_ids)
        #Computing the weights
        number_samples = np.zeros((1, self.class_number))
        for label in labels:
            number_samples[0, label] += 1
        print(number_samples)
        samples_proportions = number_samples/len(Labels)
        print(samples_proportions)
        self.class_weights = 1/samples_proportions
        print(self.class_weights)
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
            self.Train_Coordinates = []

            self.Valid_Paths = []
            self.Valid_Labels = []
            self.Valid_Coordinates = []

            if self.args.split_patch:
                image_dimensions = [self.args.image_rows, self.args.image_cols]
                patch_dimension = self.args.new_size_rows
                coordinates, self.pad_tuple = self.corner_coordinates_definition(image_dimensions, patch_dimension, self.args.overlap_porcent)
                for i in range(len(train_files_names)):
                    for j in range(coordinates.shape[0]):
                        self.Train_Paths.append(self.args.dataset_main_path + train_files_names[i])
                        self.Train_Labels.append(self.hot_encoding(np.array(train_labels_ids[i][1:-1].split(' '), dtype=np.int32)))
                        self.Train_Coordinates.append(coordinates[j, :])

                for i in range(len(valid_files_names)):
                    for j in range(coordinates.shape[0]):
                        self.Valid_Paths.append(self.args.dataset_main_path + valid_files_names[i])
                        self.Valid_Labels.append(self.hot_encoding(np.array(valid_labels_ids[i][1:-1].split(' '), dtype=np.int32)))
                        self.Valid_Coordinates.append(coordinates[j, :])

            else:
                self.pad_tuple = []
                for i in range(len(train_files_names)):
                    self.Train_Paths.append(self.args.dataset_main_path + train_files_names[i])
                    self.Train_Labels.append(self.hot_encoding(np.array(train_labels_ids[i][1:-1].split(' '), dtype=np.int32)))
                    self.Train_Coordinates.append([])
                for i in range(len(valid_files_names)):
                    self.Valid_Paths.append(self.args.dataset_main_path + valid_files_names[i])
                    self.Valid_Labels.append(self.hot_encoding(np.array(valid_labels_ids[i][1:-1].split(' '), dtype=np.int32)))
                    self.Valid_Coordinates.append([])

        if self.args.phase == 'test' or self.args.phase == 'gradcam':
            self.Labels_Available = True
            test_files_names = datadf[datadf['Set'] == 2]['File_Names'].values
            test_labels_ids  = datadf[datadf['Set'] == 2]['Labels_IDs'].values
            self.ts_num_samples = len(test_files_names)

            self.Test_Paths = []
            self.Test_Labels = []

            for i in range(len(test_files_names)):
                self.Test_Paths.append(self.args.dataset_main_path + test_files_names[i])
                self.Test_Labels.append(self.hot_encoding(np.array(test_labels_ids[i][1:-1].split(' '), dtype=np.int32)))

    def hot_encoding(self, labels):
        hot_vector = np.zeros((1, self.class_number))
        for i in range(len(labels)):
            hot_vector[0, labels[i]] = 1
        return hot_vector

    def read_samples(self, image_paths, hot_labels, coordinates, pad_tuple):
        # 1. Read image and labels
        images = np.zeros((self.args.batch_size, self.args.new_size_rows, self.args.new_size_cols, self.args.image_channels))
        labels = np.zeros((self.args.batch_size, self.class_number))

        for i in range(len(image_paths)):
            image = self.preprocess_input(np.array(Image.open(image_paths[i]), dtype=np.float32), self.args.backbone_name)
            labels[i, :] = hot_labels[i]

            if np.sum(coordinates[i]) == 0.0:
                images[i, :, :, :] = image
            else:
                #Applying the padding
                c = coordinates[i]
                image = np.pad(image, pad_tuple, mode='symmetric')
                images[i, :, :, :] = image[int(c[0]) : int(c[2]) , int(c[1]) : int(c[3]),:]

        return images, labels

    def corner_coordinates_definition(self, image_dimensions, patch_dimension, overlap_porcent):

        rows = image_dimensions[0]
        cols = image_dimensions[1]

        # Computing the overlaps and other things to extract patches
        overlap = round(patch_dimension * overlap_porcent)
        overlap -= overlap % 2
        stride = patch_dimension - overlap
        step_row = (stride - rows % stride) % stride
        step_col = (stride - cols % stride) % stride

        k1, k2 = (rows + step_row)//stride, (cols + step_col)//stride

        #Taking the initial coordinates
        coordinates = np.zeros((k1 * k2 , 4))
        counter = 0
        for i in range(k1):
            for j in range(k2):
                coordinates[counter, 0] = i * stride
                coordinates[counter, 1] = j * stride
                coordinates[counter, 2] = i * stride + patch_dimension
                coordinates[counter, 3] = j * stride + patch_dimension
                counter += 1

        pad_tuple = ((overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col), (0 , 0))

        return coordinates, pad_tuple

    def preprocess_input(self, data, backbone_name):
        if backbone_name == 'movilenet':
            data = tf.keras.applications.mobilenet.preprocess_input(data)
        elif backbone_name == 'resnet50':
            data = tf.keras.applications.resnet.preprocess_input(data)
        elif backbone_name == 'vgg16':
            data = tf.keras.applications.vgg16.preprocess_input(data)
        else:
            data = data/255.

        return data
