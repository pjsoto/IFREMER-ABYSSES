import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import neptune.new as neptune

from SUIM import *
from Networks import *
from Augmenter import *
from Tools import *
class LearningModels():
    def __init__(self, args, dataset, run):
        self.args = args
        self.dataset = dataset
        self.run = run
        dataset = []
        self.model = Networks(self.args)
        if self.args.phase == 'train':
            print("Initalizing Neptune")

            params = {"learning_rate": 0.001, "optimizer": "Adam"}
            self.run["parameters"] = params
            self.run["args"] = self.args
            print("Defining the data augmentation procedure")
            self.aug = Augmenter(self.args)

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            #self.model.learningmodel.summary()
            self.F1_tr, self.P_tr, self.R_tr = [], [], []
            self.F1_vl, self.P_vl, self.R_vl = [], [], []
            self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
            self.valid_loss = tf.keras.metrics.Mean(name = 'valid_loss')
            #with open(args.save_checkpoint_path + 'modelsummary.txt', 'w') as f:
            #    self.model.learningmodel.summary(print_fn=lambda x: f.write(x + '\n'))

    def weighted_cross_entropy_c(self, y, prediction_c, class_weights):
        temp = -y * tf.math.log(prediction_c + 1e-3)#[Batch_size, patch_dimension, patc_dimension, 2]
        temp_weighted = class_weights * temp
        loss = tf.reduce_sum(temp_weighted,3)
        return tf.reduce_mean(loss)

    def train_step(self, data, labels, class_weights):

        with tf.GradientTape() as tape:
            predictions = self.model.learningmodel(data, training = True)
            loss = self.weighted_cross_entropy_c(labels, predictions, class_weights)
        gradients = tape.gradient(loss, self.model.learningmodel.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.learningmodel.trainable_variables))
        #self.Loss.append(loss.numpy())
        y_pred = np.argmax(predictions.numpy(), axis = 3).reshape((self.args.batch_size * self.args.crop_size * self.args.crop_size,1))
        y_true = np.argmax(labels, axis = 3).reshape((self.args.batch_size * self.args.crop_size * self.args.crop_size,1))
        F1, P, R = compute_metrics(y_true, y_pred, 'macro')
        self.F1_tr.append(F1)
        self.P_tr.append(P)
        self.R_tr.append(R)
        self.train_loss(loss)

    def test_step(self, data, labels, class_weights):
        predictions = self.model.learningmodel(data, training = False)
        loss = self.weighted_cross_entropy_c(labels, predictions, class_weights)

        y_pred = np.argmax(predictions.numpy(), axis = 3).reshape((predictions.shape[0] * self.args.crop_size * self.args.crop_size,1))
        y_true = np.argmax(labels, axis = 3).reshape((predictions.shape[0] * self.args.crop_size * self.args.crop_size,1))
        F1, P, R = compute_metrics(y_true, y_pred, 'macro')

        self.F1_vl.append(F1)
        self.P_vl.append(P)
        self.R_vl.append(R)
        self.valid_loss(loss)

    def Train(self):

        print("Dataset pre-processing according tensorflow methods...")
        # Pre-proecessing the Train set
        train_dataset = tf.data.Dataset.from_tensor_slices((self.dataset.Train_Paths, self.dataset.Train_Label_Paths))
        train_dataset = train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #Encoding the labels' maps
        train_dataset = train_dataset.map(lambda x: tf.py_function(label_encode_sample,
                                                                    inp=[x['image'], x['label']],
                                                                    Tout=(tf.float32, tf.int32))).map(create_dict)
        #Applying Data augmrntation transformation to the images and as well to the labels
        train_dataset = train_dataset.map(lambda x: tf.py_function(self.aug.apply_augmentations,
                                                                   inp = [x['image'], x['label']],
                                                                   Tout = (tf.float32, tf.int32))).map(create_dict)

        train_dataset = (train_dataset.batch(self.args.batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE))

        # Pre-proecessing the Validation set
        valid_dataset = tf.data.Dataset.from_tensor_slices((self.dataset.Valid_Paths, self.dataset.Valid_Label_Paths))
        valid_dataset = valid_dataset.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #Encoding the labels' maps
        valid_dataset = valid_dataset.map(lambda x: tf.py_function(label_encode_sample,
                                                                    inp=[x['image'], x['label']],
                                                                    Tout=(tf.float32, tf.int32))).map(create_dict)
        #Applying Data augmrntation transformation to the images and as well to the labels
        valid_dataset = valid_dataset.map(lambda x: tf.py_function(self.aug.apply_augmentations,
                                                                   inp = [x['image'], x['label']],
                                                                   Tout = (tf.float32, tf.int32))).map(create_dict)

        valid_dataset = (valid_dataset.batch(self.args.batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE))

        #Loop for epochs
        #self.args.epochs = 3
        counter = 0
        for e in range(self.args.epochs):
            self.train_loss.reset_states()
            self.valid_loss.reset_states()
            self.F1_tr, self.P_tr, self.R_tr = [], [], []
            self.F1_vl, self.P_vl, self.R_vl = [], [], []
            for batch in train_dataset:
                images = batch["image"]
                labels = batch["label"]
                #Hot encoding the labels
                labels_ = tf.keras.utils.to_categorical(labels, self.args.classes)
                #Computing class weights to mitigate the imabalance between classes
                labels_sum = tf.reduce_sum(labels_, [0, 1, 2])/tf.constant(images.shape[0] * self.args.crop_size * self.args.crop_size,dtype = tf.float32)
                array = tf.constant(1, shape = [images.shape[0], self.args.crop_size, self.args.crop_size, self.args.classes], dtype = tf.float32)
                class_weights = tf.multiply(labels_sum, array)
                self.train_step(images, labels_, class_weights)

            F1_mean = np.mean(self.F1_tr)
            P_mean = np.mean(self.P_tr)
            R_mean = np.mean(self.R_tr)
            self.run["train/loss"].log(self.train_loss.result())
            self.run["train/F1-Score"].log(F1_mean)
            print(f'Epoch {e + 1}, ' f'Loss: {self.train_loss.result()} ,' f'Precision: {P_mean}, ' f'Recall: {R_mean}, ' f'F1-Score: {F1_mean}')

            for batch in valid_dataset:
                images = batch["image"]
                labels = batch["label"]
                labels_ = tf.keras.utils.to_categorical(labels, self.args.classes)
                class_weights = tf.constant(1, shape = [images.shape[0], self.args.crop_size, self.args.crop_size, self.args.classes], dtype = tf.float32)
                self.test_step(images, labels_, class_weights)

            F1_mean = np.mean(self.F1_vl)
            P_mean = np.mean(self.P_vl)
            R_mean = np.mean(self.R_vl)
            self.run["valid/loss"].log(self.train_loss.result())
            self.run["valid/F1-Score"].log(F1_mean)
            print(f'Epoch {e + 1}, ' f'Loss: {self.valid_loss.result()} ,' f'Precision: {P_mean}, ' f'Recall: {R_mean}, ' f'F1-Score: {F1_mean}')

                #f, axarr = plt.subplots(1,2)
                #axarr[0].imshow(images[0,:,:,:])
                #axarr[1].imshow(labels[0,:,:,])
                #plt.show()
