import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import neptune.new as neptune
from tensorflow.keras.backend import epsilon

from SUIM import *
from Networks import *
from Augmenter import *
from Tools import *
class LearningModels():
    def __init__(self, args, dataset, run):
        self.args = args
        self.dataset = dataset
        self.args.classes = self.dataset.classes
        self.run = run
        dataset = []
        self.model = Networks(self.args)
        #ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
        self.checkpoint = tf.train.Checkpoint(net = self.model.learningmodel)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.args.save_checkpoint_path, max_to_keep = 2)
        if self.args.phase == 'train':
            self.run["args"] = self.args
            print("Defining the data augmentation procedure")
            self.aug = Augmenter(self.args)
            if self.args.optimizer == 'Adam':
                self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.args.lr, beta_1 = 0.9)
                params = {"learning_rate": self.args.lr, "optimizer": "Adam"}
            if self.args.optimizer == 'SGD':
                self.optimizer = tf.keras.optimizers.SGD(learning_rate = self.args.lr)
                params = {"learning_rate": self.args.lr, "optimizer": "SGD"}
            self.run["parameters"] = params
            #self.model.learningmodel.summary()
            self.F1_tr, self.P_tr, self.R_tr = [], [], []
            self.F1_vl, self.P_vl, self.R_vl = [], [], []
            self.train_loss, self.valid_loss = [], []
            #self.train_loss = tf.keras.metrics.Mean(name = 'train_loss')
            #self.valid_loss = tf.keras.metrics.Mean(name = 'valid_loss')
            #with open(args.save_checkpoint_path + 'modelsummary.txt', 'w') as f:
            #    self.model.learningmodel.summary(print_fn=lambda x: f.write(x + '\n'))
        if self.args.phase == 'test':
            if self.manager.latest_chekpoint:
                self.checkpoint.restore(self.manager.latest_chekpoint)
                print('[*]The trained model has been loaded sucessfully!')
            else:
                print('[!]There are no checkpoints in the current {} path'.format(self.args.save_checkpoint_path))
    #def weighted_cross_entropy_c(self, y, prediction_c, class_weights):
    #    temp = -y * tf.math.log(prediction_c + 1e-3)#[Batch_size, patch_dimension, patc_dimension, 2]
    #    temp_weighted = class_weights * temp
    #    loss = tf.reduce_sum(temp_weighted,3)
    #    return tf.reduce_mean(loss)

    def weighted_cross_entropy_c(self, y_true, y_pred, class_weights):
        epsilon_ = tf.convert_to_tensor(epsilon(), dtype=y_pred.dtype.base_dtype)
        y_pred_ = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
        cost = tf.multiply(tf.multiply(y_true, tf.math.log(y_pred_)), class_weights) #+ tf.multiply((1-y_true), tf.math.log(1-y_pred_))
        return -tf.reduce_mean(cost)

    @tf.function
    def train_step(self, data, labels, class_weights):

        with tf.GradientTape() as tape:
            predictions = self.model.learningmodel(data, training = True)
            loss = self.weighted_cross_entropy_c(labels, predictions, class_weights)
        gradients = tape.gradient(loss, self.model.learningmodel.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.learningmodel.trainable_variables))
        #self.Loss.append(loss.numpy())
        #self.train_loss(loss)
        return loss, predictions

    def test_step(self, data, labels, class_weights):
        predictions = self.model.learningmodel(data, training = False)
        loss = self.weighted_cross_entropy_c(labels, predictions, class_weights)
        #self.valid_loss(loss)
        return loss, predictions

    def Train(self):

        print("Dataset pre-processing according tensorflow methods...")
        # Pre-proecessing the Train set
        train_dataset = tf.data.Dataset.from_tensor_slices((self.dataset.Train_Paths, self.dataset.Train_Label_Paths))
        train_dataset = train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #Encoding the labels' maps
        train_dataset = train_dataset.map(lambda x: tf.py_function(self.dataset.label_encode_sample,
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
        best_F1 = 0
        for e in range(self.args.epochs):
            #self.train_loss.reset_states()
            #self.valid_loss.reset_states()
            self.F1_tr, self.P_tr, self.R_tr = [], [], []
            self.F1_vl, self.P_vl, self.R_vl = [], [], []
            self.train_loss, self.valid_loss = [], []
            for batch in train_dataset:
                images = batch["image"]
                labels = batch["label"]
                #Hot encoding the labels
                labels_ = tf.keras.utils.to_categorical(labels, self.dataset.classes)
                array = tf.constant(1, shape = [images.shape[0], self.args.crop_size, self.args.crop_size, self.args.classes], dtype = tf.float32)
                if self.args.classweight_type == 'batch':
                    #Computing class weights to mitigate the imabalance between classes
                    labels_sum = tf.reduce_sum(labels_, [0, 1, 2])/tf.constant(images.shape[0] * self.args.crop_size * self.args.crop_size, dtype = tf.float32)
                    class_weights = tf.multiply((1-labels_sum) * 10, array)
                else:
                    class_weights = tf.multiply(self.dataset.class_weights, array)

                loss, predictions = self.train_step(images, labels_, class_weights)

                y_pred = np.argmax(predictions.numpy(), axis = 3).reshape((images.shape[0] * self.args.crop_size * self.args.crop_size,1))
                y_true = labels.numpy().reshape((images.shape[0] * self.args.crop_size * self.args.crop_size,1))

                F1, P, R = compute_metrics(y_true, y_pred, 'macro')

                self.train_loss.append(loss)
                self.F1_tr.append(F1)
                self.P_tr.append(P)
                self.R_tr.append(R)

            train_loss = np.mean(self.train_loss)
            F1_mean = np.mean(self.F1_tr)
            P_mean = np.mean(self.P_tr)
            R_mean = np.mean(self.R_tr)
            self.run["train/loss"].log(train_loss)
            self.run["train/F1-Score"].log(F1_mean)
            print(f'Epoch {e + 1}, ' f'Loss: {train_loss} ,' f'Precision: {P_mean}, ' f'Recall: {R_mean}, ' f'F1-Score: {F1_mean}')

            for batch in valid_dataset:
                images = batch["image"]
                labels = batch["label"]
                labels_ = tf.keras.utils.to_categorical(labels, self.dataset.classes)
                class_weights = tf.constant(1, shape = [images.shape[0], self.args.crop_size, self.args.crop_size, self.args.classes], dtype = tf.float32)
                loss, predictions = self.test_step(images, labels_, class_weights)

                y_pred = np.argmax(predictions.numpy(), axis = 3).reshape((images.shape[0] * self.args.crop_size * self.args.crop_size,1))
                y_true = labels.numpy().reshape((images.shape[0] * self.args.crop_size * self.args.crop_size,1))

                F1, P, R = compute_metrics(y_true, y_pred, 'macro')

                self.valid_loss.append(loss)
                self.F1_vl.append(F1)
                self.P_vl.append(P)
                self.R_vl.append(R)

            valid_loss = np.mean(self.valid_loss)
            F1_mean = np.mean(self.F1_vl)
            P_mean = np.mean(self.P_vl)
            R_mean = np.mean(self.R_vl)
            self.run["valid/loss"].log(valid_loss)
            self.run["valid/F1-Score"].log(F1_mean)
            print(f'Epoch {e + 1}, ' f'Loss: {valid_loss} ,' f'Precision: {P_mean}, ' f'Recall: {R_mean}, ' f'F1-Score: {F1_mean}')

            #Saving the best model according the best value of F1-f1_score
            if F1_mean > best_F1:
                print('Model has improved from {} to {}'.format(best_F1, F1_mean))
                best_F1 = F1_mean
                validation_counter = 0
                ckpt_save_path = self.manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(e + 1, ckpt_save_path))
            else:
                validation_counter += 1
                if validation_counter >= 150:
                    break
                #f, axarr = plt.subplots(1,2)
                #axarr[0].imshow(images[0,:,:,:])
                #axarr[1].imshow(labels[0,:,:,])
                #plt.show()
