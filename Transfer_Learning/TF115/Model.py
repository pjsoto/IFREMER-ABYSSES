import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from tqdm import trange
from sklearn.manifold import TSNE
from tensorflow.keras.backend import epsilon

from IFREMER import *
from Networks import *
from Augmenter import *
from Tools import *

class Model():
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.args.class_number = self.dataset.class_number
        # Initializing the placeholders
        #Changing  the seed  at any run
        tf.set_random_seed(int(time.time()))
        tf.reset_default_graph()

        self.data = tf.placeholder(tf.float32, [None, self.args.new_size_rows, self.args.new_size_cols, self.args.image_channels], name = "data")
        self.class_weights = tf.placeholder(tf.float32, [None, self.args.class_number], name="class_weights")
        self.label = tf.placeholder(tf.float32, [None, self.args.class_number], name = "label")
        self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")

        self.model = Networks(self.args)
        Classifier_Outputs = self.model.learningmodel.build_Model(self.data, reuse = False, name = "CNN" )
        self.logits_c = Classifier_Outputs[-2]
        self.prediction_c = Classifier_Outputs[-1]
        self.features = Classifier_Outputs[self.args.layer_index]
        self.feature_shape = self.features.get_shape().as_list()[1:]

        if self.args.phase == 'train':
            self.summary(Classifier_Outputs, 'Classifier:')
            #Defining the data augmentation object
            self.augmenter = Augmenter(self.args)
            if self.args.labels_type == 'onehot_labels':
                self.classifier_loss = self.weighted_cat_cross_entropy(self.label, self.prediction_c, self.class_weights)
            elif self.args.labels_type == 'multiple_labels':
                self.classifier_loss = self.weighted_bin_cross_entropy(self.label, self.prediction_c, self.class_weights)

            if self.args.optimizer == 'MomentumOptimizer':
                self.training_optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.classifier_loss)
            elif self.args.optimizer == 'Adam':
                self.training_optimizer = tf.train.AdamOptimizer(self.learning_rate, 0.9).minimize(self.classifier_loss)
            else:
                print("Such optiizer has not been included in this implementation")
                sys.exit()

            self.saver = tf.train.Saver(max_to_keep=5)
            self.sess=tf.Session()
            self.sess.run(tf.initialize_all_variables())

        elif self.args.phase == 'test':
            self.dataset = dataset
            self.saver = tf.train.Saver(max_to_keep=5)
            self.sess=tf.Session()
            self.sess.run(tf.initialize_all_variables())
            print('[*]Loading the feature extractor and classifier trained models...')
            mod = self.load(self.args.trained_model_path)
            if mod:
                print(" [*] Load with SUCCESS")
            else:
                print(" [!] Load failed...")
                sys.exit()

    def weighted_cat_cross_entropy(self, y_true, y_pred, class_weights):
        epsilon_ = tf.convert_to_tensor(epsilon(), dtype=y_pred.dtype.base_dtype)
        y_pred_ = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
        cost = tf.multiply(tf.multiply(y_true, tf.math.log(y_pred_)), class_weights)
        if self.args.train_task == 'Semantic_Segmentation':
            cost = tf.reduce_sum(cost, axis = 3)
        elif self.args.train_task == 'Image_Classification':
            cost = tf.reduce_sum(cost, axis = 1)
        return -tf.reduce_mean(cost)

    def weighted_bin_cross_entropy(self, y_true, y_pred, class_weights):
        epsilon_ = tf.convert_to_tensor(epsilon(), dtype=y_pred.dtype.base_dtype)
        y_pred_ = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
        cost = tf.multiply(tf.multiply(y_true, tf.math.log(y_pred_)), class_weights) + tf.multiply((1-y_true), tf.math.log(1-y_pred_))
        if self.args.train_task == 'Semantic_Segmentation':
            cost = tf.reduce_sum(cost, axis = 3)
        elif self.args.train_task == 'Image_Classification':
            cost = tf.reduce_sum(cost, axis = 1)
        return -tf.reduce_mean(cost)

    def Learning_rate_decay(self):
        lr = self.args.lr / (1. + 10 * self.p)**0.75
        return lr

    def summary(self, net, name):
        print(net)
        f = open(self.args.save_checkpoint_path + "Architecture.txt","a")
        f.write(name + "\n")
        for i in range(len(net)):
            print(net[i].get_shape().as_list())
            f.write(str(net[i].get_shape().as_list()) + "\n")
        f.close()

    def save(self, checkpoint_dir, epoch):
        model_name = "DANN.model"

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=epoch)
        print("Checkpoint Saved with SUCCESS!")

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        print(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            aux = 'model_example'
            for i in range(len(ckpt_name)):
                if ckpt_name[-i-1] == '-':
                    aux = ckpt_name[-i:]
                    break
            return aux
        else:
            return ''
    def tsne_features(self, data):
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)

        features_tsne_proj = tsne.fit_transform(data)
        features_tsne_proj_min, features_tsne_proj_max = np.min(features_tsne_proj, 0), np.max(features_tsne_proj, 0)
        features_tsne_proj = (features_tsne_proj - features_tsne_proj_min) / (features_tsne_proj_max - features_tsne_proj_min)

        return features_tsne_proj

    def Train(self):
        best_val_fs = 0
        pat = 0

        #Computing the number of batches
        num_batches_tr = len(self.dataset.Train_Paths)//self.args.batch_size
        num_batches_vl = len(self.dataset.Valid_Paths)//self.args.batch_size

        e = 0
        while (e < self.args.epochs):
            #Shuffling the data and the labels
            num_samples = len(self.dataset.Train_Paths)
            index = np.arange(num_samples)
            np.random.shuffle(index)
            self.dataset.Train_Paths = np.array(self.dataset.Train_Paths)[index]
            self.dataset.Train_Labels = np.array(self.dataset.Train_Labels)[index]
            self.dataset.Train_Coordinates = np.array(self.dataset.Train_Coordinates)[index]

            num_samples = len(self.dataset.Valid_Paths)
            index = np.arange(num_samples)
            np.random.shuffle(index)

            self.dataset.Valid_Paths = np.array(self.dataset.Valid_Paths)[index]
            self.dataset.Valid_Labels = np.array(self.dataset.Valid_Labels)[index]
            self.dataset.Valid_Coordinates = np.array(self.dataset.Valid_Coordinates)[index]

            f = open(self.args.save_checkpoint_path + "Log.txt","a")
            #Initializing loss metrics
            loss_cl_tr = np.zeros((1 , 2))
            loss_cl_vl = np.zeros((1 , 2))

            self.Ac_tr, self.F1_tr, self.P_tr, self.R_tr = [], [], [], []
            self.Ac_vl, self.F1_vl, self.P_vl, self.R_vl = [], [], [], []
            self.train_loss, self.valid_loss = [], []

            #Computing some parameters
            if self.args.learning_ratedecay:
                self.p = float(e) / self.args.epochs
                print("Percentage of epochs: " + str(self.p))
                self.lr = self.Learning_rate_decay()
                print("Learning rate decay: " + str(self.lr))


            batchs = trange(num_batches_tr)
            print('Training procedure...')
            for b in batchs:

                trainpaths_batch = self.dataset.Train_Paths[b * self.args.batch_size : (b + 1) * self.args.batch_size]
                trainlabels_batch = self.dataset.Train_Labels[b * self.args.batch_size : (b + 1) * self.args.batch_size]
                traincoordinates_batch = self.dataset.Train_Coordinates[b * self.args.batch_size : (b + 1) * self.args.batch_size]

                data_batch, labels_batch = self.dataset.read_samples(trainpaths_batch, trainlabels_batch, traincoordinates_batch, self.dataset.pad_tuple)

                if self.args.data_augmentation:
                    data_batch = self.augmenter.apply(data_batch)


                array = np.ones((data_batch.shape[0], self.args.class_number), dtype=np.float32)
                if self.args.weights_definition == 'automatic':
                    weights = self.dataset.class_weights * array
                if self.args.weights_definition == 'manual':
                    weights = [1, 1, 1] * array

                _, batch_loss, batch_prediction = self.sess.run([self.training_optimizer, self.classifier_loss, self.prediction_c],
                                                                feed_dict={self.data: data_batch, self.label: labels_batch,
                                                                           self.class_weights: weights, self.learning_rate: self.lr})
                #Computing the Metrics
                if self.args.labels_type == 'onehot_labels':
                    y_pred = np.argmax(batch_prediction, axis = 1)
                    y_true = np.argmax(labels_batch, axis = 1)

                    Ac, F1, P, R = compute_metrics(y_true, y_pred, 'macro')
                if self.args.labels_type == 'multiple_labels':
                    y_pred = ((batch_prediction > 0.5) * 1.0)
                    y_true = labels_batch

                    Ac, F1, P, R = compute_metrics(y_true, y_pred, 'macro')

                    F1 = np.mean(F1)
                    P = np.mean(P)
                    R = np.mean(R)

                self.train_loss.append(batch_loss)
                self.Ac_tr.append(Ac)
                self.F1_tr.append(F1)
                self.P_tr.append(P)
                self.R_tr.append(R)

            train_loss = np.mean(self.train_loss)
            Ac_mean = np.mean(self.Ac_tr)
            F1_mean = np.mean(self.F1_tr)
            P_mean = np.mean(self.P_tr)
            R_mean = np.mean(self.R_tr)
            #self.run["train/loss"].log(train_loss)
            #self.run["train/F1-Score"].log(F1_mean)
            #self.run["train/Accuracy"].log(Ac_mean)
            print(f'Epoch {e + 1}, ' f'Loss: {train_loss} ,' f'Precision: {P_mean}, ' f'Recall: {R_mean}, ' f'F1-Score: {F1_mean}, ' f'Accuracy: {Ac_mean}')
            f.write("%d [Tr loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (e, train_loss, Ac_mean, P_mean, R_mean, F1_mean))

            print('Computing the validation metrics...')
            if self.args.feature_representation:
                features = np.zeros((self.args.batch_size * num_batches_vl, np.prod(self.feature_shape)))
                true_labels = np.zeros((self.args.batch_size * num_batches_vl, 1))
            batchs = trange(num_batches_vl)
            for b in batchs:

                validpaths_batch = self.dataset.Valid_Paths[b * self.args.batch_size : (b + 1) * self.args.batch_size]
                validlabels_batch = self.dataset.Valid_Labels[b * self.args.batch_size : (b + 1) * self.args.batch_size]
                validcoordinates_batch = self.dataset.Valid_Coordinates[b * self.args.batch_size : (b + 1) * self.args.batch_size]

                data_batch, labels_batch = self.dataset.read_samples(validpaths_batch, validlabels_batch, validcoordinates_batch, self.dataset.pad_tuple)

                if self.args.data_augmentation:
                    data_batch = self.augmenter.apply(data_batch)


                array = np.ones((data_batch.shape[0], self.args.class_number), dtype=np.float32)
                if self.args.weights_definition == 'automatic':
                    weights = self.dataset.class_weights * array
                if self.args.weights_definition == 'manual':
                    weights = [1, 1, 1] * array

                batch_loss, batch_prediction, batch_features = self.sess.run([self.classifier_loss, self.prediction_c, self.features],
                                                             feed_dict={self.data: data_batch, self.label: labels_batch,
                                                                           self.class_weights: weights})
                #Computing the Metrics
                if self.args.labels_type == 'onehot_labels':
                    y_pred = np.argmax(batch_prediction, axis = 1)
                    y_true = np.argmax(labels_batch, axis = 1)

                    Ac, F1, P, R = compute_metrics(y_true, y_pred, 'macro')
                if self.args.labels_type == 'multiple_labels':
                    y_pred = ((batch_prediction > 0.5) * 1.0)
                    y_true = labels_batch

                    Ac, F1, P, R = compute_metrics(y_true, y_pred, 'macro')

                    F1 = np.mean(F1)
                    P = np.mean(P)
                    R = np.mean(R)

                self.valid_loss.append(batch_loss)
                self.Ac_vl.append(Ac)
                self.F1_vl.append(F1)
                self.P_vl.append(P)
                self.R_vl.append(R)

                if self.args.feature_representation:
                    true_labels[b * self.args.batch_size : (b + 1) * self.args.batch_size, 0] = y_true
                    if len(self.feature_shape) > 2:
                        features[b * self.args.batch_size : (b + 1) * self.args.batch_size, :] = batch_features.reshape((self.batch_features.shape[0], self.batch_features.shape[1] * self.batch_features.shape[2] * self.batch_features.shape[3]))
                    else:
                        features[b * self.args.batch_size : (b + 1) * self.args.batch_size, :] = batch_features

            valid_loss = np.mean(self.valid_loss)
            Ac_mean = np.mean(self.Ac_vl)
            F1_mean = np.mean(self.F1_vl)
            P_mean = np.mean(self.P_vl)
            R_mean = np.mean(self.R_vl)
            #self.run["train/loss"].log(train_loss)
            #self.run["train/F1-Score"].log(F1_mean)
            #self.run["train/Accuracy"].log(Ac_mean)
            print(f'Epoch {e + 1}, ' f'Loss: {valid_loss} ,' f'Precision: {P_mean}, ' f'Recall: {R_mean}, ' f'F1-Score: {F1_mean}, ' f'Accuracy: {Ac_mean}')
            f.write("%d [Vl loss: %f, acc.: %.2f%%, precission: %.2f%%, recall: %.2f%%, fscore: %.2f%%]\n" % (e, valid_loss, Ac_mean, P_mean, R_mean, F1_mean))
            if best_val_fs < F1_mean:
                best_val_fs = F1_mean
                pat = 0
                print('[!]Saving best model ...')
                self.save(self.args.save_checkpoint_path, e)
            else:
                pat += 1
                if pat > self.args.patience:
                    break
            # plotting the representation if required
            if self.args.feature_representation:
                features_projected = self.tsne_features(features)
                plottsne_features(features_projected, true_labels, save_path = self.args.save_checkpoint_path , epoch = e, USE_LABELS = True)
            e += 1
