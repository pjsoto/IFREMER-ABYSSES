import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import applications

class Networks():
    def __init__(self, args):
        self.args = args
        if self.args.train_task == 'Semantic_Segmentation':
            if self.args.learning_model == 'Unet':
                self.learningmodel = Unet(self.args)
            if self.args.learning_model == 'DeepLab':
                print("Coming soon...")
        if self.args.train_task == 'Image_Classification':
            if self.args.backbone_name != 'None' and self.args.learning_model == 'CNN':
                self.learningmodel = CNN(self.args)

class Unet(Model):
    def __init__(self, args):
        super(Unet, self).__init__()
        self.args = args
        self.conv1 = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')
        self.conv2 = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')
        self.conv3 = layers.Conv2D(128, 3, padding = 'same', activation = 'relu')
        self.conv4 = layers.Conv2D(128, 3, padding = 'same', activation = 'relu')
        #self.pool2 = layers.MaxPooling2D(pool_size = (2 , 2))
        self.conv5 = layers.Conv2D(256, 3, padding = 'same', activation = 'relu')
        self.conv6 = layers.Conv2D(256, 3, padding = 'same', activation = 'relu')
        self.conv7 = layers.Conv2D(512, 3, padding = 'same', activation = 'relu')
        self.conv8 = layers.Conv2D(512, 3, padding = 'same', activation = 'relu')
        self.conv9 = layers.Conv2D(1024, 3, padding = 'same', activation = 'relu')
        self.conv10 = layers.Conv2D(1024, 3, padding = 'same', activation = 'relu')
        self.softmax = layers.Conv2D(self.args.classes, 1, padding = 'same', activation = 'softmax')

        self.dconv1 = layers.Conv2DTranspose(512, 2, strides = (2 , 2), padding = 'same')
        self.dconv2 = layers.Conv2DTranspose(256, 2, strides = (2 , 2), padding = 'same')
        self.dconv3 = layers.Conv2DTranspose(128, 2, strides = (2 , 2), padding = 'same')
        self.dconv4 = layers.Conv2DTranspose(64, 2,  strides = (2 , 2), padding = 'same')
        self.conv1d = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')
        self.conv2d = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')
        self.conv3d = layers.Conv2D(128, 3, padding = 'same', activation = 'relu')
        self.conv4d = layers.Conv2D(128, 3, padding = 'same', activation = 'relu')
        #self.pool2 = layers.MaxPooling2D(pool_size = (2 , 2))
        self.conv5d = layers.Conv2D(256, 3, padding = 'same', activation = 'relu')
        self.conv6d = layers.Conv2D(256, 3, padding = 'same', activation = 'relu')
        self.conv7d = layers.Conv2D(512, 3, padding = 'same', activation = 'relu')
        self.conv8d = layers.Conv2D(512, 3, padding = 'same', activation = 'relu')
        self.pool = layers.MaxPooling2D(pool_size = (2 , 2))

    def call(self, input):
        oc1  = self.conv1(input)
        oc2  = self.conv2(oc1)
        om1  = self.pool(oc2)

        oc3  = self.conv3(om1)
        oc4  = self.conv4(oc3)
        om2  = self.pool(oc4)

        oc5  = self.conv5(om2)
        oc6  = self.conv6(oc5)
        om3  = self.pool(oc6)

        oc7  = self.conv7(om3)
        oc8  = self.conv8(oc7)
        om4  = self.pool(oc8)

        oc9  = self.conv9(om4)
        oc10 = self.conv10(oc9)

        dc1 = self.dconv1(oc10)
        cn1 = tf.concat([oc8, dc1],3)
        dc2 = self.conv8d(cn1)
        dc3 = self.conv7d(dc2)

        dc4 = self.dconv2(dc3)
        cn2 = tf.concat([oc6, dc4],3)
        dc5 = self.conv6d(cn2)
        dc6 = self.conv5d(dc5)

        dc7 = self.dconv3(dc6)
        cn3 = tf.concat([oc4, dc7],3)
        dc8 = self.conv4d(cn3)
        dc9 = self.conv3d(dc8)

        dc10 = self.dconv4(dc9)
        cn4 = tf.concat([oc2, dc10],3)
        dc11 = self.conv2d(cn4)
        features = self.conv1d(dc11)

        output = self.softmax(features)
        return output, features, oc10

class CNN(Model):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        IMG_SHAPE = (None, None, 3)
        if self.args.backbone_name == 'movilenet':
            self.base_model = applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        if self.args.backbone_name == 'resnet50':
            self.base_model = applications.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

        self.globalaveragepool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(self.args.classes, activation = 'softmax')

    def call(self, input):

        out1 = self.base_model(input)
        features = self.globalaveragepool(out1)
        output = self.dense(features)

        return output, features
