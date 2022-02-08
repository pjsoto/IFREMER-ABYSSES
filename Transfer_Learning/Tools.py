import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def encode_single_sample_labels(image_path, label_path):
    # 1. Read image and labels
    img = tf.io.read_file(image_path)
    lbl = tf.io.read_file(label_path)
    # 2. Decode
    img = tf.io.decode_jpeg(img, channels=3)
    lbl = tf.io.decode_bmp(lbl, channels=3)
    # 3. Convert to float32 in [0,1] range
    img = tf.cast(img, tf.float32)
    img = img/255.

    return {"image": img, "label": lbl}

def encode_single_sample(image_path):
    # 1. Read image and labels
    img = tf.io.read_file(image_path)
    # 2. Decode
    img = tf.io.decode_jpeg(img, channels=3)
    # 3. Convert to float32 in [0,1] range
    img = tf.cast(img, tf.float32)
    img = img/255.

    return {"image": img}

def create_dict(image, label):
    return {"image": image, "label": label}

def compute_metrics(y_true, y_pred, average):
    f1score = 100*f1_score(y_true, y_pred, average=average, zero_division = 1)
    recall = 100*recall_score(y_true, y_pred,average=average, zero_division = 1)
    precision = 100*precision_score(y_true, y_pred,average=average, zero_division = 1)
    return f1score, precision, recall

def plottsne_features(features, labels, save_path, USE_LABELS = True):
    colors = []
    plt.figure(figsize=(20,20))
    ax = plt.subplot(111)
    if USE_LABELS:
        colors.append('#1724BD')
        colors.append('#0EB7C2')
        colors.append('#BF114B')
        colors.append('#E98E2C')
        colors.append('#008f39')
        colors.append('#663300')
        colors.append('#8D1354')
        for i in range(features.shape[0]):
            # plot colored number
            ax.plot(features[i, 0], features[i, 1], marker='o',
                      color=colors[int(labels[i,0])])
    else:
        for i in range(features.shape[0]):
            # plot colored number
            ax.plot(features[i, 0], features[i, 1], marker='o',
                      color='b')

    plt.savefig(save_path + 'scatter_plot.png')
    plt.clf()
