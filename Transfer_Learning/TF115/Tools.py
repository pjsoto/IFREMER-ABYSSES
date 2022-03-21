import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def preprocess_input(data, backbone_name):
    if backbone_name is None:
        data = data/255.
    elif backbone_name == 'movilenet':
        data = tf.keras.applications.mobilenet.preprocess_input(data)
    elif backbone_name == 'resnet50':
        data = tf.keras.applications.resnet.preprocess_input(data)
    elif backbone_name == 'vgg16':
        data = tf.keras.applications.vgg16.preprocess_input(data)

    return data

def create_dict(image, label):
    return {"image": image, "label": label}

def compute_metrics(y_true, y_pred, average):
    accuracy = 100*accuracy_score(y_true, y_pred)
    f1score = 100*f1_score(y_true, y_pred, average=average, zero_division = 1)
    recall = 100*recall_score(y_true, y_pred,average=average, zero_division = 1)
    precision = 100*precision_score(y_true, y_pred,average=average, zero_division = 1)
    return accuracy, f1score, precision, recall

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
