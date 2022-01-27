import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def encode_single_sample(image_path, label_path):
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

def create_dict(image, label):
    return {"image": image, "label": label}

def compute_metrics(y_true, y_pred, average):
    f1score = 100*f1_score(y_true, y_pred, average=average, zero_division = 1)
    recall = 100*recall_score(y_true, y_pred,average=average, zero_division = 1)
    precision = 100*precision_score(y_true, y_pred,average=average, zero_division = 1)
    return f1score, precision, recall
