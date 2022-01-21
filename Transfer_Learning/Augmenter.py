import os
import sys
import json
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class Augmenter():
    def __init__(self, args):
        self.args = args
        self.seq = iaa.Sequential([
                   iaa.CropToFixedSize(width=self.args.crop_size, height=self.args.crop_size),
                   iaa.Sharpen((0.0, 1.0)),
                   iaa.Affine(rotate=(-45, 45))
                   ])
    def apply_augmentations(self, image, label):
        image_numpy = image.numpy()
        label_numpy = label.numpy()
        segmap = SegmentationMapsOnImage(label_numpy, shape=image_numpy.shape)
        image_reshape = image_numpy.reshape((1, image_numpy.shape[0], image_numpy.shape[1], image_numpy.shape[2]))
        images_aug, labels_aug = self.seq(images = image_reshape, segmentation_maps = segmap)
        return (images_aug[0], labels_aug.get_arr())
