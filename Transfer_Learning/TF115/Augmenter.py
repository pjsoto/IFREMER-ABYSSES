import os
import sys
import json
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class Augmenter():
    def __init__(self, args):
        self.args = args
        if self.args.phase == 'train':
            self.ss_seq = iaa.Sequential([
                          iaa.CropToFixedSize(width=self.args.crop_size_cols, height=self.args.crop_size_rows),
                          iaa.Rot90((1,3)),
                       ])
            self.ic_seq = iaa.Sequential([
                          iaa.size.Resize({"height": self.args.crop_size_rows, "width": self.args.crop_size_cols}),
                          iaa.Rot90((1,3)),
                       ])
        if self.args.phase == 'test':
            if self.args.testcrop_size_cols is not None and self.args.testcrop_size_rows is not None:
                self.ss_seq = iaa.Sequential([
                           iaa.CropToFixedSize(width=self.args.testcrop_size_cols, height=self.args.testcrop_size_rows)
                           ])
                self.ic_seq = iaa.Sequential([
                            iaa.size.Resize({"height": self.args.testcrop_size_rows, "width": self.args.testcrop_size_cols})
                            ])

    def apply_augmentations_train(self, image, label):
        image_numpy = image.numpy()
        label_numpy = label.numpy()
        if self.args.train_task == 'Semantic_Segmentation':
            segmap = SegmentationMapsOnImage(label_numpy, shape=image_numpy.shape)
            image_reshape = image_numpy.reshape((1, image_numpy.shape[0], image_numpy.shape[1], image_numpy.shape[2]))
            images_aug, labels_aug = self.ss_seq(images = image_reshape, segmentation_maps = segmap)
            return (images_aug[0], labels_aug.get_arr())
        if self.args.train_task == 'Image_Classification':
            image_reshape = image_numpy.reshape((1, image_numpy.shape[0], image_numpy.shape[1], image_numpy.shape[2]))
            images_aug = self.ic_seq(images = image_reshape)
            return (images_aug[0], label_numpy)

    def apply_augmentations_test(self, image, label):
        image_numpy = image.numpy()
        label_numpy = label.numpy()
        image_reshape = image_numpy.reshape((1, image_numpy.shape[0], image_numpy.shape[1], image_numpy.shape[2]))
        if self.args.test_task_level == 'Pixel_Level':
            images_aug = self.ss_seq(images = image_reshape)
        if self.args.test_task_level == 'Image_Level':
            images_aug = self.ic_seq(images = image_reshape)
        return (images_aug[0], label_numpy)
