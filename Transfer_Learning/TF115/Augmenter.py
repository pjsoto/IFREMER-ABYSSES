import os
import sys
import json
import numpy as np
#from imgaug import augmenters as iaa
#from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from PIL import Image
class Augmenter():
    def __init__(self, args):
        self.args = args

    def apply(self, data):
        if data.shape[1] != data.shape[2]:
            size = data.shape[1] if data.shape[1] < data.shape[2] else data.shape[2]
            data_ = np.zeros((data.shape[0], size, size, data.shape[3]))
            for i in range(data.shape[0]):
                if data.shape[1] < data.shape[2]:
                    data_[i, :, :, :] = data[i, :, :size, :]
                else:
                    data_[i, :, :, :] = data[i, :size, :, :]
            data = data_

        for i in range(data.shape[0]):
            transformation_index = np.random.randint(4, size = 1)[0]
            if transformation_index == 0:
                data[i, :, :, :] = data[i, :, :, :]
            if transformation_index == 1:
                data[i, :, :, :] = np.rot90(data[i, :, :, :])
            if transformation_index == 2:
                data[i, :, :, :] = np.flip(data[i, :, :, :], 0)
            if transformation_index == 3:
                data[i, :, :, :] = np.flip(data[i, :, :, :], 1)
        return data
