import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class SUIM():
    def __init__(self, args):

        self.args = args
        self.classes = 8
        self.Train_Paths = []
        self.Label_Paths = []
        if self.args.phase == 'train':

            self.images_main_path = self.args.dataset_main_path + 'images/'
            self.labels_main_path = self.args.dataset_main_path + 'masks/'
            #Listing the images
            images_names = os.listdir(self.images_main_path)
            for image_name in images_names:
                if image_name[-4:] in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_path = self.images_main_path + image_name
                    label_path = self.labels_main_path + image_name[:-4] + '.bmp'

                    #reading images and labels
                    image = mpimg.imread(image_path)
                    label = mpimg.imread(label_path)
                    if image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1]:

                        self.Train_Paths.append(image_path)
                        self.Label_Paths.append(label_path)


                    #    data = np.zeros((image.shape[0],image.shape[1],image.shape[2] + 1))

                        #Converting rgb labels to a int map
                    #    label = self.Label_Converter(label)
                    #    data[:,:,:3] = image.astype('float32')/255
                    #    data[:,:, 3] = label

                    #    self.data_list.append(data)

            print("Splitting the data into Training and Validation sets")
            num_samples = len(self.Train_Paths)
            num_samples_val = int((num_samples * 10)/100)
            # Applying a shuffle aiming at avoid the network uses always the same samples
            index = np.arange(num_samples)
            np.random.shuffle(index)
            self.Train_Paths = np.asarray(self.Train_Paths)[index]
            self.Label_Paths = np.asarray(self.Label_Paths)[index]
            self.Valid_Paths = self.Train_Paths[:num_samples_val]
            self.Valid_Label_Paths = self.Label_Paths[:num_samples_val]
            self.Train_Paths = self.Train_Paths[num_samples_val:]
            self.Train_Label_Paths = self.Label_Paths[num_samples_val:]
