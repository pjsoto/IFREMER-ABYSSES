import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
#import neptune.new as neptune

from IFREMER import *
from Model import *
parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_task', dest='train_task', type=str, default='Image_Classification', help='Learning Task, user can take among two alternatives Semantic_Segmentation|Image_Classification')
parser.add_argument('--learning_model', dest='learning_model', type=str, default='CNN', help='Learning model used')
parser.add_argument('--backbone_name', dest='backbone_name', type=str, default='Xception', help='users can chosse between resnet50 and movilenet')
parser.add_argument('--pretrained_backbone', dest = 'pretrained_backbone', type=eval, choices=[True, False], default=False, help = 'Decide if the bockbone will be a pretrained one or will be trained from scratch')
parser.add_argument('--labels_type', dest='labels_type', type=str, default='multiple_labels', help='users can choose between onehot_labels(Image Classification) or multiple_labels(Multilabel Image Classification)')

parser.add_argument('--weights_definition', dest='weights_definition', type=str, default = 'automatic', help='users can chosse between automatic|manual for the way the weights will be obtained')
parser.add_argument('--learning_ratedecay', dest = 'learning_ratedecay', type=eval, choices=[True, False], default=True, help = 'Decide if learning rate decay can be used')
parser.add_argument('--lr', dest='lr', type = float, default = 0.01, help = 'The learning rate parameter that will be used in the optimezer')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=5, help='number images in batch')
parser.add_argument('--epochs', dest = 'epochs', type = int, default = 100, help='Number of epochs')
parser.add_argument('--patience', dest='patience', type=int, default=10, help='number of epochs without improvement to apply early stop')
parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of executions of the algorithm')

parser.add_argument('--phase', dest='phase', type = str,default='train', help='train, test, generate_image, create_dataset')
parser.add_argument('--tracking_training', dest = 'tracking_training', type = eval, choices = [True, False], default = True, help = 'Set this parameter to True if the training will be tracked')
parser.add_argument('--continue_training', dest = 'continue_training', type = eval, choices = [True, False], default = True, help = 'Set this parameter to True if the training must continue from a previously saved model')
parser.add_argument('--identifier', dest = 'identifier', type = str, default = 'Lithology_MS', help = 'Used for diferentiating the tracking files of each category')
parser.add_argument('--optimizer', dest = 'optimizer', type = str, default = 'MomentumOptimizer', help = 'The optimizer that will update the gradients computed by backprop')
parser.add_argument('--feature_representation', dest = 'feature_representation', type=eval, choices=[True, False], default=False, help = 'This paraemeter is used to decide if a feature representation will be accomplished')
parser.add_argument('--layer_index', dest = 'layer_index', type = int, default = 17, help = 'Definition of the layer where the feature will be taken')

parser.add_argument('--image_rows', dest='image_rows', type=int, default=1024, help='Image dimensions in rows')
parser.add_argument('--image_cols', dest='image_cols', type=int, default=1024, help='Image dimensions in columns')
parser.add_argument('--image_channels', dest='image_channels', type=int, default=3, help='Number of channels of images')
parser.add_argument('--new_size_rows', dest='new_size_rows', type=int, default=224, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--new_size_cols', dest='new_size_cols', type=int, default=224, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--split_patch',dest='split_patch' ,type=eval, choices=[True, False], default=True, help = 'Decide if the images can be splitted during training')
parser.add_argument('--data_augmentation',dest='data_augmentation' ,type=eval, choices=[True, False], default=True, help = 'Decide if data augmentation will be applied to the images during training')
parser.add_argument('--overlap_porcent', dest = 'overlap_porcent', type = float, default = 0, help = 'Specifies the overlap between the pateches extracted from the images')
# Images dir and names
parser.add_argument('--dataset_name', dest='dataset_name', type = str, default='OTUSIFREMER_IMAGELABEL', help = 'Dataset Name: SUIM')
parser.add_argument('--dataset_csv_main_path', dest='dataset_csv_main_path', type=str, default='/e/DATARMOR/DATA/IFREMER_OTUS/BIGLI_CSVs/', help='Dataset CSV main path')
parser.add_argument('--csvfile_name', dest = 'csvfile_name', type = str, default = 'OTUS_Image_Classification_F1_Morphology.csv', help = 'CSV file name')
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/e/OTUS/2018/Donneesbrutes/Biigle_Montsegur(MS)1024/', help='Main path of the dataset images')
parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='Prove', help='Checkpoints folder name')
parser.add_argument('--checkpoints_main_path', dest='checkpoints_main_path', type=str, default='/d/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning/', help='Path where checkpoints will be saved' )

args = parser.parse_args()

def main(args):

    args.r = 0
    continue_ = False
    args.checkpoints_main_path = args.checkpoints_main_path + '/CHECKPOINTS/'
    if not os.path.exists(args.checkpoints_main_path):
        os.makedirs(args.checkpoints_main_path)

    args.checkpoint_dir = args.checkpoints_main_path + args.dataset_name + '_checkpoints/' + args.checkpoint_name
    args.tracking_files = args.checkpoints_main_path + args.dataset_name + '_checkpoints/'

    print("Dataset pre-processing...")
    if args.dataset_name == 'SUIM':
        dataset = SUIM(args)
    if args.dataset_name == 'IFREMERData':
        dataset = IFREMER(args)
    if args.dataset_name == 'OTUSIFREMER_IMAGELABEL':
        dataset = OTUSIFREMER_IMAGELABEL(args)
    if args.continue_training:
        ##Loading the hyperparameters for the last trained model
        continue_, args = Recover_hyperparameters_MS(args)
    #Running several times
    while args.r < args.runs:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # Creating the dir for saving the model
        if continue_:
            continue_ = False
            args.save_checkpoint_path = args.save_checkpoint_path + '/'
        else:
            args.continue_training = False
            args.initial_epoch = 0
            args.best_val_fs = 0
            args.save_checkpoint_path = args.checkpoint_dir + '/' + args.learning_model + '_' + dt_string + '/'

        if not os.path.exists(args.save_checkpoint_path):
            os.makedirs(args.save_checkpoint_path)
        #Writing the args into a file
        with open(args.save_checkpoint_path + 'commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        print("[*] Initializing the model...")
        model = Model(args, dataset)
        model.Train()
        args.r += 1


if __name__ == '__main__':
    main(args)
