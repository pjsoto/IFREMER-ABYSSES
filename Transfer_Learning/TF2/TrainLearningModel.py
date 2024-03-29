import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
#import neptune.new as neptune

from SUIM import *
from LearningModels import *
from IFREMER import *
parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_task', dest='train_task', type=str, default='Semantic_Segmentation', help='Learning Task, user can take among two alternatives Semantic_Segmentation|Image_Classification')
parser.add_argument('--learning_model', dest='learning_model', type=str, default='Unet', help='Learning model used')
parser.add_argument('--backbone_name', dest='backbone_name', type=str, default='movilenet', help='users can chosse between resnet50 and movilenet')
parser.add_argument('--labels_type', dest='labels_type', type=str, default='onehot', help='users can choose between onehot_labels(Image Classification) or multiple_labels(Multilabel Image Classification)')
parser.add_argument('--classweight_type', dest='classweight_type', type=str, default = 'global', help='users can chosse between global|batch fro the way the weights will be computed')
parser.add_argument('--learning_ratedecay', dest = 'learning_ratedecay', type=eval, choices=[True, False], default=True, help = 'Decide if some classes in the dataset can be mixtured')
parser.add_argument('--lr', dest='lr', type = float, default = 0.01, help = 'The learning rate parameter that will be used in the optimezer')
parser.add_argument('--gamma', dest = 'gamma', type = float, default = 2.0, help = 'Parameter used in the focal loss function')
parser.add_argument('--alpha', dest = 'alpha', type = float, default = 4.0, help = 'Parameter used in the focal loss function')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=5, help='number images in batch')
parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of executions of the algorithm')
parser.add_argument('--crop_size_rows', dest='crop_size_rows', type=int, default=256, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--crop_size_cols', dest='crop_size_cols', type=int, default=256, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--epochs', dest = 'epochs', type = int, default = 100, help='Number of epochs')
# Phase
parser.add_argument('--phase', dest='phase', type = str,default='train', help='train, test, generate_image, create_dataset')
parser.add_argument('--optimizer', dest = 'optimizer', type = str, default = 'Adam', help = 'The optimizer that will update the gradients computed by backprop')
parser.add_argument('--loss', dest = 'loss', type = str, default = 'weighted_crossentropy', help = 'Definition of the loss function. Users can choose among weighted_binary_crossentropy|weighted_categorical_crossentropy|focal_loss')
parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='Prove', help='Checkpoints folder name')
# Images dir and names
parser.add_argument('--dataset_name', dest='dataset_name', type = str, default='SUIM', help = 'Dataset Name: SUIM')
parser.add_argument('--class_grouping', dest = 'class_grouping', type=eval, choices=[True, False], default=True, help = 'Decide if some classes in the dataset can be mixtured')
parser.add_argument('--dataset_csv_main_path', dest='dataset_csv_main_path', type=str, default=' ', help='Dataset CSV main path')
parser.add_argument('--csvfile_name', dest = 'csvfile_name', type = str, default = ' ', help = 'CSV file name')
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/d/DATA/Pedro_Work/IFREMER_Work/DATA/Under_water_Image_Segmenetation/SUIM/train_val/train_val/', help='Main path of the dataset images')
parser.add_argument('--checkpoints_main_path', dest='checkpoints_main_path', type=str, default='/d/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning/', help='Path where checkpoints will be saved' )

args = parser.parse_args()

def main():

#    run = neptune.init(
#        project="pjsotove/UnderWater-image-Segmentation",
#        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMjI4NTlkMS0zNzE4LTRjYTEtYWMwMi02MzQzMTY3ZWI5NzUifQ==",
#    )  # your credentials

#    run["sys/tags"].add([args.train_task, args.learning_model, args.backbone_name, args.classweight_type])

    args.checkpoints_main_path = args.checkpoints_main_path + '/CHECKPOINTS/'
    if not os.path.exists(args.checkpoints_main_path):
        os.makedirs(args.checkpoints_main_path)

    args.checkpoint_dir = args.checkpoints_main_path + args.dataset_name + '_checkpoints/' + args.checkpoint_name

    print("Dataset pre-processing...")
    if args.dataset_name == 'SUIM':
        dataset = SUIM(args)
    if args.dataset_name == 'IFREMERData':
        dataset = IFREMER(args)
    if args.dataset_name == 'OTUSIFREMER_IMAGELABEL':
        dataset = OTUSIFREMER_IMAGELABEL(args)
    #Running several times
    for r in range(args.runs):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        # Creating the dir for saving the model
        args.save_checkpoint_path = args.checkpoint_dir + '/' + args.learning_model + '_' + dt_string + '/'

        if not os.path.exists(args.save_checkpoint_path):
            os.makedirs(args.save_checkpoint_path)
        #Writing the args into a file
        with open(args.save_checkpoint_path + 'commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        print("[*] Initializing the model...")
        model = LearningModels(args, dataset, run)
        model.Train()


if __name__ == '__main__':
    main()
