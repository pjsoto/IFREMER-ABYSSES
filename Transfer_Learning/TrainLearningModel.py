import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
import neptune.new as neptune

from SUIM import *
from LearningModels import *

parser = argparse.ArgumentParser(description='')

parser.add_argument('--task', dest='task', type=str, default='Semantic_Segmentation', help='Learning Task, user can take among two alternatives Semantic_Segmentation|Image_Classification')
parser.add_argument('--learning_model', dest='learning_model', type=str, default='Unet', help='Learning model used')
parser.add_argument('--backbone_name', dest='backbone_name', type=str, default='movilenet', help='users can chosse between resnet50 and movilenet')
parser.add_argument('--classweight_type', dest='classweight_type', type=str, default = 'global', help='users can chosse between global|batch fro the way the weights will be computed')
parser.add_argument('--lr', dest='lr', type = float, default = 0.01, help = 'The learning rate parameter that will be used in the optimezer')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=5, help='number images in batch')
parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of executions of the algorithm')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--epochs', dest = 'epochs', type = int, default = 100, help='Number of epochs')
# Phase
parser.add_argument('--phase', dest='phase', type = str,default='train', help='train, test, generate_image, create_dataset')
parser.add_argument('--optimizer', dest = 'optimizer', type = str, default = 'Adam', help = 'The optimizer that will update the gradients computed by backprop')
parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='Prove', help='Checkpoints folder name')
# Images dir and names
parser.add_argument('--dataset_name', dest='dataset_name', type = str, default='SUIM', help = 'Dataset Name: SUIM')
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/d/DATA/Pedro_Work/IFREMER_Work/DATA/Under_water_Image_Segmenetation/SUIM/train_val/train_val/', help='Main path of the dataset images')
parser.add_argument('--checkpoints_main_path', dest='checkpoints_main_path', type=str, default='/d/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning/', help='Path where checkpoints will be saved' )

args = parser.parse_args()

def main():

    run = neptune.init(
        project="pjsotove/UnderWater-image-Segmentation",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMjI4NTlkMS0zNzE4LTRjYTEtYWMwMi02MzQzMTY3ZWI5NzUifQ==",
    )  # your credentials

    run["sys/tags"].add([args.task, args.learning_model, args.backbone_name])

    print(args)
    if not os.path.exists(args.checkpoints_main_path + '/checkpoints/'):
        os.makedirs(args.checkpoints_main_path + '/checkpoints/')

    args.checkpoint_dir = args.checkpoints_main_path + args.dataset_name + '_checkpoints/' + args.checkpoint_name

    print("Dataset pre-processing...")
    if args.dataset_name == 'SUIM':
        dataset = SUIM(args)

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
