import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
import neptune.new as neptune

from SUIM import *
from IFREMER import *
from LearningModels import *


parser = argparse.ArgumentParser(description='')

#General parameters
parser.add_argument('--phase', dest='phase', type = str,default='train', help='train or test')

#Parameters used during the training
parser.add_argument('--train_task', dest='train_task', type=str, default='Semantic_Segmentation', help='Learning Task, user can take among two alternatives Semantic_Segmentation|Image_Classification')
parser.add_argument('--learning_model', dest='learning_model', type=str, default='Unet', help='Learning model used')
parser.add_argument('--backbone_name', dest='backbone_name', type=str, default='movilenet', help='users can chosse between resnet50 and movilenet')
parser.add_argument('--class_grouping', dest = 'class_grouping', type=eval, choices=[True, False], default=True, help = 'Decide if some classes in the dataset can be mixtured')

parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='Prove', help='Checkpoints folder name')
parser.add_argument('--train_dataset_name', dest='train_dataset_name', type = str, default='SUIM', help = 'Dataset Name where the model has been trained: SUIM')
parser.add_argument('--checkpoints_main_path', dest='checkpoints_main_path', type=str, default='/d/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning/', help='Path where checkpoints have been saved')
# Evaluation parameters
parser.add_argument('--test_task', dest = 'test_task', type = str, default = 'Classification', help = 'Evaluation task, user can take among two alternatives for evaluate the trained model: Classification|Feature_representation')
parser.add_argument('--test_task_level', dest = 'test_task_level', type = str, default = 'Image_Level', help = 'Desired level of depth to perform the analysis, the user can choose between Image_Level|Pixel_Level')
parser.add_argument('--image_handling', dest = 'image_handling', type = str, default = 'Entire_Image', help = 'Parameter to manage the way test images will be processed during test, the user can choose between Entire_Image|Patches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=5, help='number images in batch')
parser.add_argument('--testcrop_size_rows', dest='testcrop_size_rows', type=int, default=256, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--testcrop_size_cols', dest='testcrop_size_cols', type=int, default=256, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--test_dataset_name', dest = 'test_dataset_name', type = str, default = 'SUIM', help = 'Dataset name where the model will be tested')

parser.add_argument('--results_name', dest = 'results_name', default='Prove', help = 'Results folder name')
parser.add_argument('--results_main_path', dest='results_main_path', type=str, default='/home/d/DATA/Pedro_Work/Jose_Work/', help='Path were results will be saved')
# Images dir and names
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/d/DATA/Pedro_Work/IFREMER_Work/DATA/Under_water_Image_Segmenetation/SUIM/TEST/TEST/', help='Main path of the dataset images')



args = parser.parse_args()

def main():
    run = neptune.init(
        project="pjsotove/UnderWater-image-Segmentation",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMjI4NTlkMS0zNzE4LTRjYTEtYWMwMi02MzQzMTY3ZWI5NzUifQ==",
    )  # your credentials

    run["sys/tags"].add([args.phase, args.test_task, args.test_task_level, args.image_handling, args.learning_model, args.backbone_name])

    if not os.path.exists(args.results_main_path + 'RESULTS/'):
        os.makedirs(args.results_main_path + 'RESULTS/')

    if not os.path.exists(args.results_main_path + 'RESULTS/' + args.results_name + '/'):
        os.makedirs(args.results_main_path + 'RESULTS/' + args.results_name + '/')

    print("Dataset pre-processing...")
    if args.test_dataset_name == 'SUIM':
        dataset = SUIM(args)
    if args.test_dataset_name == 'IFREMERData_S1':
        dataset = IFREMER(args)

    args.checkpoint_dir = args.checkpoints_main_path + args.train_dataset_name + '_checkpoints/' + args.checkpoint_name
    args.results_dir = args.results_main_path + 'RESULTS/' + args.results_name + '/'
    #Listing the runs inside the checkpoints folder
    checkpoint_folders = os.listdir(args.checkpoint_dir)
    #Loop for evry folder inside the checkpoint
    for r in range(len(checkpoint_folders)):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        model_folder = checkpoint_folders[r]
        # Creating the dir where the checkpoints are stored
        args.save_checkpoint_path = args.checkpoint_dir + '/' + model_folder + '/'
        print(args.save_checkpoint_path)
        args.save_results_dir = args.results_dir + 'Results_Trained_' + model_folder + '_Tested_' + args.test_dataset_name + '_' + dt_string + '/'
        if not os.path.exists(args.save_results_dir):
            os.makedirs(args.save_results_dir)
        #Writing the args into a file
        with open(args.save_results_dir + 'commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        print("[*] Initializing the model...")
        model = LearningModels(args, dataset, run)
        model.Test()


if __name__ == '__main__':
    main()
