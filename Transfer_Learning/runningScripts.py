import os
import sys
import argparse


parser = argparse.ArgumentParser(description='')

parser.add_argument('--running_in', dest='running_in', type=str, default='Docker_Container', help='Decide wether the script will be running')
args = parser.parse_args()

Schedule = []

if args.running_in == 'Docker_Container':
    Schedule.append("python TrainLearningModel.py --task Semantic_Segmentation --learning_model Unet --crop_size 256 --checkpoint_name Prove "
                    "--dataset_name SUIM --dataset_main_path /d/DATA/Pedro_Work/IFREMER_Work/DATA/Under_water_Image_Segmenetation/SUIM/train_val/train_val/ "
                    "--checkpoints_main_path /d/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning")
if args.running_in == 'Anaconda_Environment':
    Schedule.append("python TrainLearningModel.py --task Semantic_Segmentation --learning_model Unet --crop_size 256 --checkpoint_name Prove "
                    "--dataset_name SUIM --dataset_main_path D:/DATA/Pedro_Work/IFREMER_Work/DATA/Under_water_Image_Segmenetation/SUIM/train_val/train_val/ "
                    "--checkpoints_main_path D:/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning")



for i in range(len(Schedule)):
    os.system(Schedule[i])
