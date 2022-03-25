import os
import sys
import argparse


parser = argparse.ArgumentParser(description='')

parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_Interactive', help='Decide wether the script will be running')
parser.add_argument('--phase', dest = 'phase', type = str, default = 'train', help = 'Decide wether the phase: Train|Test will be running')
args = parser.parse_args()


Schedule = []

if args.phase == 'train':

    BACKBONE_NAME  = ['Vgg']
    CSV_FILES_NAMES = ['OTUS_Image_Classification_F1_Shells_White_fragments.csv']

    if args.running_in == 'Datarmor_Interactive':
        MAIN_COMMAND = "TrainModel.py"
    if args.running_in == 'Datarmor_PBS':
        MAIN_COMMAND = "$HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TF115/TrainModel.py"
    for backbone_name in BACKBONE_NAME:
        for csv_name in CSV_FILES_NAMES:
            Schedule.append("python " + MAIN_COMMAND + " --train_task Image_Classification --learning_model CNN --backbone_name " + backbone_name + " --pretrained_backbone False --labels_type onehot_labels "
                            "--weights_definition manual --learning_ratedecay True --lr 0.01 --batch_size 5 --epochs 100 --runs 1 --phase train --optimizer MomentumOptimizer "
                            "--image_rows 4000 --image_cols 6000 --image_channels 3 --new_size_rows 224 --new_size_cols 224 --split_patch True --data_augmentation True --overlap_porcent 0.25 "
                            "--dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name " + csv_name + " --checkpoint_name " + backbone_name + "/Model_CNN_" + backbone_name + "_" + csv_name + " "
                            "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                            "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur/ "
                            "--checkpoints_main_path /datawork/EXPERIMENTS/")
