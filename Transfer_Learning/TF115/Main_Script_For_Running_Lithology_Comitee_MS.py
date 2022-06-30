import os
import sys
import argparse

from Tools import *
parser = argparse.ArgumentParser(description='')

parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_Interactive', help='Decide wether the script will be running')
args = parser.parse_args()


Schedule = []


c = 0

tracking_list = []
BACKBONE_NAMES  = ['Vgg', 'ResNetV1_18', 'ResNetV1_50', 'ResNetV2_18', 'ResNetV2_50', 'Xception']
CSV_FILES_NAMES_TRAIN = ['OTUS_Image_Classification_F2_Lithology_MS_Ltd.csv','OTUS_Image_Classification_F3_Lithology_MS_Ltd.csv']
CSV_FILES_NAMES_TEST  = ['OTUS_Image_Classification_F2_Lithology_MS_Ltd.csv','OTUS_Image_Classification_F3_Lithology_MS_Ltd.csv']
DATASET_MAIN_PATH_TRAIN = ['/datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/','/datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/']
DATASET_MAIN_PATH_TEST  = ['/datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/','/datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/']

if args.running_in == 'Datarmor_Interactive':
    TestComitee_MAIN_COMMAND = "TestModelsComitee.py"
if args.running_in == 'Datarmor_PBS':
    TestComitee_MAIN_COMMAND = "$HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TF115/TestModelsComitee.py"

while c < len(CSV_FILES_NAMES_TRAIN):
    csv_name_train = CSV_FILES_NAMES_TRAIN[c]
    csv_name_test = CSV_FILES_NAMES_TEST[c]
    Dataset_main_path_train = DATASET_MAIN_PATH_TRAIN[c]
    Dataset_main_path_test = DATASET_MAIN_PATH_TEST[c]
    print(Dataset_main_path_test)

    Schedule.append("python " + TestComitee_MAIN_COMMAND + " --train_task Image_Classification --learning_model CNN --backbone_names " + str(BACKBONE_NAMES) + " --pretrained_backbone False --labels_type onehot_labels "
                    "--phase test "
                    "--image_rows 1024 --image_cols 1024 --image_channels 3 --new_size_rows 1024 --new_size_cols 1024 --split_patch False --overlap_porcent 0.25 "
                    "--dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name_train " + csv_name_train + " --csvfile_name_test " + csv_name_test + " "
                    "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                    "--dataset_main_path " + Dataset_main_path_test + " "
                    "--checkpoints_main_path /datawork/EXPERIMENTS "
                    "--results_main_path /datawork/EXPERIMENTS")

    c += 1


for i in range(len(Schedule)):
    os.system(Schedule[i])
