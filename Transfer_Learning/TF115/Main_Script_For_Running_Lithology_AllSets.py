import os
import sys
import argparse

from Tools import *
parser = argparse.ArgumentParser(description='')

parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_Interactive', help='Decide wether the script will be running')
parser.add_argument('--phase', dest = 'phase', type = str, default = 'train', help = 'Decide wether the phase: Train|Test will be running')
parser.add_argument('--tracking_training', dest = 'tracking_training', type = eval, choices = [True, False], default = True, help = 'Set this parameter to True if the training will be tracked')
parser.add_argument('--continue_training', dest = 'continue_training', type = eval, choices = [True, False], default = True, help = 'Set this parameter to True if the training musy continue from a previously saved model')
parser.add_argument('--tracking_files_path', dest = 'tracking_files_path', type = str, default = '/datawork/EXPERIMENTS/CHECKPOINTS/OTUSIFREMER_IMAGELABEL_checkpoints/')
args = parser.parse_args()


Schedule = []

if args.phase == 'train':
    b = 0
    c = 0
    continue_ = False
    tracking_list = []
    BACKBONE_NAME  = ['Vgg', 'ResNetV1_18', 'ResNetV1_50', 'ResNetV2_18', 'ResNetV2_50', 'Xception']
    CSV_FILES_NAMES_TRAIN = ['OTUS_Image_Classification_F1_Lithology_AllSets.csv', 'OTUS_Image_Classification_F2_Lithology_AllSets.csv','OTUS_Image_Classification_F3_Lithology_AllSets.csv']
    CSV_FILES_NAMES_TEST  = ['OTUS_Image_Classification_F1_Lithology_AllSets.csv', 'OTUS_Image_Classification_F2_Lithology_AllSets.csv','OTUS_Image_Classification_F3_Lithology_AllSets.csv']
    DATASET_MAIN_PATH_TRAIN = ['/datawork/DATA/','/datawork/DATA/','/datawork/DATA/']
    DATASET_MAIN_PATH_TEST  = ['/datawork/DATA/','/datawork/DATA/','/datawork/DATA/']

    if args.running_in == 'Datarmor_Interactive':
        Train_MAIN_COMMAND = "TrainModel.py"
        Test_MAIN_COMMAND = "TestModel.py"
        GradCAM_MAIN_COMMAND = "TestModelGradCam.py"
    if args.running_in == 'Datarmor_PBS':
        Train_MAIN_COMMAND = "$HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TF115/TrainModel.py"
        Test_MAIN_COMMAND = "$HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TF115/TestModel.py"
        GradCAM_MAIN_COMMAND = "$HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TF115/TestModelGradCam.py"
    if args.continue_training:
        continue_, b, c = Recover_hyperparameters_GM(args.tracking_files_path + "General_tracking_LT_AllSets.txt", b, c)
    while b < len(BACKBONE_NAME):
        if not continue_:
            c = 0
            continue_training = False
        backbone_name = BACKBONE_NAME[b]
        if 'MobileNet' in backbone_name:
            layer_position = '17'
        if  'Vgg' in backbone_name:
            layer_position = '19'
        if 'Xception' in backbone_name:
            layer_position = '55'
        if 'ResNetV1_18' in backbone_name or 'ResNetV2_18' in backbone_name:
            layer_position = '15'
        if 'ResNetV1_50' in backbone_name or 'ResNetV2_50' in backbone_name:
            layer_position = '23'

        while c < len(CSV_FILES_NAMES_TRAIN):
            csv_name_train = CSV_FILES_NAMES_TRAIN[c]
            csv_name_test = CSV_FILES_NAMES_TEST[c]
            Dataset_main_path_train = DATASET_MAIN_PATH_TRAIN[c]
            Dataset_main_path_test = DATASET_MAIN_PATH_TEST[c]
            print(Dataset_main_path_train)
            if args.tracking_training:
                tracking_list.append(str(b) + "/" + str(c) + "/Train")

            if continue_:
                continue_training = True
                continue_ = False

            Schedule.append("python " + Train_MAIN_COMMAND + " --train_task Image_Classification --learning_model CNN --backbone_name " + backbone_name + " --pretrained_backbone False --labels_type onehot_labels "
                            "--weights_definition automatic --learning_ratedecay True --lr 0.0001 --batch_size 5 --epochs 100 --patience 10 --runs 1 --phase train --tracking_training " + str(args.tracking_training) + " --continue_training " + str(continue_training) + " --identifier Lithology_AllSets --optimizer Adam --feature_representation True --layer_index " + layer_position + " "
                            "--image_rows 1024 --image_cols 1024 --image_channels 3 --new_size_rows 224 --new_size_cols 224 --split_patch True --data_augmentation True --overlap_porcent 0.25 "
                            "--dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name " + csv_name_train + " --checkpoint_name " + backbone_name + "/Model_CNN_" + backbone_name + "_" + csv_name_train + " "
                            "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                            "--dataset_main_path " + Dataset_main_path_train + " "
                            "--checkpoints_main_path /datawork/EXPERIMENTS/")
            if args.tracking_training:
                tracking_list.append(str(b) + "/" + str(c) + "/Test")

            Schedule.append("python " + Test_MAIN_COMMAND + " --train_task Image_Classification --learning_model CNN --backbone_name " + backbone_name + " --pretrained_backbone False --labels_type onehot_labels "
                            "--phase test --feature_representation False --layer_index " + layer_position + " "
                            "--image_rows 1024 --image_cols 1024 --image_channels 3 --new_size_rows 1024 --new_size_cols 1024 --split_patch False --overlap_porcent 0.25 "
                            "--dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name " + csv_name_test + " --checkpoint_name " + backbone_name + "/Model_CNN_" + backbone_name + "_" + csv_name_train + " "
                            "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                            "--dataset_main_path " + Dataset_main_path_test + " "
                            "--checkpoints_main_path /datawork/EXPERIMENTS/ "
                            "--results_main_path /datawork/EXPERIMENTS/")

            #if args.tracking_training:
            #    tracking_list.append(str(b) + "/" + str(c) + "/GradCam")
            #Schedule.append("python " + GradCAM_MAIN_COMMAND + " --train_task Image_Classification --learning_model CNN --backbone_name " + backbone_name + " --pretrained_backbone False --labels_type onehot_labels "
            #                "--phase gradcam --layer_index 1 "
            #                "--image_rows 1024 --image_cols 1024 --image_channels 3 --new_size_rows 1024 --new_size_cols 1024 --split_patch False --overlap_porcent 0.25 "
            #                "--dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name " + csv_name_test + " --checkpoint_name " + backbone_name + "/Model_CNN_" + backbone_name + "_" + csv_name_train + " "
            #                "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
            #                "--dataset_main_path " + Dataset_main_path_test + " "
            #                "--checkpoints_main_path /datawork/EXPERIMENTS/ "
            #                "--results_main_path /datawork/EXPERIMENTS/")
            c += 1
            continue_training = False
        b += 1

for i in range(len(Schedule)):
    if args.tracking_training:
        t = open(args.tracking_files_path + "General_tracking_LT_AllSets.txt", "a")
        t.write(tracking_list[i] + "\n")
        t.close()
    os.system(Schedule[i])

if args.tracking_training:
    t = open(args.tracking_files_path + "General_tracking_LT_AllSets.txt", "a")
    t.write("Completed\n")
    t.close()
