import os
import sys
import argparse


parser = argparse.ArgumentParser(description='')

parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_Interactive', help='Decide wether the script will be running')
parser.add_argument('--phase', dest = 'phase', type = str, default = 'train', help = 'Decide wether the phase: Train|Test will be running')
args = parser.parse_args()


Schedule = []

if args.phase == 'train':

    BACKBONE_NAME  = ['Vgg', 'ResNetV1_18', 'ResNetV1_50', 'ResNetV2_18', 'ResNetV2_50', 'MobileNet', 'Exception']
    #CSV_FILES_NAMES = ['OTUS_Image_Classification_F1_Lithology.csv', 'OTUS_Image_Classification_F2_Lithology.csv', 'OTUS_Image_Classification_F3_Lithology.csv', 'OTUS_Image_Classification_F1_Shells_White_fragments.csv', 'OTUS_Image_Classification_F2_Shells_White_fragments.csv', 'OTUS_Image_Classification_F3_Shells_White_fragments.csv']
    CSV_FILES_NAMES = ['OTUS_Image_Classification_F1_Morphology.csv', 'OTUS_Image_Classification_F2_Morphology.csv', 'OTUS_Image_Classification_F3_Morphology.csv']
    #CSV_FILES_NAMES = ['OTUS_Image_Classification_F3_Lithology.csv']
    if args.running_in == 'Datarmor_Interactive':
        Train_MAIN_COMMAND = "TrainModel.py"
        Test_MAIN_COMMAND = "TestModel.py"
        GradCAM_MAIN_COMMAND = "TestModelGradCam.py"
    if args.running_in == 'Datarmor_PBS':
        Train_MAIN_COMMAND = "$HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TF115/TrainModel.py"
        Test_MAIN_COMMAND = "$HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TF115/TestModel.py"
        GradCAM_MAIN_COMMAND = "$HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TF115/TestModelGradCam.py"
    for backbone_name in BACKBONE_NAME:
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
        for csv_name in CSV_FILES_NAMES:
            Schedule.append("python " + Train_MAIN_COMMAND + " --train_task Image_Classification --learning_model CNN --backbone_name " + backbone_name + " --pretrained_backbone False --labels_type multiple_labels "
                            "--weights_definition automatic --learning_ratedecay True --lr 0.0001 --batch_size 5 --epochs 100 --patience 10 --runs 1 --phase train --optimizer Adam --feature_representation False --layer_index " + layer_position + " "
                            "--image_rows 1024 --image_cols 1024 --image_channels 3 --new_size_rows 224 --new_size_cols 224 --split_patch True --data_augmentation True --overlap_porcent 0.25 "
                            "--dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name " + csv_name + " --checkpoint_name " + backbone_name + "/Model_CNN_" + backbone_name + "_" + csv_name + " "
                            "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                            "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                            "--checkpoints_main_path /datawork/EXPERIMENTS/")

            Schedule.append("python " + Test_MAIN_COMMAND + " --train_task Image_Classification --learning_model CNN --backbone_name " + backbone_name + " --pretrained_backbone False --labels_type multiple_labels "
                            "--phase test --feature_representation False --layer_index " + layer_position + " "
                            "--image_rows 1024 --image_cols 1024 --image_channels 3 --new_size_rows 1024 --new_size_cols 1024 --split_patch False --overlap_porcent 0.25 "
                            "--dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name " + csv_name + " --checkpoint_name " + backbone_name + "/Model_CNN_" + backbone_name + "_" + csv_name + " "
                            "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                            "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                            "--checkpoints_main_path /datawork/EXPERIMENTS/ "
                            "--results_main_path /datawork/EXPERIMENTS/")
            Schedule.append("python " + GradCAM_MAIN_COMMAND + " --train_task Image_Classification --learning_model CNN --backbone_name " + backbone_name + " --pretrained_backbone False --labels_type multiple_labels "
                            "--phase gradcam --layer_index 1 "
                            "--image_rows 1024 --image_cols 1024 --image_channels 3 --new_size_rows 1024 --new_size_cols 1024 --split_patch False --overlap_porcent 0.25 "
                            "--dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name " + csv_name + " --checkpoint_name " + backbone_name + "/Model_CNN_" + backbone_name + "_" + csv_name + " "
                            "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                            "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                            "--checkpoints_main_path /datawork/EXPERIMENTS/ "
                            "--results_main_path /datawork/EXPERIMENTS/")


for i in range(len(Schedule)):
    os.system(Schedule[i])
