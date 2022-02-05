import os
import sys
import argparse


parser = argparse.ArgumentParser(description='')

parser.add_argument('--running_in', dest='running_in', type=str, default='Docker_Container', help='Decide wether the script will be running')
parser.add_argument('--phase', dest = 'phase', type = str, default = 'Train', help = 'Decide wether the phase: Train|Test will be running')
args = parser.parse_args()

Schedule = []

if args.phase == 'Train':
    if args.running_in == 'Local_Docker_Container':
        Schedule.append("python TrainLearningModel.py --task Semantic_Segmentation --learning_model Unet --crop_size 256 --checkpoint_name Prove "
                        "--classweight_type global --dataset_name SUIM --dataset_main_path /d/DATA/Pedro_Work/IFREMER_Work/DATA/Under_water_Image_Segmenetation/SUIM/train_val/train_val/ "
                        "--checkpoints_main_path /d/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning")
    if args.running_in == 'Local_Anaconda_Environment':
        Schedule.append("python TrainLearningModel.py --task Semantic_Segmentation --learning_model Unet --crop_size 256 --checkpoint_name Prove "
                        "--classweight_type global --dataset_name SUIM --dataset_main_path D:/DATA/Pedro_Work/IFREMER_Work/DATA/Under_water_Image_Segmenetation/SUIM/train_val/train_val/ "
                        "--checkpoints_main_path D:/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning")

    if args.running_in == 'Datarmor_Interactive':
        Schedule.append("python TrainLearningModel.py --task Semantic_Segmentation --learning_model Unet "
                        "--backbone_name None --classweight_type global --lr 0.0001 --optimizer Adam --batch_size 2 --runs 1 --crop_size_rows 320 --crop_size_cols 240 "
                        "--epochs 400 --phase train "
                        "--dataset_name SUIM --class_grouping True --checkpoint_name SUIM_SS_Train "
                        "--dataset_main_path /datawork/DATA/SUIM/train_val/train_val/ "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/CHECKPOINTS/")

    if args.running_in == 'Datarmor_PBS':
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --task Semantic_Segmentation --learning_model Unet "
                        "--backbone_name None --classweight_type global --lr 0.0001 --optimizer Adam --batch_size 2 --runs 1 --crop_size_rows 320 --crop_size_cols 240 "
                        "--epochs 400 --phase train "
                        "--dataset_name SUIM --class_grouping True --checkpoint_name SUIM_SS_Train "
                        "--dataset_main_path /datawork/DATA/SUIM/train_val/train_val/ "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/CHECKPOINTS/")
if args.phase == 'Test':
    if args.running_in == 'Local_Anaconda_Environment':
        Schedule.append("python TestLearningModel.py --phase test "
                        "--train_task Semantic_Segmentation --learning_model Unet --backbone_name None --class_grouping True --train_dataset_name SUIM --checkpoint_name SUIM_SS_Train "
                        "--checkpoints_main_path D:/DATA/Pedro_Work/IFREMER_Work/CODE/IFREMER-ABYSSES-checkpoints/ "
                        "--test_task Feature_representation --test_task_level Pixel_Level --image_handling Entire_Image --batch_size 1 --testcrop_size_rows 128 --testcrop_size_cols 128 "
                        "--test_dataset_name SUIM --results_name SUIM_SS_Train "
                        "--results_main_path D:/DATA/Pedro_Work/IFREMER_Work/CODE/IFREMER-ABYSSES-results/ "
                        "--dataset_main_path D:/DATA/Pedro_Work/IFREMER_Work/DATA/Under_water_Image_Segmenetation/SUIM/TEST/TEST/"
                        )
    if args.running_in == 'Datarmor_Interactive':
        Schedule.append("python TestLearningModel.py --phase test "
                        "--train_task Semantic_Segmentation --learning_model Unet --backbone_name None --class_grouping True --train_dataset_name SUIM --checkpoint_name SUIM_SS_Train "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/CHECKPOINTS/ "
                        "--test_task Feature_representation --test_task_level Pixel_Level --image_handling Entire_Image --batch_size 1 --testcrop_size_rows 128 --testcrop_size_cols 128 "
                        "--test_dataset_name SUIM --results_name SUIM_SS_Train "
                        "--results_main_path /datawork/EXPERIMENTS/ "
                        "--dataset_main_path /datawork/DATA/SUIM/test_subset/"
                        )
    if args.running_in == 'Datarmor_PBS':
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TestLearningModel.py --phase test "
                        "--train_task Semantic_Segmentation --learning_model Unet --backbone_name None --class_grouping True --train_dataset_name SUIM --checkpoint_name SUIM_SS_Train "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/CHECKPOINTS/ "
                        "--test_task Feature_representation --test_task_level Pixel_Level --image_handling Entire_Image --batch_size 1 --testcrop_size_rows 128 --testcrop_size_cols 128 "
                        "--test_dataset_name SUIM --results_name SUIM_SS_Train "
                        "--results_main_path /datawork/EXPERIMENTS/ "
                        "--dataset_main_path /datawork/DATA/SUIM/test_subset/"
                        )




for i in range(len(Schedule)):
    os.system(Schedule[i])
