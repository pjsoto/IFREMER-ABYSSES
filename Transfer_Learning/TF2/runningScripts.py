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
        Schedule.append("python TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name vgg16 "
                        "--crop_size_rows 224 --crop_size_cols 512 --labels_type multiple_labels --learning_ratedecay True --lr 0.001 --gamma 2.0 --alpha 4.0 --optimizer SGD --batch_size 2 --runs 1 "
                        "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name Prove "
                        "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F1.csv "
                        "--dataset_csv_main_path D:/DATA/Pedro_Work/IFREMER_Work/DATA/IFREMER_OTUS/BIGLI_CSVs/ "
                        "--dataset_main_path E:/OTUS/2018/Donneesbrutes/Montsegur(MS)/acq_20180825T010450/ "
                        "--checkpoints_main_path D:/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning/")

    if args.running_in == 'Local_Anaconda_Environment':
        Schedule.append("python TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name vgg16 "
                        "--crop_size_rows 224 --crop_size_cols 512 --labels_type multiple_labels --learning_ratedecay True --lr 0.001 --gamma 2.0 --alpha 4.0 --optimizer SGD --batch_size 2 --runs 1 "
                        "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name Prove "
                        "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F1.csv "
                        "--dataset_csv_main_path D:/DATA/Pedro_Work/IFREMER_Work/DATA/IFREMER_OTUS/BIGLI_CSVs/ "
                        "--dataset_main_path E:/OTUS/2018/Donneesbrutes/Biigle_Montsegur(MS)/ "
                        "--checkpoints_main_path D:/DATA/Pedro_Work/IFREMER_Work/CODE/Transfer_Learning/")

    if args.running_in == 'Datarmor_Interactive':
        Schedule.append("python TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name movilenet "
                        "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                        "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name VGG16_F1 "
                        "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F1_H1.csv "
                        "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                        "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/")

    if args.running_in == 'Datarmor_PBS':
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name vgg16 "
                        "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                        "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name VGGH1/VGG16_F1 "
                        "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F1_H1.csv "
                        "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                        "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/")
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name vgg16 "
                        "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                        "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name VGGH1/VGG16_F2 "
                        "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F2_H1.csv "
                        "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                        "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/")
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name vgg16 "
                        "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                        "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name VGGH1/VGG16_F3 "
                        "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F3_H1.csv "
                        "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                        "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/")

        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name movilenet "
                         "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                         "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name MovileNetH1/MONT_F1 "
                         "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F1_H1.csv "
                         "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                         "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                         "--checkpoints_main_path /datawork/EXPERIMENTS/")
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name movilenet "
                         "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                         "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name MovileNetH1/MONT_F2 "
                         "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F2_H1.csv "
                         "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                         "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                         "--checkpoints_main_path /datawork/EXPERIMENTS/")
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name movilenet "
                         "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                         "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name MovileNetH1/MONT_F3 "
                         "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F3_H1.csv "
                         "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                         "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                         "--checkpoints_main_path /datawork/EXPERIMENTS/")

        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name resnet50 "
                        "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                        "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name ResNetH1/ResNet50_F1 "
                        "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F1_H1.csv "
                        "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                        "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/")
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name resnet50 "
                        "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                        "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name ResNetH1/ResNet50_F2 "
                        "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F2_H1.csv "
                        "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                        "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/")
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TrainLearningModel.py --train_task Image_Classification --learning_model CNN --backbone_name resnet50 "
                        "--crop_size_rows 1024 --crop_size_cols 1024 --labels_type multiple_labels --learning_ratedecay True --lr 0.0001 --gamma 2.0 --alpha 4.0 --optimizer Adam --batch_size 2 --runs 1 "
                        "--epochs 400 --phase train --loss weighted_binary_crossentropy --checkpoint_name ResNetH1/ResNet50_F3 "
                        "--dataset_name OTUSIFREMER_IMAGELABEL --class_grouping False --classweight_type global --csvfile_name OTUS_Image_Classification_F3_H1.csv "
                        "--dataset_csv_main_path /datawork/DATA/CSVs/OTUS_2018/ "
                        "--dataset_main_path /datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/ "
                        "--checkpoints_main_path /datawork/EXPERIMENTS/")
if args.phase == 'Test':
    if args.running_in == 'Local_Anaconda_Environment':
        Schedule.append("python TestLearningModel.py --phase test "
                        "--train_task Image_Classification --learning_model CNN --backbone_name vgg16 --train_dataset_name OTUSIFREMER_IMAGELABEL --checkpoint_name VGGH1/VGG16_F3 "
                        "--checkpoints_main_path E:/DATARMOR/EXPERIMENTS/CHECKPOINTS/ "
                        "--test_task Classification --test_task_level Image_Level --image_handling Entire_Image --labels_type multiple_labels --batch_size 1 --testcrop_size_rows 1024 --testcrop_size_cols 1024 "
                        "--test_dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name OTUS_Image_Classification_F3_H1.csv --class_grouping False "
                        "--results_main_path D:/DATA/Pedro_Work/IFREMER_Work/CODE/IFREMER-ABYSSES-results/ --results_name VGG16_F3 "
                        "--dataset_csv_main_path E:/DATARMOR/DATA/IFREMER_OTUS/BIGLI_CSVs/ "
                        "--dataset_main_path E:/OTUS/2018/Donneesbrutes/Biigle_Montsegur(MS)/"
                        )
    if args.running_in == 'Local_Docker_Container':
        Schedule.append("python TestLearningModel.py --phase test "
                        "--train_task Image_Classification --learning_model CNN --backbone_name vgg16 --train_dataset_name OTUSIFREMER_IMAGELABEL --checkpoint_name MovileNet/VGG16_F2 "
                        "--checkpoints_main_path E:/DATARMOR/EXPERIMENTS/CHECKPOINTS/ "
                        "--test_task Classification --test_task_level Image_Level --image_handling Entire_Image --labels_type multiple_labels --batch_size 1 --testcrop_size_rows 1024 --testcrop_size_cols 1024 "
                        "--test_dataset_name OTUSIFREMER_IMAGELABEL --csvfile_name OTUS_Image_Classification_F2.csv --class_grouping False "
                        "--results_main_path /d/DATA/Pedro_Work/IFREMER_Work/CODE/IFREMER-ABYSSES-results/ --results_name VGG16_F2 "
                        "--dataset_csv_main_path E:/DATARMOR/DATA/IFREMER_OTUS/BIGLI_CSVs/ "
                        "--dataset_main_path E:/OTUS/2018/Donneesbrutes/Biigle_Montsegur(MS)/"
                        )
    if args.running_in == 'Datarmor_Interactive':
        Schedule.append("python TestLearningModel.py --phase test "
                        "--train_task Image_Classification --learning_model CNN --backbone_name resnet50 --class_grouping True --train_dataset_name Imagenet --checkpoint_name None "
                        "--checkpoints_main_path None "
                        "--test_task Feature_representation --test_task_level Image_Level --image_handling Entire_Image --batch_size 1 --testcrop_size_rows 128 --testcrop_size_cols 128 "
                        "--test_dataset_name IFREMERData_S2 --results_name Resnet_Trained_Imagenet_Tested_IFREMERData "
                        "--results_main_path /datawork/EXPERIMENTS/ "
                        "--dataset_main_path /datawork/DATA/IFREMERData_S2/"
                        )
    if args.running_in == 'Datarmor_PBS':
        Schedule.append("python $HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TestLearningModel.py --phase test "
                        "--train_task Image_Classification --learning_model CNN --backbone_name resnet50 --class_grouping True --train_dataset_name Imagenet --checkpoint_name None "
                        "--checkpoints_main_path None "
                        "--test_task Feature_representation --test_task_level Image_Level --image_handling Entire_Image --batch_size 1 --testcrop_size_rows 512 --testcrop_size_cols 512 "
                        "--test_dataset_name IFREMERData_S2 --results_name Resnet_Trained_Imagenet_Tested_IFREMERData "
                        "--results_main_path /datawork/EXPERIMENTS/ "
                        "--dataset_main_path /datawork/DATA/IFREMERData_S2/"
                        )




for i in range(len(Schedule)):
    os.system(Schedule[i])
