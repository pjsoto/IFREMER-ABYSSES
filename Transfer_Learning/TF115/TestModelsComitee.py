import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
#import neptune.new as neptune

from IFREMER import *
from Model import *
parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_task', dest='train_task', type=str, default='Image_Classification', help='Learning Task, user can take among two alternatives Semantic_Segmentation|Image_Classification')
parser.add_argument('--learning_model', dest='learning_model', type=str, default='CNN', help='Learning model used')
parser.add_argument('--backbone_names', dest='backbone_names', type=str, default=['Vgg','ResNetV2_50'], help='users can chosse between resnet50 and movilenet')
parser.add_argument('--pretrained_backbone', dest = 'pretrained_backbone', type=eval, choices=[True, False], default=False, help = 'Decide if the bockbone will be a pretrained one or will be trained from scratch')
parser.add_argument('--labels_type', dest='labels_type', type=str, default='multiple_labels', help='users can choose between onehot_labels(Image Classification) or multiple_labels(Multilabel Image Classification)')

parser.add_argument('--phase', dest='phase', type = str,default='test', help='train, test, generate_image, create_dataset')
parser.add_argument('--feature_representation', dest = 'feature_representation', type=eval, choices=[True, False], default=False, help = 'This paraemeter is used to decide if a feature representation will be accomplished')
parser.add_argument('--layer_index', dest = 'layer_index', type = int, default = 19, help = 'Definition of the layer where the feature will be taken')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='number images in batch')
# Images pre-processing hyper-parameters
parser.add_argument('--image_rows', dest='image_rows', type=int, default=1024, help='Image dimensions in rows')
parser.add_argument('--image_cols', dest='image_cols', type=int, default=1024, help='Image dimensions in columns')
parser.add_argument('--image_channels', dest='image_channels', type=int, default=3, help='Number of channels of images')
parser.add_argument('--new_size_rows', dest='new_size_rows', type=int, default=1024, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--new_size_cols', dest='new_size_cols', type=int, default=1024, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--split_patch',dest='split_patch' ,type=eval, choices=[True, False], default=False, help = 'Decide if the images can be splitted during training')
parser.add_argument('--overlap_porcent', dest = 'overlap_porcent', type = float, default = 0, help = 'Specifies the overlap between the pateches extracted from the images')
# Images dir and names
parser.add_argument('--dataset_name', dest='dataset_name', type = str, default='OTUSIFREMER_IMAGELABEL', help = 'Dataset Name: SUIM')
parser.add_argument('--dataset_csv_main_path', dest='dataset_csv_main_path', type=str, default='/datawork/DATA/CSVs/OTUS_2018/', help='Dataset CSV main path')
parser.add_argument('--csvfile_name_train', dest = 'csvfile_name_train', type = str, default = 'OTUS_Image_Classification_F1_Lithology_MS_Ltd.csv', help = 'CSV file name')
parser.add_argument('--csvfile_name_test', dest = 'csvfile_name_test', type = str, default = 'OTUS_Image_Classification_F1_Lithology_MS_Ltd.csv', help = 'CSV file name')
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/', help='Main path of the dataset images')
#parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='ResNetV2_50/Model_CNN_ResNetV2_50_OTUS_Image_Classification_F1_Morphology.csv', help='Checkpoints folder name')
parser.add_argument('--checkpoints_main_path', dest='checkpoints_main_path', type=str, default='/datawork/EXPERIMENTS/', help='Path where checkpoints have been saved' )
parser.add_argument('--results_main_path', dest = 'results_main_path', default = '/datawork/EXPERIMENTS/', help = 'Path where the results files will be saved')

args = parser.parse_args()

def main():

    args.checkpoint_name =  args.backbone_names[0] + "/Model_CNN_" + args.backbone_names[0] + "_" + args.csvfile_name_train + " "
    args.checkpoints_main_path = args.checkpoints_main_path + '/CHECKPOINTS/'
    args.checkpoint_dir = args.checkpoints_main_path + args.dataset_name + '_checkpoints/' + args.checkpoint_name

    args.results_main_path = args.results_main_path + '/RESULTS/'
    if not os.path.exists(args.results_main_path):
        os.makedirs(args.results_main_path)

    args.results_dir = args.results_main_path + args.dataset_name + '_results/Comitee_' + str(len(args.backbone_names)) + '_Train_' + args.csvfile_name_train + '_Test_' + args.csvfile_name_test + '/'
    print(args.results_dir)
    sys.exit()
    print("Dataset pre-processing...")
    if args.dataset_name == 'SUIM':
        dataset = SUIM(args)
    if args.dataset_name == 'IFREMERData':
        dataset = IFREMER(args)
    if args.dataset_name == 'OTUSIFREMER_IMAGELABEL':
        dataset = OTUSIFREMER_IMAGELABEL(args)

    checkpoint_files = os.listdir(args.checkpoint_dir)

    for i in range(len(checkpoint_files)):
        model_folder = checkpoint_files[i]
        args.trained_model_path = args.checkpoint_dir + '/' + model_folder + '/'

        model_folder_fields = model_folder.split('_')
        Trained_Name = ''
        for f in range(len(model_folder_fields)):
            if f != 0:
                Trained_Name = Trained_Name + model_folder_fields[f] + '_'
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

        args.save_results_dir = args.results_dir + args.learning_model + '_' + 'Trained_' + Trained_Name + 'Tested_' + dt_string +'/'
        if not os.path.exists(args.save_results_dir):
            os.makedirs(args.save_results_dir)

        print('[*]Initializing the model...')
        model = Model(args, dataset)
        print('[*]Model evaluation running...')
        model.Test()

if __name__ == '__main__':
    main()
