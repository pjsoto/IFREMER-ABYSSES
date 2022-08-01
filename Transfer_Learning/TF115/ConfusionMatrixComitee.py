import os
import sys
import json
import argparse
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
#import neptune.new as neptune

from IFREMER import *
from Model import *
parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_task', dest='train_task', type=str, default='Image_Classification', help='Learning Task, user can take among two alternatives Semantic_Segmentation|Image_Classification')
parser.add_argument('--learning_model', dest='learning_model', type=str, default='CNN', help='Learning model used')
parser.add_argument('--backbone_names', dest='backbone_names', type=list, default=['Vgg','ResNetV1_18','ResNetV2_18','ResNetV1_50','ResNetV2_50','Xception'], help='users can chosse between resnet50 and movilenet')
# Images dir and names
parser.add_argument('--dataset_name', dest='dataset_name', type = str, default='OTUSIFREMER_IMAGELABEL', help = 'Dataset Name: SUIM')
parser.add_argument('--dataset_csv_main_path', dest='dataset_csv_main_path', type=str, default='/datawork/DATA/CSVs/OTUS_2018/', help='Dataset CSV main path')
parser.add_argument('--csvfile_name_train', dest = 'csvfile_name_train', type = list, default = ['OTUS_Image_Classification_F1_Shells_White_fragments_ET_Ltd.csv','OTUS_Image_Classification_F2_Shells_White_fragments_ET_Ltd.csv','OTUS_Image_Classification_F3_Shells_White_fragments_ET_Ltd.csv'], help = 'CSV file name')
parser.add_argument('--csvfile_name_test',  dest = 'csvfile_name_test',  type = list, default = ['OTUS_Image_Classification_F1_Shells_White_fragments_ET_Ltd.csv','OTUS_Image_Classification_F2_Shells_White_fragments_ET_Ltd.csv','OTUS_Image_Classification_F3_Shells_White_fragments_ET_Ltd.csv'], help = 'CSV file name')
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/datawork/DATA/OTUS_2018_Doneesbrutes_Montsegur1024/', help='Main path of the dataset images')
parser.add_argument('--checkpoints_main_path', dest='checkpoints_main_path', type=str, default='/datawork/EXPERIMENTS', help='Path where checkpoints have been saved' )
parser.add_argument('--results_main_path', dest = 'results_main_path', default = '/datawork/EXPERIMENTS', help = 'Path where the results files will be saved')

args = parser.parse_args()

def main():

    if 'Shells_White_fragments' in args.csvfile_name_train[0]:
        class_names = ['0-10%','10-50%','50-100%']
    if 'Lithology' in args.csvfile_name_train[0]:
        class_names = ['Slab','Sulfurs','Vocanoclastic']
    if 'Morphology' in args.csvfile_name_train[0]:
        class_names = ['Fractured','Marbled','ScreeRubbles', 'Sedimented']

    args.results_main_path = args.results_main_path + '/RESULTS/'
    if not os.path.exists(args.results_main_path):
        print("[!] RESULTS main path doesn`t exists...")
        sys.exit()

    confusion_matrix = []
    for i in range(len(args.csvfile_name_train)):
        args.results_dir = args.results_main_path + args.dataset_name + '_results/Comitee/Comitee_' + str(len(args.backbone_names)) + '_Train_' + args.csvfile_name_train[i] + '_Test_' + args.csvfile_name_test[i] + '/'
        args.results_dir = args.results_dir + os.listdir(args.results_dir)[0] + '/'
        confusion_matrix_path = args.results_dir + 'confusion_matrix.npy'
        confusion_matrix_i = np.load(confusion_matrix_path)
        if i == 0:
            confusion_matrix = confusion_matrix_i.copy()
        else:
            confusion_matrix += confusion_matrix_i

    # Computing the average confusion matrix
    confusion_matrix = confusion_matrix/len(args.csvfile_name_train)
    ax = sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot = True, fmt='.2%', cmap='Blues')
    #ax.set_title('Tr: MS | Ts: MS\n\n')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names, va = 'center')

    plt.savefig(args.results_main_path + args.dataset_name + '_results/Comitee/Shells_White_fragments_CM_Tr_ET_Ts_ET.png', dpi=400)

if __name__ == '__main__':
    main()
