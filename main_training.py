import os
import sys
import shutil
import time
from tkinter import image_names
# from typing_extensions import Required
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm
import warnings

from dataset import HCP3Dtest, HCP3D, HCP3D_sex, HCP3D_SFCN_Test, HCP3D_SFCN
warnings.filterwarnings(action='ignore')
import multiprocessing
#from utils import vararg_callback_bool, vararg_callback_int
#from dataloader import  *
from torch.utils.data import DataLoader
import torch
#from engine import classification_engine



#!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!
# train part 선택!!!!!!!!!!
##!!!!!!!!!!!!!!!!!!!!!!!!!!
from training import Trains
#from training_sex import Trains
#from SFCN_training import Trains

#from SFCN_no_transfer import Trains

########!!!!!!!!!!!!!!!
#######################


sys.setrecursionlimit(40000)

def get_args_parser():
    parser = OptionParser()

    # parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=None, action="callback",
    #                   callback=vararg_callback_int)
    #parser.add_option("--model", dest="model_name", help="DenseNet121", default="Resnet50", type="string")
    #parser.add_option("--init", dest="init",
    #                  help="Random | ImageNet| or any other pre-training method",
    #                  default="Random", type="string")
    #parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
    #                  default=2, type="int") # 14 -> 4


    parser.add_option("--data_set", dest="data_set", help="ChestXray14|CheXpert", default="HCP3Dtrain", type="string") # ChestXray14 -> Covid19


    #parser.add_option("--data_set", dest="data_set", help="ChestXray14|CheXpert", default="HCP3D_sex", type="string") # ChestXray14 -> Covid19

    #parser.add_option("--data_set", dest="data_set", help="ChestXray14|CheXpert", default="HCP3D_SFCN", type="string") # ChestXray14 -> Covid19


    #parser.add_option("--normalization", dest="normalization", help="how to normalize data (imagenet|chestx-ray)", default="imagenet",
    #                  type="string")
    #parser.add_option("--img_size", dest="img_size", help="input image resolution", default=224, type="int")
    #parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
    parser.add_option("--data_dir", dest="data_dir", help="dataset dir",default=None, type="string")
    parser.add_option("--train_list", dest="train_list", help="file for training list",
                      default=None, type="string")
    #parser.add_option("--val_list", dest="val_list", help="file for validating list",
    #                  default=None, type="string")
    #parser.add_option("--test_list", dest="test_list", help="file for test list",
    #                  default=None, type="string")
    parser.add_option("--mode", dest="mode", help="train | test", default="train", type="string")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=8, type="int")
    parser.add_option("--num_epoch", dest="num_epoch", help="num of epoches", default=5, type="int")
    parser.add_option("--optimizer", dest="optimizer", help="Adam | SGD", default="Adam", type="string")
    parser.add_option("--lr", dest="lr", help="learning rate", default=1e-6, type="float")
    parser.add_option("--lr_Scheduler", dest="lr_Scheduler", help="learning schedule", default="ReduceLROnPlateau",
                      type="string")
    parser.add_option("--patience", dest="patience", help="num of patient epoches", default=30, type="int")
    #parser.add_option("--early_stop", dest="early_stop", help="whether use early_stop", default=False, action="callback",
    #                  callback=vararg_callback_bool)
    #parser.add_option("--trial", dest="num_trial", help="number of trials", default=1, type="int")
    #parser.add_option("--start_index", dest="start_index", help="the start model index", default=0, type="int")
    #parser.add_option("--clean", dest="clean", help="clean the existing data", default=False, action="callback",
    #                  callback=vararg_callback_bool)
    # parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default=False, action="callback",
    #                   callback=vararg_callback_bool)
    #parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default='', type="string")
    parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
    #parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=50, type="int")
    #parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
    #                  default=True, action="callback", callback=vararg_callback_bool)
    #parser.add_option("--proxy_dir", dest="proxy_dir", help="Path to the Pretrained model", default=None, type="string")
    #parser.add_option("--anno_percent", dest="anno_percent", help="data percent", default=100, type="int")
    parser.add_option("--device", dest="device", help="cpu|cuda", default="cuda:0", type="string")
    #parser.add_option("--activate", dest="activate", help="Sigmoid", default="Sigmoid", type="string")
    #parser.add_option("--uncertain_label", dest="uncertain_label",
    #                  help="the label assigned to uncertain data (Ones | Zeros | LSR-Ones | LSR-Zeros)",
    #                  default="LSR-Ones", type="string")
    #parser.add_option("--unknown_label", dest="unknown_label", help="the label assigned to unknown data",
    #                  default=0, type="int")

    #parser.add_option("--exp", dest="exp_num", help="exp number", default=1, type="int")
    
    (options, args) = parser.parse_args()

    return options

def main(args):
    #if args.train_list is None : 
    #    args.train_list = args.data_dir + 'train.csv'
        #args.val_list = args.data_dir + 'val.csv'
        #args.test_list = args.data_dir + 'test.csv'

    #print(args)

    #assert args.data_dir is not None
    #assert args.train_list is not None
    #assert args.val_list is not None
    #assert args.test_list is not None
    #if args.init.lower() != 'imagenet' and args.init.lower() != 'random':
    #    assert args.proxy_dir is not None

    #args.exp_name = args.model_name + "_" + args.init
    #model_path = os.path.join("./Models/Classification",args.data_set) if args.exp_num == 1 else os.path.join("./Models/Classification",args.data_set+"-"+str(args.exp_num))
    #output_path = os.path.join("./Outputs/Classification",args.data_set) if args.exp_num == 1 else os.path.join("./Outputs/Classification",args.data_set+"-"+str(args.exp_num))

    if args.data_set == 'HCP3Dtrain':
        Trains(args)


    #if args.data_set == 'HCP3D_sex':
    #    Trains(args)

    #if args.data_set == 'HCP3D_SFCN':
    #    Trains(args)


    '''
    if args.data_set == "ChestXray14":
        diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        dataset_train = ChestXray14Dataset(images_path=args.data_dir, file_path=args.train_list,
                                           augment=build_transform_classification(normalize=args.normalization, mode="train"))

        dataset_val = ChestXray14Dataset(images_path=args.data_dir, file_path=args.val_list,
                                         augment=build_transform_classification(normalize=args.normalization, mode="valid"))
        dataset_test = ChestXray14Dataset(images_path=args.data_dir, file_path=args.test_list,
                                          augment=build_transform_classification(normalize=args.normalization, mode="test"))

        classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test)
    '''



if __name__ == '__main__':
    args = get_args_parser()
    main(args)

