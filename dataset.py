import os
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel import processing
from collections import defaultdict
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

#import medicaltorch.transforms as mt_transforms
import torchvision as tv
import torchvision.utils as vutils
import transforms as tf

import albumentations
import albumentations.pytorch
from tqdm import *

torch.cuda.empty_cache()

def linked_augmentation(gm_batch, wm_batch, transform):

    gm_batch_size = gm_batch.size(0)

    gm_batch_cpu = gm_batch.cpu().detach()
    gm_batch_cpu = gm_batch_cpu.numpy()

    wm_batch_cpu = wm_batch.cpu().detach()
    wm_batch_cpu = wm_batch_cpu.numpy()

    samples_linked_aug = []
    sample_linked_aug = {'input': [gm_batch_cpu,
                                   wm_batch_cpu]}
    # print('GM: ', sample_linked_aug['input'][0].shape)
    # print('WM: ', sample_linked_aug['input'][1].shape)
    out = transform(sample_linked_aug)
    # samples_linked_aug.append(out)

    # samples_linked_aug = mt_datasets.mt_collate(samples_linked_aug)
    return out

class PAC2019(Dataset):
    def __init__(self, ctx, set, split=0.8):
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]
        csv_path = os.path.join(dataset_path, "PAC2019_BrainAge_Training.csv")
        dataset = []
        stratified_dataset = []

        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3].replace('\n',''))
                })

        sites = defaultdict(list)
        for data in dataset:
            sites[data['site']].append(data)

        for site in sites.keys():
            length = len(sites[site])
            if set == 'train':
                stratified_dataset += sites[site][0:int(length*split)]
            if set == 'val':
                stratified_dataset += sites[site][int(length*split):]

        self.dataset = stratified_dataset

        self.transform = tv.transforms.Compose([
            mt_transforms.ToPIL(labeled=False),
            mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                           sigma_range=(3.5, 4.0),
                                           p=0.3, labeled=False),
            mt_transforms.RandomAffine(degrees=4.6,
                                       scale=(0.98, 1.02),
                                       translate=(0.03, 0.03),
                                       labeled=False),
            mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
            mt_transforms.ToTensor(labeled=False),
        ])

    def __getitem__(self, idx):
        data = self.dataset[idx]
        filename = os.path.join(self.ctx["dataset_path"], 'gm', data['subject'] + '_gm.nii.gz')
        gm_image = torch.FloatTensor(nib.load(filename).get_fdata())
        gm_image = gm_image.permute(2, 0, 1)

        filename = os.path.join(self.ctx["dataset_path"], 'wm', data['subject'] + '_wm.nii.gz')
        wm_image = torch.FloatTensor(nib.load(filename).get_fdata())
        wm_image = wm_image.permute(2, 0, 1)

        # transformed = {
        #     'input': gm_image
        # }
        # self.transform(transformed)

        # plt.imshow(gm_image[60,:,:])
        # plt.show()
        # plt.imshow(gm_image[:,60,:])
        # plt.show()
        # plt.imshow(gm_image[:,:,60])
        # plt.show()
        #
        # raise


        return {
            'gm': gm_image,
            'wm': wm_image,
            'label': data['age']
        }

    def __len__(self):
        return len(self.dataset)


class PAC20192D(Dataset):
    def __init__(self, ctx, set, split=0.7, portion=0.8):
        """
        split: train/val split
        portion: portion of the axial slices that enter the dataset
        """
        self.ctx = ctx
        self.portion = portion
        dataset_path = ctx["dataset_path"]
        csv_path = os.path.join(dataset_path, "PAC2019_BrainAge_Training.csv")
        dataset = []
        stratified_dataset = []

        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3].replace('\n',''))
                })

        sites = defaultdict(list)
        for data in dataset:
            sites[data['site']].append(data)

        for site in sites.keys():
            length = len(sites[site])
            if set == 'train':
                stratified_dataset += sites[site][0:int(length*split)]
            if set == 'val':
                stratified_dataset += sites[site][int(length*split):]


        self.dataset = stratified_dataset
        self.slices = []

        self.transform = tv.transforms.Compose([
            mt_transforms.ToPIL(labeled=False),
            mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                           sigma_range=(3.5, 4.0),
                                           p=0.3, labeled=False),
            mt_transforms.RandomAffine(degrees=4.6,
                                       scale=(0.98, 1.02),
                                       translate=(0.03, 0.03),
                                       labeled=False),
            mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
            mt_transforms.ToTensor(labeled=False),
        ])

        self.preprocess_dataset()



    def preprocess_dataset(self):
        for i, data in enumerate(tqdm(self.dataset, desc="Loading dataset")):
            filename_gm = os.path.join(self.ctx["dataset_path"], 'gm', data['subject'] + '_gm.nii.gz')
            input_image_gm = torch.FloatTensor(nib.load(filename_gm).get_fdata())
            input_image_gm = input_image_gm.permute(2, 0, 1)

            filename_wm = os.path.join(self.ctx["dataset_path"], 'wm', data['subject'] + '_wm.nii.gz')
            input_image_wm = torch.FloatTensor(nib.load(filename_wm).get_fdata())
            input_image_wm = input_image_wm.permute(2, 0, 1)

            start = int((1.-self.portion)*input_image_gm.shape[0])
            end = int(self.portion*input_image_gm.shape[0])
            input_image_gm = input_image_gm[start:end,:,:]
            input_image_wm = input_image_wm[start:end,:,:]
            for slice_idx in range(input_image_gm.shape[0]):
                slice_gm = input_image_gm[slice_idx,:,:]
                slice_wm = input_image_wm[slice_idx,:,:]

                slice_gm = slice_gm.unsqueeze(0)
                slice_wm = slice_wm.unsqueeze(0)

                slice = torch.cat([slice_gm, slice_wm], dim=0)

                # print(slice.max(), slice.min())
                self.slices.append({
                    'image': slice,
                    'age': data['age']
                })
                # plt.imshow(slice.squeeze())
                # plt.show()




            # raise


    def __getitem__(self, idx):

        data = self.slices[idx]
        # transformed = {
        #     'input': data['image']
        # }
        # plt.imshow(data['image'][0])
        # plt.title('gm')
        # plt.show()
        # plt.imshow(data['image'][1])
        # plt.title('wm')
        # plt.show()
        gm = data['image'][0].unsqueeze(0)
        wm = data['image'][1].unsqueeze(0)

        batch = linked_augmentation(gm, wm, self.transform)
        # print('gm: ', batch['input'][0].shape)
        # print('wm: ', batch['input'][1].shape)
        batch = torch.cat([batch['input'][0], batch['input'][1]], dim=0)
        # print('Final shape: ', batch.shape)

        # transformed = self.transform(transformed)

        return {
            'input': batch,
            'label': data['age']
        }

    def __len__(self):
        return len(self.slices)

class PAC20193D(Dataset):
    def __init__(self, ctx, set, split=0.8):
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]
        csv_path = os.path.join(dataset_path, "PAC2019_BrainAge_Training.csv")
        dataset = []
        stratified_dataset = []

        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3].replace('\n',''))
                })

        sites = defaultdict(list)
        for data in dataset:
            sites[data['site']].append(data)

        for site in sites.keys():
            length = len(sites[site])
            if set == 'train':
                stratified_dataset += sites[site][0:int(length*split)]
            if set == 'val':
                stratified_dataset += sites[site][int(length*split):]

        self.dataset = stratified_dataset

        self.transform = tv.transforms.Compose([
            tf.ImgAugTranslation(10),
            tf.ImgAugRotation(40),
            tf.ToTensor(),
        ])


    def __getitem__(self, idx):
        data = self.dataset[idx]
        filename = os.path.join(self.ctx["dataset_path"], 'gm', data['subject'] + '_gm.nii.gz')
        input_image = torch.FloatTensor(nib.load(filename).get_fdata())
        input_image = input_image.permute(2, 0, 1)

        transformed = {
            'input': input_image
        }

        transformed = self.transform(transformed['input'])
        transformed = transformed.unsqueeze(0)
        # print(transformed.shape)


        return {
            'input': transformed,
            'label': data['age']
        }

    def __len__(self):
        return len(self.dataset)



####### HCP 3DCNN

class HCP3D(Dataset):
    def __init__(self, ctx, set, split=0.8):
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]
        #csv_path = os.path.join(dataset_path, "/HCPtrain.csv")
        dataset = []
        stratified_dataset = []


        ## stratified sex
        
        #csv_path = dataset_path+"/CAMCANtrain.csv"
        #csv_path = dataset_path+"/HCP_1mm_train.csv"
        csv_path = dataset_path+"/camcan_1mm_lsq_train.csv"
        
        
        #csv_path = dataset_path+"/CAMCAN_preprocessed_lsq_train.csv"


        #csv_path = dataset_path + "/CAMCAN+HCP_train.csv"

        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')

                
                dataset.append({
                    'Subject': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'filename': line[3].replace('\n', '')
                    #'site': int(line[3].replace('\n',''))
                })


                


                '''
                dataset.append({
                    'ID': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'site': int(line[3]),
                    'filename': line[4].replace('\n', '')
                })
                '''


             
        genders = defaultdict(list)
        for data in dataset:
            genders[data['Sex']].append(data)

        
        for gender in genders.keys():
            length = len(genders[gender])
            if set == 'train':
                stratified_dataset += genders[gender][0:int(length*split)]
            if set == 'val':
                stratified_dataset += genders[gender][int(length*split):]        

        
        ## stratified site
        
        
        #csv_path = dataset_path+"/HCPtrain_site.csv"

        '''
        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'Subject': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'site' : int(line[3]),
                    'filename': line[4].replace('\n','')
                })
        '''

        '''
        sites = defaultdict(list)
        for data in dataset:
            sites[data['site']].append(data)

        
        for site in sites.keys():
            length = len(sites[site])
            if set == 'train':
                stratified_dataset += sites[site][0:int(length*split)]
            if set == 'val':
                stratified_dataset += sites[site][int(length*split):]
        
        '''
        

        self.dataset = stratified_dataset

        self.transform = tv.transforms.Compose([
            #tf.ImgAugTranslation(10),
            #tf.ImgAugRotation(40),
            tf.ToTensor(),
        ])


    def __getitem__(self, idx):
        data = self.dataset[idx]
        #filename = os.path.join(self.ctx["dataset_path"], 'wm', data['subject'] + '_strc_T1w_')

        filename = os.path.join(self.ctx["dataset_path"], data['filename'])
        
        input_image = torch.FloatTensor(nib.load(filename).get_fdata())
        input_image = input_image.permute(2, 0, 1)


        '''
        transformed = {
            'input': input_image
        }
        '''
        transformed = {
            'input': input_image
        }
        transformed = self.transform(transformed['input'])
        transformed = transformed.unsqueeze(0)
        # print(transformed.shape)

        return {
            'input': transformed,
            'label': data['Age'],
            #'sex' : data['Sex'],
            #'Site' : data['site']
        }


    def __len__(self):
        return len(self.dataset)



### HCP 3DCNN test

class HCP3Dtest(Dataset):
    def __init__(self, ctx):
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]
        #csv_path = os.path.join(dataset_path, "HCPtest.csv")
        
        dataset = []
        #stratified_dataset = []
        
        #csv_path = dataset_path+"/CAMCANtest.csv"

        #csv_path = dataset_path+"/CAMCAN_preprocessed_ws_test.csv"
        #csv_path = dataset_path+"/CAMCAN_1mm_test.csv"
        csv_path = dataset_path+"/camcan_1mm_lsq_test.csv"
        
        #csv_path = dataset_path+"/CAMCAN_preprocessed_lsq_test.csv"
        #csv_path = dataset_path + "/CAMCAN+HCP_test.csv"

        #csv_path = dataset_path+"/CAMCANtest.csv"
        #csv_path = dataset_path+"/HCPtest.csv"


        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                
                '''
                dataset.append({
                    'Subject': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'filename': line[3].replace('\n', '')
                    #'site': int(line[3].replace('\n',''))
                })
                

                '''

                
                dataset.append({
                    'ID': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'filename': line[3].replace('\n', '')
                    #'site': int(line[3].replace('\n',''))
                })
                
                '''
                dataset.append({
                    'ID': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'site': int(line[3]),
                    'filename': line[4].replace('\n', '')
                })

                '''
        '''
        # site
        csv_path = dataset_path+"/HCPtest_site.csv"

        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'Subject': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'site' : int(line[3]),
                    'filename': line[4].replace('\n','')
                })



        
        sites = defaultdict(list)
        for data in dataset:
            sites[data['site']].append(data)

        
        for site in sites.keys():
            length = len(sites[site])
            if set == 'train':
                stratified_dataset += sites[site][0:int(length*split)]
            if set == 'val':
                stratified_dataset += sites[site][int(length*split):]     
        '''
        
        self.dataset = dataset

        self.transform = tv.transforms.Compose([
            #tf.ImgAugTranslation(10),
            #tf.ImgAugRotation(40),
            tf.ToTensor(),
        ])


    def __getitem__(self, idx):
        data = self.dataset[idx]
        #filename = os.path.join(self.ctx["dataset_path"], 'wm', data['subject'] + '_strc_T1w_')

        filename = os.path.join(self.ctx["dataset_path"], data['filename'])
        
        input_image = torch.FloatTensor(nib.load(filename).get_fdata())
        input_image = input_image.permute(2, 0, 1)

        transformed = {
            'input': input_image
        }

        transformed = self.transform(transformed['input'])
        transformed = transformed.unsqueeze(0)
        # print(transformed.shape)


        return {
            'input': transformed,
            'label': data['Age'],
            'sex' : data['Sex'],
            #'Site' : data['site']
        }

    def __len__(self):
        return len(self.dataset)




####### HCP 3DCNN_SFCN

class HCP3D_SFCN(Dataset):
    def __init__(self, ctx, set, split=0.8):
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]+"/HCP_preprocessed/1mm"
        #csv_path = os.path.join(dataset_path, "/HCPtrain.csv")
        dataset = []
        stratified_dataset = []


        ## stratified site
        
        csv_path = dataset_path+"/SFCN_HCP_train.csv"

        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'Subject': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'site' : int(line[3]),
                    'filename_x': line[4],
                    'filename_y' : line[5].replace('\n','')
                })

        sites = defaultdict(list)
        for data in dataset:
            sites[data['site']].append(data)

        
        for site in sites.keys():
            length = len(sites[site])
            if set == 'train':
                stratified_dataset += sites[site][0:int(length*split)]
            if set == 'val':
                stratified_dataset += sites[site][int(length*split):]
        

        

        self.dataset = stratified_dataset

        self.transform = tv.transforms.Compose([
            #tf.ImgAugTranslation(10),
            #tf.ImgAugRotation(40),
            tf.ToTensor(),
        ])


    def __getitem__(self, idx):
        data = self.dataset[idx]
        #filename = os.path.join(self.ctx["dataset_path"], 'wm', data['subject'] + '_strc_T1w_')

        filename = os.path.join(self.ctx["dataset_path"] +"/HCP_preprocessed/1mm/"+ data['filename_y'])


        hdr = nib.Nifti1Header()
        hdr.set_data_shape((160,192,160))
        hdr.set_zooms((1., 1., 1.))
        dst_aff = hdr.get_best_affine()
        to_img = nib.Nifti1Image(np.empty((160,192,160)), affine=dst_aff, header=hdr)
        # Resample input image.
        out_img = nib.processing.resample_from_to(from_img=nib.load(filename), to_vox_map=to_img, order=3, cval=0)
        # Cast to uint8.
        input_image = torch.FloatTensor(out_img.get_fdata())
        
        #input_image = torch.FloatTensor(nib.load(filename).get_fdata())

        #voxel_1 = [1,1,1]

        #input_image = nib.processing.resample_to_output(input_image, voxel_1)


        input_image = input_image.permute(2, 0, 1)

        transformed = {
            'input': input_image
        }

        transformed = self.transform(transformed['input'])
        transformed = transformed.unsqueeze(0)
        # print(transformed.shape)


        return {
            'input': transformed,
            'label': data['Age']
        }

    def __len__(self):
        return len(self.dataset)




##############################

####### HCP 3DCNN_SFCN_Test
class HCP3D_SFCN_Test(Dataset):
    def __init__(self, ctx):
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]+"/HCP_preprocessed/1mm"
        #csv_path = os.path.join(dataset_path, "HCPtest.csv")
        
        dataset = []
        #stratified_dataset = []
        
        
        # sex
        
        csv_path = ctx["dataset_path"]+"/HCP_SFCN.csv"
        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'Subject': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'filename': line[3].replace('\n', '')
                    #'site': int(line[3].replace('\n',''))
                })

        self.dataset = dataset

        self.transform = tv.transforms.Compose([
            #tf.ImgAugTranslation(10),
            #tf.ImgAugRotation(40),
            #tv.transforms.Resize((160,192,160), T.InterpolationMode.BICUBIC),
            tf.ToTensor()
        ])


    def __getitem__(self, idx):
        data = self.dataset[idx]
        #filename = os.path.join(self.ctx["dataset_path"], 'wm', data['subject'] + '_strc_T1w_')

        filename = os.path.join(self.ctx["dataset_path"]+"/HCP_preprocessed/1mm", data['filename'])
        
        input_image = torch.FloatTensor(nib.load(filename).get_fdata())
        #print(input_image.shape)
        #input_image = input_image.permute(2, 0, 1)

        #transformed = {
        #    'input': input_image
        #}

        sp = (1,) + input_image.shape

        input_image = input_image.reshape(sp)

        transformed = {
            'input': input_image
        }


        transformed = self.transform(transformed['input'])
        transformed = transformed.unsqueeze(0)


        print(transformed.shape)


        return {
            'input': transformed,
            'label': data['Age']
        }

    def __len__(self):
        return len(self.dataset)


##################################
##################################


class HCP3D_sex(Dataset):
    def __init__(self, ctx, set, split=0.8):
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]
        dataset = []
        stratified_dataset = []


        ## stratified sex
        
        csv_path = dataset_path+"/HCPtrain.csv"
        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'Subject': line[0],
                    'Age': float(line[1]),
                    'Sex': int(line[2]),
                    'filename': line[3].replace('\n', '')
                    #'site': int(line[3].replace('\n',''))
                })
        genders = defaultdict(list)
        for data in dataset:
            genders[data['Sex']].append(data)

        
        for gender in genders.keys():
            length = len(genders[gender])
            if set == 'train':
                stratified_dataset += genders[gender][0:int(length*split)]
            if set == 'val':
                stratified_dataset += genders[gender][int(length*split):]        


        self.dataset = stratified_dataset

        self.transform = tv.transforms.Compose([

            tf.ToTensor(),
        ])


    def __getitem__(self, idx):
        data = self.dataset[idx]

        filename = os.path.join(self.ctx["dataset_path"], data['filename'])
        
        input_image = torch.FloatTensor(nib.load(filename).get_fdata())
        input_image = input_image.permute(2, 0, 1)

        transformed = {
            'input': input_image 
        }

        transformed = self.transform(transformed['input'])
        transformed = transformed.unsqueeze(0)
        # print(transformed.shape)

        return {
            'input': transformed,
            'label': data['Age'],
            'sex' : data['Sex']
        }


    def __len__(self):
        return len(self.dataset)