import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from model import Model, ColeModel, ColeModel_bn_change, ColeModel_double_bn, SFCN, Nature_ver1, Nature_ver2, Nature_sex, Nature_sex_site


#from model import Model, VGGBasedModel, VGGBasedModel2D, ColeModel
import json
from pathlib import Path
#from torchsummary import summary

from dataset import HCP3Dtest,HCP3D_SFCN_Test


#from dataset import PAC2019, PAC20192D, PAC20193D, HCP3D, HCP3Dtest
#from model_resnet import ResNet, resnet18

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import scipy.stats as stats
from tqdm import *

import os

## GPU number 설정
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'


torch.cuda.empty_cache()

#with open('config_mj.json') as fid :
#    ctx = json.load(fid)

with open("config_mj.json", 'r', encoding='utf-8') as fid:
    ctx = json.load(fid)
    #print(ctx)

test_set = HCP3Dtest(ctx)
#test_set = HCP3D_SFCN_Test(ctx)

SAVEPATH_csv = Path(ctx['save_path'][:-2] + 'csv')
SAVEPATH_txt = Path(ctx['save_path'][:-2] + 'txt')

# Model 선택
model = ColeModel_double_bn()
#model = ColeModel_bn_change()
#model = ColeModel()
#model = SFCN()
#model = Nature_ver1()
#model = Nature_ver2()

#model = Nature_sex_site()

#model = Nature_sex()
model.load_state_dict(torch.load(ctx['save_path']))
'''
models_dict = {}
key = 'run_20190719_00_epoch_best_mae'
models_dict['key'] = [ctx['save_path']]

model_dir = './models/'
model_weights = ctx['save_path']
model_checkpoint = model_dir + model_weights
'''

'''
if os.path.isfile(model_checkpoint):
    print("=> loading checkpoint '{}'".format(model_checkpoint))

    print('model checkpoint', model_checkpoint)
    checkpoint = torch.load(model_checkpoint, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}



    #print('state_dict', state_dict)


    for k,v in state_dict.items():
        #if k in ["module.fc.0.bias" , "module.fc.0.weight"] :

        k_ = k.replace('module.', '')
        new_state_dict[k_] = v
        #print('k_ !!!!!!!!!!!!!, ',k_)
        
        if k_ in ["fc.0.bias" , "fc.0.weight"] : 

            k_ = k_.replace(".0","")
            
            #k_ = k_.replace("moudle.","")

            #k_ = k_.replace("module","")

            #k_ = k

            #print('kkkkkkkk :,', k_)
            new_state_dict[k_] = v
        else : 
            new_state_dict[k_] = v
        
    model.load_state_dict(new_state_dict)
'''
model.cuda()

torch.cuda.empty_cache()
#model.load_state_dict(torch.load(ctx["save_path"]))

#print(model.cuda())
test_loader = DataLoader(test_set, shuffle = False, num_workers = 0, batch_size = ctx['batch_size'])

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
subject = []

output_list = []
label_list = []
test_mae_loss = []
for i, data in enumerate(test_loader):
    
    torch.cuda.empty_cache()

    input_image = Variable(data["input"]).float().cuda()
    
    with torch.no_grad():

        ### sextype

        #Sextype = Variable(data['sex']).float().cuda()        
        #output = model(input_image, Sextype)

        output = model(input_image)


        #Sextype = Variable(data['sex']).float().cuda()
        #Sitetype = Variable(data['Site']).float().cuda()
        
        #output = model(input_image, Sextype, Sitetype)

        label = Variable(data["label"].float()).cuda()



        #loss = mse_loss(output.squeeze(), label)

        loss = mae_loss(output.squeeze(), label)
        test_mae_loss.append(loss.data)
        #print('val')
        #print(output, label, val_mae_loss, len(val_mae_loss))

        # loss = torch.mean(torch.abs(output.squeeze() - label))
        # val_mae_loss.append(loss.data)
        #print(Variable(data['info']).float().cuda())
        #print(output.squeeze())
        #print(output.squeeze().tolist)
        ### 해당 부분 확인 후 list를 csv로 추가해서 r값 구하기
        #output_list.append(output.squeeze())
        #label_list.append(label)
        #print(label)

        #print(output.squeeze().tolist)

        #new_tensor = torch.tensor(output.squeeze().tolist(), device = 'cpu')
        new_tensor = output.squeeze().detach().cpu().tolist()

        #output_list.append(output.squeeze().tolist().cpu())
        #print('New tensor type: ', type(new_tensor))

        output_list.append(new_tensor)
        #label_list.append(label)

        #new_label = torch.tensor(label, device = 'cpu')
        new_label = label.detach().cpu().tolist()

    #print('New label type: ', type(new_label))
    label_list.append(new_label)
    torch.cuda.empty_cache()

torch.cuda.empty_cache()
# print('Validation Loss (MSE): ', torch.mean(torch.stack(val_mse_loss)))

#tqdm.write('TEST Loss (MAE): %f' % torch.mean(torch.stack(test_mae_loss)).item())
tqdm.write('TEST Loss (MSE): %f' % torch.mean(torch.stack(test_mae_loss)).item())


#print(output_list)


#outputs = output_list.numpy()
#labels = label_list.cpu().numpy()

# 2D -> 1D
output_list = sum(output_list, [])
label_list = sum(label_list, [])

outputs = np.array(output_list)
labels = np.array(label_list)



#outputs = torch.cat(output_list).cpu().numpy()
#labels = torch.cat(label_list).cpu().numpy()

print('output', outputs)
print('label', labels)

df = pd.DataFrame({
    'Age' : labels,
    'output' : outputs
})
df['Diff'] = abs(df['Age'] - df['output'])
df.to_csv(SAVEPATH_csv, index = False)

# 상관계수
r2 = r2_score(df['Age'], df['output'])

print('r2 score :', r2)

corr = stats.pearsonr(df.Age, df.output)
print('correlation , p-value : ', corr)

textfile = open(SAVEPATH_txt, 'w')
textfile.write('Test loss : '+ str(torch.mean(torch.stack(test_mae_loss)).item())+ '\n')
textfile.write('r2 score :'+ str(r2) + '\n')
textfile.write('correlation , p-value :'+ str(corr) + '\n')
textfile.close()

'''
# visualization
plt.scatter(labels, output, s=3)
plt.xlabel('Age')
plt.ylabel('Predicted Age')
#plt.plot(labels, output, color = 'r')
plt.show()

'''

torch.cuda.empty_cache()