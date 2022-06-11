import json
from pathlib import Path

from dataset import HCP3D#PAC2019, PAC20192D, PAC20193D, HCP3D, HCP3D_SFCN
from model import Nature_ver2, Nature_ver1, Nature_sex, ColeModel, ColeModel_double_bn, Nature_sex_site#Model, VGGBasedModel, VGGBasedModel2D, ColeModel, ColeModel_double_bn, SFCN, Nature_ver1, Nature_ver2
from model_resnet import ResNet, resnet18

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from tqdm import *

from time import sleep
import multiprocessing
torch.cuda.empty_cache()

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def cosine_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr * cosine_rampdown(current_epoch, num_epochs)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def Trains(args):

    with open("config_mj.json", 'r', encoding='utf-8') as fid:
        ctx = json.load(fid)
        #print(ctx)

    if ctx["3d"]:
        #train_set = PAC20193D(ctx, set='train')
        #val_set = PAC20193D(ctx, set='val')


        train_set = HCP3D(ctx, set='train')
        val_set = HCP3D(ctx, set = 'val')

        #train_set = HCP3D_SFCN(ctx, set='train')
        #val_set = HCP3D_SFCN(ctx, set = 'val')


        ############
        # model 선택!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ############
        #model = ColeModel()
        #model = ColeModel_bn_change()
        model = ColeModel_double_bn()
        #model = SFCN()
        #model = Nature_ver1()
        #model = Nature_ver2()
        #model = Nature_sex()

        #model = Nature_sex_site()

        # Cole Model
        #optimizer = torch.optim.SGD(model.parameters(), lr=ctx["learning_rate"],
        #                             momentum=0.9, weight_decay=ctx["weight_decay"])


        # Nature Model
        optimizer = torch.optim.Adam(model.parameters(), lr=ctx["learning_rate"], weight_decay=ctx["weight_decay"])   

        ########################
        # Scheduler 선택!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ########################

        # 1. LR Step
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

        # 2. Multi Step
        #   1) 100 epoch
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10,20,30,40,50,60,70,80,90], gamma=0.95)
        
        #   2) 150 epoch
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [15,30,45,60,75,90,105,120,135], gamma=0.95)


    else:
        train_set = PAC20192D(ctx, set='train', split=0.8)
        val_set = PAC20192D(ctx, set='val', split=0.8)
        model = resnet18()
        optimizer = torch.optim.Adam(model.parameters(), lr=ctx["learning_rate"],
                                    weight_decay=ctx["weight_decay"])

    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                                num_workers=args.workers, batch_size=ctx["batch_size"])
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False,
                                num_workers=args.workers, batch_size=ctx["batch_size"])#, multiprocessing_context=multiprocessing.get_context('loky'))

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    model.cuda()

    torch.cuda.empty_cache()

    #nn.init.kaiming_uniform_(model.we)


    #print(list(model.parameters()))
    SAVEPATH_txt = Path(ctx['save_path'][:-3] + '_loss.txt')
    textfile = open(SAVEPATH_txt, 'w')
    textfile.write('Epochs'+ '\t' + 'Train'+'\t' + 'Val' + '\n')

    best = np.inf
    for e in tqdm(range(1, ctx["epochs"]+1), desc="Epochs"):
        model.train()
        last_100 = []
        #last_5 = []
        train_mae_loss = []
        total = 0
        input_train = 0
        if ctx["3d"]:
            
            if e > 1 :
            ###
            #cole model scheduler
                scheduler.step()
            #tqdm.write('Learning Rate: {:.6f}'.format(scheduler.get_lr()[0]))
            tqdm.write('Learning Rate: {:.6f}'.format(scheduler.get_last_lr()[0]))
            

            '''    
            if e <= ctx["initial_lr_rampup"]:
                lr = ctx["learning_rate"] * sigmoid_rampup(e, ctx["initial_lr_rampup"])
            else:
                lr = cosine_lr(e-ctx["initial_lr_rampup"],
                                ctx["epochs"]-ctx["initial_lr_rampup"],
                                ctx["learning_rate"])
            tqdm.write("Learning Rate: {:.6f}".format(lr))
            '''
            #lr = ctx["learning_rate"]
            #tqdm.write("Learning Rate: {:.6f}".format(lr))



        else:
            if e <= ctx["initial_lr_rampup"]:
                lr = ctx["learning_rate"] * sigmoid_rampup(e, ctx["initial_lr_rampup"])
            else:
                lr = cosine_lr(e-ctx["initial_lr_rampup"],
                                ctx["epochs"]-ctx["initial_lr_rampup"],
                                ctx["learning_rate"])

            for param_group in optimizer.param_groups:
                tqdm.write("Learning Rate: {:.6f}".format(lr))
                param_group['lr'] = lr


        for i, data in enumerate(train_loader): 
            if ctx["mixup"]:
                lam = np.random.beta(ctx["mixup_alpha"], ctx["mixup_alpha"])

                length_data = data["input"].size(0)//2
                data1_x = data["input"][0:length_data]
                data1_y = data["label"][0:length_data]
                data2_x = data["input"][length_data:]
                data2_y = data["label"][length_data:]

                data["input"] = lam*data1_x + (1.-lam)*data2_x
                data["label"] = lam*data1_y + (1.-lam)*data2_y

            # print(data["input"].shape)
            input_image = Variable(data["input"], requires_grad=True).float().cuda()
            output = model(input_image)

            ### 추가본
            #Sextype = Variable(data['sex']).float().cuda()
            #Sitetype = Variable(data['Site']).float().cuda()
            
            #output = model(input_image, Sextype, Sitetype)
            label = Variable(data["label"].float()).cuda()
            #print('train')
            #print(output)
            #print(label)


            #3DCNN

            #loss = mse_loss(output.squeeze(), label)
            loss = mae_loss(output.squeeze(), label)
            #total += len(output) * loss.item()
            #input_train += len(output)
            #print('loss', len(output), loss, total)
            
            #SFCN
            #print('output', output)
            #print('label', label)
            # #loss = mae_loss(output, label)





            optimizer.zero_grad()
            loss.backward()
            #
            #scheduler.get_last_lr()
            #lr_scheduler.StepLR()
            #

            optimizer.step()
            #scheduler.step()
            #scheduler.get_last_lr()[0]
            
            
            
            #last_100.append(loss.data)
            #if (i+1) % 100 == 0:
            #    tqdm.write('Training Loss: %f' % torch.mean(torch.stack(last_100)).item())
            #    last_100= []
            #print('label', label)    
            torch.cuda.empty_cache()

            #last_5.append(loss.data)
            #if (i+1) % 5 == 0:
                #print('last_5', last_5)
                #tqdm.write('Training Loss: %f' % torch.mean(torch.stack(last_5)).item())
                #print('train MAE',len(last_5))
                #last_5 = []
            train_loss = loss.data.detach().cpu().tolist()

            
            #train_mae_loss.append(loss.data)
            train_mae_loss.append(train_loss)


            #print(len(train_mae_loss), train_mae_loss)
        print('Train MSE Loss', np.mean(train_mae_loss))

        #print('Train MAE Loss', np.mean(train_mae_loss))
        #print('Train MAE LOSS', torch.mean(torch.stack(train_mae_loss)))
        textfile.write(str(e) + '\t' + str(np.mean(train_mae_loss)) + '\t')
            #print('train mae', total/input_train)

        # tqdm.write('Validation...')
        torch.cuda.empty_cache()

        model.eval()
        # val_mse_loss = []

        val_mae_loss = []
        for i, data in enumerate(val_loader):
            torch.cuda.empty_cache()
            input_image = Variable(data["input"]).float().cuda()

            ### 추가본
            #Sextype = Variable(data['sex']).float().cuda()
            #Sitetype = Variable(data['Site']).float().cuda()

            with torch.no_grad():
                output = model(input_image)
                #output = model(input_image)
                label = Variable(data["label"].float()).cuda()
            
                #loss = mse_loss(output.squeeze(), label)

                loss = mae_loss(output.squeeze(), label)

                val_loss = loss.data.detach().cpu().tolist()

                val_mae_loss.append(val_loss)
                #print('val')
                #print(output, label, val_mae_loss, len(val_mae_loss))

                # loss = torch.mean(torch.abs(output.squeeze() - label))
                # val_mae_loss.append(loss.data)
        
        if np.mean(val_mae_loss) < best :
            best = np.mean(val_mae_loss)
            tqdm.write('model saved')
            torch.save(model.state_dict(), ctx["save_path"])

        #if torch.mean(torch.stack(val_mae_loss)) < best:
        #    best = torch.mean(torch.stack(val_mae_loss))
        #    tqdm.write('model saved')
        #    torch.save(model.state_dict(), ctx["save_path"])

        # print('Validation Loss (MSE): ', torch.mean(torch.stack(val_mse_loss)))
        
        print('Validation MSE Loss', np.mean(val_mae_loss))

        #print('Validation MAE Loss', np.mean(val_mae_loss))

        #tqdm.write('Validation Loss (MAE): %f' % torch.mean(torch.stack(val_mae_loss)).item())
        textfile.write(str(np.mean(val_mae_loss)) + '\n') 
    textfile.close()
    torch.cuda.empty_cache()
