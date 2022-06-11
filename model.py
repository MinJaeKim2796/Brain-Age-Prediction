import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from batchrenorm import BatchRenorm3d
import torch
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(121, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout2d(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(4320, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 1),
            # nn.BatchNorm1d(),
            nn.ReLU()
        )


        # self.linear1 = nn.Linear()


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]))
        # print(x.shape)

        return x


class VGGBasedModel(nn.Module):
    def __init__(self):
        super(VGGBasedModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(121, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.classifier = nn.Sequential(
            nn.Linear(3072, 2048),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 1),
            # nn.BatchNorm1d(),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]))
        # print(x.shape)

        return x


class VGGBasedModel2D(nn.Module):
    def __init__(self):
        super(VGGBasedModel2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.classifier = nn.Sequential(
            nn.Linear(3072, 2048),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 1),
            # nn.BatchNorm1d(),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]))
        # print(x.shape)

        return x

class ColeModel(nn.Module):
    def __init__(self):
        super(ColeModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1),
            nn.ReLU(inplace = True),
            nn.Conv3d(8, 8, 3, stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, stride=1),
            nn.ReLU(inplace = True),
            nn.Conv3d(16, 16, 3, stride=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2))

        self.block3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=1),
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 32, 3, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2))

        self.block4 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2))

        self.block5 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2)
        )

        # 1.5mm
        #self.classifier = nn.Linear(1536, 1)

        # 1mm
        #self.classifier = nn.Linear(10240, 1)
        self.classifier = nn.Linear(31360, 1)
        
        # 0.7mm
        #self.classifier = nn.Linear(72576, 1)

    def forward(self, x):
        x = self.block1(x) 
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)    
        x = self.block5(x)
        # print(x.shape)
        x = self.classifier(x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))
        # print(x.shape)

        return x


class ColeModel_bn_change(nn.Module):
    def __init__(self):
        super(ColeModel_bn_change, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, stride=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2))

        self.block3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2))

        self.block4 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=1, padding=1),            
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2))

        self.block5 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )

        self.classifier = nn.Linear(1536, 1)


    def forward(self, x):
        x = self.block1(x) 
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)    
        x = self.block5(x)
        # print(x.shape)
        x = self.classifier(x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))
        # print(x.shape)

        return x


class ColeModel_double_bn(nn.Module):
    def __init__(self):
        super(ColeModel_double_bn, self).__init__()


        self.block1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True),
            nn.Conv3d(8, 8, 3, stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, stride=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.Conv3d(16, 16, 3, stride=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2))

        self.block3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 32, 3, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2))

        self.block4 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=1, padding=1),            
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2))

        self.block5 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True),
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool3d(2, stride=2)
        )

        # 추가부분 
        #self.dropout = nn.Dropout(p=0.2)

        self.classifier = nn.Linear(31360, 1)


        ##
        #self.classifier = nn.Linear(10240, 1)
        #self.classifier = nn.Linear(1536, 1)


    def forward(self, x):
        x = self.block1(x) 
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)    
        x = self.block5(x)
        # print(x.shape)


        # 추가부분 
        #x = self.dropout(x)        
        ##


        x = self.classifier(x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))
        # print(x.shape)

        return x




class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        #avg_shape = [5, 5,6]
  
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        out.append(x)
        return out



class Nature_ver1(nn.Module):
    def __init__(self):#, is_plain = True):
        super(Nature_ver1, self).__init__()

        self.after_add = nn.Sequential(
             nn.ELU(),
             nn.MaxPool3d(2, stride = 2, padding = 1)
        )

        #self.is_plain = is_plain
        #self.iden_1 = nn.Conv3d(1, 8, 3, stride = 1)
        #self.iden_2 = nn.Conv3d(8, 16, 3, stride = 1)
        #self.iden_3 = nn.Conv3d(16, 32, 3, stride = 1)
        #self.iden_4 = nn.Conv3d(32, 64, 3, stride = 1)
        #self.iden_5 = nn.Conv3d(64, 128, 3, stride = 1)
        self.iden = nn.Identity()

        self.block1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding = 1),
            BatchRenorm3d(8),
            nn.ELU(),
            nn.Conv3d(8, 8, 3, stride=1, padding = 1),
            BatchRenorm3d(8))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, stride=1, padding = 1),
            BatchRenorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 16, 3, stride=1, padding = 1),
            BatchRenorm3d(16))

        self.block3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=1, padding = 1),
            BatchRenorm3d(32),
            nn.ELU(),
            nn.Conv3d(32, 32, 3, stride=1, padding = 1),
            BatchRenorm3d(32))

        self.block4 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=1, padding=1),            
            BatchRenorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            BatchRenorm3d(64))

        self.block5 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            BatchRenorm3d(128),
            nn.ELU(),
            nn.Conv3d(128, 128, 3, stride=1),#, padding=1),
            BatchRenorm3d(128))

        self.classifier = nn.Linear(10240, 1)
        #self.classifier = nn.Linear(31360, 1)

        '''


        nn.init.uniform_(self.block1.weight.data)
        nn.init.uniform_(self.block2.weight.data)
        nn.init.uniform_(self.block3.weight.data)
        nn.init.uniform_(self.block4.weight.data)
        nn.init.uniform_(self.block5.weight.data)

        '''

        '''
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                nn.init.uniform_(m.weight.data)
        '''


    def forward(self, x):
        x = self.block1(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

        x = self.block2(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

        x = self.block3(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

        x = self.block4(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)    

        x = self.block5(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

###여기서부터 fully connected

        x = self.classifier(x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4]))
        # print(x.shape)

        return x

class Nature_ver2(nn.Module):
    def __init__(self):#, is_plain = True):
        super(Nature_ver2, self).__init__()

        self.flatten = nn.Flatten()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.2)
        self.after_add = nn.Sequential(
             nn.ELU(),
             nn.MaxPool3d(2, stride = 2, padding = 1)
        )
        #self.is_plain = is_plain
        self.iden_1 = nn.Conv3d(1, 8, 3, stride = 1)
        self.iden_2 = nn.Conv3d(8, 16, 3, stride = 1)
        self.iden_3 = nn.Conv3d(16, 32, 3, stride = 1)
        self.iden_4 = nn.Conv3d(32, 64, 3, stride = 1)
        self.iden_5 = nn.Conv3d(64, 128, 3, stride = 1)
        self.iden = nn.Identity()


        self.iden_1 = nn.Conv3d(1,8,1,1,1)


        self.block1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding = 1),
            BatchRenorm3d(8),
            nn.ELU(),
            nn.Conv3d(8, 8, 3, stride=1, padding = 1),
            BatchRenorm3d(8))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, stride=1, padding = 1),
            BatchRenorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 16, 3, stride=1, padding = 1),
            BatchRenorm3d(16))

        self.block3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=1, padding = 1),
            BatchRenorm3d(32),
            nn.ELU(),
            nn.Conv3d(32, 32, 3, stride=1, padding = 1),
            BatchRenorm3d(32))

        self.block4 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=1, padding=1),            
            BatchRenorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            BatchRenorm3d(64))

        self.block5 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            BatchRenorm3d(128),
            nn.ELU(),
            nn.Conv3d(128, 128, 3, stride=1),#, padding=1),
            BatchRenorm3d(128))

        self.classifier = nn.Linear(10240, 1)

    def forward(self, x):
        x = self.block1(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)

        # print(x.shape)
        x = self.block2(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)

        # print(x.shape)

        x = self.block3(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

        x = self.block4(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)
        #     
        x = self.block5(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

###여기서부터 fully connected

        x = self.flatten(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.classifier(x)


        # print(x.shape)

        return x



class Nature_sex(nn.Module):
    def __init__(self):#, is_plain = True):
        super(Nature_sex, self).__init__()

        self.flatten = nn.Flatten()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.2)
        self.after_add = nn.Sequential(
             nn.ELU(),
             nn.MaxPool3d(2, stride = 2, padding = 1)
        )
        #self.is_plain = is_plain
        self.iden_1 = nn.Conv3d(1, 8, 3, stride = 1)
        self.iden_2 = nn.Conv3d(8, 16, 3, stride = 1)
        self.iden_3 = nn.Conv3d(16, 32, 3, stride = 1)
        self.iden_4 = nn.Conv3d(32, 64, 3, stride = 1)
        self.iden_5 = nn.Conv3d(64, 128, 3, stride = 1)
        self.iden = nn.Identity()


        self.iden_1 = nn.Conv3d(1,8,1,1,1)


        self.block1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding = 1),
            BatchRenorm3d(8),
            nn.ELU(),
            nn.Conv3d(8, 8, 3, stride=1, padding = 1),
            BatchRenorm3d(8))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, stride=1, padding = 1),
            BatchRenorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 16, 3, stride=1, padding = 1),
            BatchRenorm3d(16))

        self.block3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=1, padding = 1),
            BatchRenorm3d(32),
            nn.ELU(),
            nn.Conv3d(32, 32, 3, stride=1, padding = 1),
            BatchRenorm3d(32))

        self.block4 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=1, padding=1),            
            BatchRenorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            BatchRenorm3d(64))

        self.block5 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            BatchRenorm3d(128),
            nn.ELU(),
            nn.Conv3d(128, 128, 3, stride=1),#, padding=1),
            BatchRenorm3d(128))

        self.classifier = nn.Linear(10240, 1)

    def forward(self, x, sex):
        x = self.block1(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)

        # print(x.shape)
        x = self.block2(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)

        # print(x.shape)

        x = self.block3(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

        x = self.block4(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)
        #     
        x = self.block5(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

###여기서부터 fully connected

        x = self.flatten(x)
        x = self.elu(x)
        x = self.dropout(x)


        
        #print(sex.shape)
        #scanner = (1,)
        #gender = (1,)
        #print(sex.item())
        #gender = sex.item()
        #for i in range(len(sex.item())):
        #    gender[i] = sex.item()
        #print('type gender', type(gender), 'gender : ', gender)
        #x = torch.cat((x, scanner, gender), dim = 1)
        #print(x.shape)
        #print(sex.shape)
        x = torch.cat([x, sex.reshape(-1,1)], axis = 1)#, dim = 0)
        #print(sex.shape)
        #print(x.shape)



        x = self.classifier(x)


        # print(x.shape)

        return x

class Nature_sex_site(nn.Module):
    def __init__(self):#, is_plain = True):
        super(Nature_sex_site, self).__init__()

        self.flatten = nn.Flatten()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.2)
        self.after_add = nn.Sequential(
             nn.ELU(),
             nn.MaxPool3d(2, stride = 2, padding = 1)
        )
        #self.is_plain = is_plain
        self.iden_1 = nn.Conv3d(1, 8, 3, stride = 1)
        self.iden_2 = nn.Conv3d(8, 16, 3, stride = 1)
        self.iden_3 = nn.Conv3d(16, 32, 3, stride = 1)
        self.iden_4 = nn.Conv3d(32, 64, 3, stride = 1)
        self.iden_5 = nn.Conv3d(64, 128, 3, stride = 1)
        self.iden = nn.Identity()


        self.iden_1 = nn.Conv3d(1,8,1,1,1)


        self.block1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding = 1),
            BatchRenorm3d(8),
            nn.ELU(),
            nn.Conv3d(8, 8, 3, stride=1, padding = 1),
            BatchRenorm3d(8))

        self.block2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, stride=1, padding = 1),
            BatchRenorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 16, 3, stride=1, padding = 1),
            BatchRenorm3d(16))

        self.block3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=1, padding = 1),
            BatchRenorm3d(32),
            nn.ELU(),
            nn.Conv3d(32, 32, 3, stride=1, padding = 1),
            BatchRenorm3d(32))

        self.block4 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=1, padding=1),            
            BatchRenorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            BatchRenorm3d(64))

        self.block5 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            BatchRenorm3d(128),
            nn.ELU(),
            nn.Conv3d(128, 128, 3, stride=1),#, padding=1),
            BatchRenorm3d(128))

        self.classifier = nn.Linear(10242, 1)

    def forward(self, x, sex, site):
        x = self.block1(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)

        # print(x.shape)
        x = self.block2(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)

        # print(x.shape)

        x = self.block3(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

        x = self.block4(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)
        #     
        x = self.block5(x)
        #if self.is_plain :
        x = x + self.iden(x)
        x = self.after_add(x)
        # print(x.shape)

###여기서부터 fully connected

        x = self.flatten(x)
        x = self.elu(x)
        x = self.dropout(x)


        
        #print(sex.shape)
        #scanner = (1,)
        #gender = (1,)
        #print(sex.item())
        #gender = sex.item()
        #for i in range(len(sex.item())):
        #    gender[i] = sex.item()
        #print('type gender', type(gender), 'gender : ', gender)
        #x = torch.cat((x, scanner, gender), dim = 1)
        #print(x.shape)
        #print(sex.shape)
        x = torch.cat([x, sex.reshape(-1,1), site.reshape(-1,1)], axis = 1)#, dim = 0)
        #print(sex.shape)
        #print(x.shape)



        x = self.classifier(x)


        # print(x.shape)

        return x


class SFCN_no_transfer(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=8, dropout=True):
        super(SFCN_no_transfer, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        #avg_shape = [5, 6, 5]
        avg_shape = [5, 5,6]
  
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

        self.Linear = nn.Linear(5000, 1)
    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        #x = self.Linear(x_f)
        x = F.log_softmax(x, dim=1)
        out.append(x)
        return x