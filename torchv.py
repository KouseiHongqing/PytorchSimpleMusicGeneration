'''
函数说明: 
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-16 17:02:32
'''
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from numpy.core.fromnumeric import mean, std
# %matplotlib inline
np.random.seed(1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,Dataset
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from pytorch_data_utils import *

class mydata(Dataset):
    def __init__(self) -> None:
        super().__init__()
        X, Y, n_values, indices_values = load_music_utils()
        Y = np.transpose(Y,[1,0,2]).argmax(-1)
        # Y = Y[:,:,np.newaxis]
        self.X = X
        self.Y = Y
    def __getitem__(self, index):
       
        return self.X[index],self.Y[index]

    def __len__(self):
        return self.X.shape[0]

def my_collate(batch):
    data = [item[0] for item in batch]
    data = torch.FloatTensor(data)
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

batchsize=4
dataset = mydata()
datas = dataloader.DataLoader(dataset,batchsize,True,collate_fn=my_collate)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(90,64,1)
        self.dense = nn.Linear(64,256)
        self.dense2 = nn.Linear(256,512)
        self.fc = nn.Linear(512,90)

    def forward(self,x,h=None,c=None):
        hp=(h,c) if h!=None and c!=None else None
        x,hid = self.lstm(x,hp)
        x = F.relu(self.dense(x))
        x = F.relu(self.dense2(x))
        return self.fc(x),hid

net = Net()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
lossfunc = nn.CrossEntropyLoss()

for i in range(100000):
    for _,(x,y) in enumerate(datas):
        out,hid = net(x)
        loss = lossfunc(out.reshape(-1,90),y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('episode{} finished,loss = {}'.format(i,loss.item()))
generate_music(net)