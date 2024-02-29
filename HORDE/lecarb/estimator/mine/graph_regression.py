import time
import logging
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, dataset

from ..estimator import Estimator, OPS
from ..utils import report_model, qerror, evaluate, run_test
from ...dataset.dataset import load_table
from ...workload.workload import load_queryset, load_labels, query_2_triple
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS

import pandas as pd
import sys

import pickle
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
def load_data_from_pkl_file(file_name):
    load_addr="./lecarb/estimator/mine/vec_data/"+file_name
    with open(load_addr, 'rb') as f:
        data = pickle.load(f)
    print("has successfully loaded data from "+load_addr)
    return data

from torch.autograd import Variable
from torch.utils import data

def Q(prediction,real,data_number):
    q_error=[]
    for i in range(len(prediction)):
        if prediction[i]==real[i]:
            q_error.append(1)
        elif prediction[i]==0:
            q_error.append(real[i].detach().numpy()*data_number)
        elif real[i]==0:
            q_error.append(prediction[i].detach().numpy()*data_number)
        elif prediction[i]>real[i]:
            q_error.append((prediction[i]/real[i]).detach().numpy())
        else:
            q_error.append((real[i]/prediction[i]).detach().numpy())

    for i in range(len(q_error)):
        if q_error[i]>100000:
            print(prediction[i],real[i])
    print("Max:",np.max(q_error)," 99th:",np.percentile(q_error,99)," 95th:",np.percentile(q_error,95)," 90th:",np.percentile(q_error,90)," 50th:",np.percentile(q_error,50)," mean:",np.mean(q_error))


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=2,
                            stride=2,
                            padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,2,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,2,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,64,2,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #nn.Dropout(0.3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64,128,2,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.mlp1 = nn.Linear(512,1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.sigmoid(x)
        return x

class q_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y ,data_num):
        plus=1/data_num
        return torch.mean((x/(y+plus))+(y/(x+plus)))


def CNN_regression(dataset,version):
    train_x=load_data_from_pkl_file("train_graph_x_"+dataset+"_"+version+".pkl")
    train_y=load_data_from_pkl_file("train_y_"+dataset+"_"+version+".pkl")
    train_probs=load_data_from_pkl_file("train_probs_"+dataset+"_"+version+".pkl")

    train_prob_correct=[None for i in range(len(train_probs))]
    attr_clusters_number=len(train_probs[0])
    for i in range(len(train_probs)):
        train_1=0
        flag=True
        for prob in train_probs[i]:
            if prob==1:
                train_1+=1
            elif prob==0:
                train_prob_correct[i]=0
                flag=False
                break
        if flag and train_1>=attr_clusters_number-1:
            train_prob_correct[i]=1
            for prob in train_probs[i]:
                train_prob_correct[i]*=prob

    # train_y=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl").query_label
    #version="original+original_cor_1.0"
    test_x=load_data_from_pkl_file("test_graph_x_"+dataset+"_"+version+".pkl")
    test_y=load_data_from_pkl_file("test_y_"+dataset+"_"+version+".pkl")
    test_probs=load_data_from_pkl_file("test_probs_"+dataset+"_"+version+".pkl")

    
    test_prob_correct=[None for i in range(len(test_probs))]
    attr_clusters_number=len(test_probs[0])
    for i in range(len(test_probs)):
        test_1=0
        flag=True
        for prob in test_probs[i]:
            if prob==1:
                test_1+=1
            elif prob==0:
                test_prob_correct[i]=0
                flag=False
                break
        if flag and test_1>=attr_clusters_number-1:
            test_prob_correct[i]=1
            for prob in test_probs[i]:
                test_prob_correct[i]*=prob

    train_x=torch.FloatTensor(train_x)
    train_y=torch.FloatTensor(train_y)
    test_x=torch.FloatTensor(test_x)
    test_y=torch.FloatTensor(test_y)
    
    print("training queries number:",len(train_y))
    print("testing queries number:",len(test_y))
    
    train_x=torch.unsqueeze(train_x, dim=1)
    train_y=torch.unsqueeze(train_y,dim=1)
    test_x=torch.unsqueeze(test_x,dim=1)
    test_y=torch.unsqueeze(test_y,dim=1)

    print(train_x.shape)
    print(train_y.shape)
    return
    train_data = TensorDataset(train_x, train_y)
    test_data =TensorDataset(test_x,test_y)


    train_loader = data.DataLoader(train_data,batch_size=100,shuffle=False)
    #test_loader = data.DataLoader(test_data,batch_size=100,shuffle=True)

    model=CNN()
    loss_func = torch.nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(),lr=0.00002)
    
    for epoch in range(100):
        for i,(x,y) in enumerate(train_loader):
            batch_x = Variable(x)
            batch_y = Variable(y)
            # print(batch_x.size())
            out = model(batch_x)
            
            loss = loss_func(out,batch_y)
            #print(loss)
        
            opt.zero_grad()  
            loss.backward()
            opt.step()
            
        print("epoch ",epoch,":")
        out=model(train_x)
        for i in range(len(train_prob_correct)):
            if train_prob_correct[i]!=None:
                out[i]=train_prob_correct[i]
        
        for i in range(len(out)):
            min_prob=min(train_probs[i])
            if out[i]>min_prob:
                out[i]=min_prob

        Q(out,train_y,False)
        out=model(test_x)
        for i in range(len(test_prob_correct)):
            if test_prob_correct[i]!=None:
                out[i]=test_prob_correct[i]
        for i in range(len(out)):
            min_prob=min(test_probs[i])
            if out[i]>min_prob:
                out[i]=min_prob
        Q(out,test_y,False)
        #print(out)
        #print(test_y)


class multi_net(nn.Module):
    def __init__(self):
        super(multi_net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=2,
                            stride=2,
                            padding=1),
            
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,2,2,1),
            
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,2,2,1),
            
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,64,2,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64,128,2,2,1),
            
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.mlp1 = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        
        self.corr_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=2,
                            stride=2,
                            padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.corr_conv2 = nn.Sequential(
            nn.Conv2d(16,32,2,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.corr_conv3 = nn.Sequential(
            nn.Conv2d(32,64,2,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.corr_conv4 = nn.Sequential(
            nn.Conv2d(64,64,2,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self,x,y):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)

        y =self.corr_conv1(y)
        y =self.corr_conv2(y)
        y =self.corr_conv3(y)
        y =self.corr_conv4(y)

        # cat=torch.cat((x,y),dim=1)
        # cat = self.mlp1(cat.view(cat.size(0),-1))
        # cat = self.sigmoid(cat)
        
        cat = self.mlp1(x.view(x.size(0),-1))
        cat = self.sigmoid(cat)
        #cat = self.relu(cat)
        return cat

class multi_input():
    def __init__(self,x,corr,y):
        self.x=x
        self.corr=corr
        self.y=y

def get_input(x,corr,y,batch_size):
    result=[]
    batch_num=int(len(x)/batch_size)
    for i in range(batch_num):
        result.append(multi_input(x[i*batch_size:(i+1)*batch_size],
                    corr[i*batch_size:(i+1)*batch_size],
                    y[i*batch_size:(i+1)*batch_size]))
    return result
        

def multi_net_regression(dataset,version) :
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    train_x=load_data_from_pkl_file("train_graph_x_"+dataset+"_"+version+".pkl")
    train_y=load_data_from_pkl_file("train_y_"+dataset+"_"+version+".pkl")
    train_probs=load_data_from_pkl_file("train_probs_"+dataset+"_"+version+".pkl")
    train_table = load_table(dataset, version)
    # print(train_x[1])
    # print(train_y[1])

    train_prob_correct=[None for i in range(len(train_probs))]
    attr_clusters_number=len(train_probs[0])
    for i in range(len(train_probs)):
        train_1=0
        flag=True
        for prob in train_probs[i]:
            if prob==1:
                train_1+=1
            elif prob==0:
                train_prob_correct[i]=0
                flag=False
                break
        if flag and train_1>=attr_clusters_number-1:
            train_prob_correct[i]=1
            for prob in train_probs[i]:
                train_prob_correct[i]*=prob

    # train_y=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl").query_label
    version="original+original_cor_1.0"
    test_x=load_data_from_pkl_file("test_graph_x_"+dataset+"_"+version+".pkl")
    test_y=load_data_from_pkl_file("test_y_"+dataset+"_"+version+".pkl")
    test_probs=load_data_from_pkl_file("test_probs_"+dataset+"_"+version+".pkl")
    test_table = load_table(dataset, version)


    test_prob_correct=[None for i in range(len(test_probs))]
    attr_clusters_number=len(test_probs[0])
    for i in range(len(test_probs)):
        test_1=0
        flag=True
        for prob in test_probs[i]:
            if prob==1:
                test_1+=1
            elif prob==0:
                test_prob_correct[i]=0
                flag=False
                break
        if flag and test_1>=attr_clusters_number-1:
            test_prob_correct[i]=1
            for prob in test_probs[i]:
                test_prob_correct[i]*=prob

    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")
    correlation=vec_data.correlation
    attr_name=vec_data.attr_name
    corr_metric=np.zeros((len(attr_name),len(attr_name)))
    for i in range(len(attr_name)):
        for j in range (len(attr_name)):
            corr_metric[i][j]=correlation[attr_name[i]][attr_name[j]]
    print(correlation)

    train_correlation=np.array([corr_metric for i in range(len(train_x))])
    test_correlation=np.array([corr_metric for i in range(len(test_x))])

    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(test_y)
    train_correlation=torch.FloatTensor(train_correlation)
    test_correlation=torch.FloatTensor(test_correlation)

    # print(train_y[:40])
    # print(test_y[:40])
    print("training queries number: ",len(train_y))
    print("testing queries number: ",len(test_y))
    train_x=torch.unsqueeze(train_x,dim=1)
    test_x=torch.unsqueeze(test_x,dim=1)
    train_y=torch.unsqueeze(train_y,dim=1)
    test_y=torch.unsqueeze(test_y,dim=1)

    train_correlation=torch.unsqueeze(train_correlation,dim=1)
    test_correlation=torch.unsqueeze(test_correlation,dim=1)

    print(train_x.shape)
    print(train_correlation.shape)

    # train_data=get_input(train_x,train_correlation,train_y,batch_size=100)
    
    train_input=torch.cat((train_x,train_correlation),dim=1)
    test_input=torch.cat((test_x,test_correlation),dim=1)
    print(train_input.size())
    
    train_data = TensorDataset(train_input,train_y)
    test_data = TensorDataset(test_input,test_y)
    train_loader = data.DataLoader(train_data, batch_size=100,shuffle=False)
    
    model=multi_net()
    loss_func = torch.nn.L1Loss()
    loss_func = q_loss()
    lr=0.00002
    lr=0.0001
    opt = torch.optim.Adam(model.parameters(),lr=lr)


    for epoch in range(200):
        if epoch%20==19:
            lr*=0.9
            opt = torch.optim.Adam(model.parameters(),lr=lr)
        for i, (input_data,batch_z) in enumerate(train_loader):
            batch_x=Variable(input_data[:,0:1,:,:])
            batch_y=Variable(input_data[:,1:2,:,:])
            out = model(batch_x,batch_y)
            
            # loss = loss_func(out,batch_z)
            loss = loss_func(out,batch_z,train_table.row_num)
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print ("epoch ",epoch," : ")

        out=model(train_x,train_correlation)
        for i in range(len(train_prob_correct)):
            if train_prob_correct[i]!=None:
                out[i]=train_prob_correct[i]
        for i in range(len(out)):
            min_prob=min(train_probs[i])
            if out[i]>min_prob:
                out[i]=min_prob
            if out[i]<(1/train_table.row_num):
                out[i]=0
        Q(out,train_y,train_table.row_num)

        out=model(test_x,test_correlation)
        for i in range(len(test_prob_correct)):
            if test_prob_correct[i]!=None:
                out[i]=test_prob_correct[i]
        for i in range(len(out)):
            min_prob=min(test_probs[i])
            if out[i]>min_prob:
                out[i]=min_prob
            if out[i]<(1/test_table.row_num):
                out[i]=0
        Q(out,test_y,test_table.row_num)
    print(train_table.row_num)
    print(test_table.row_num)




