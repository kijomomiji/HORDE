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

def load_data_from_pkl_file(file_name):
    load_addr="./lecarb/estimator/mine/vec_data/"+file_name
    with open(load_addr, 'rb') as f:
        data = pickle.load(f)
    print("has successfully loaded data from "+load_addr)
    return data

class table_data:
    def __init__(self,data,attr_name,correlation,value_to_int_dict,attr_type_dict):
        self.data=data
        self.attr_name=attr_name
        self.correlation=correlation
        self.value_to_int_dict=value_to_int_dict # value_to_int_dict[key][value] to get the int of str value
        self.attr_type_dict=attr_type_dict  # attr_type_dict[attr] to get the type of this attribute

class workload_data:
    def __init__(self,query_vec,query_label):
        self.query_vec=query_vec
        self.query_label=query_label

prob_list=None

def search(nums,target):
    left=0
    right=len(nums)-1
    if target>nums[right].value:
        return right+1
    while left<right:
        middle=int((right+left)/2)
        if nums[middle].value<target:
            left=middle+1
        else:
            right=middle
    return left

class tuple_index:
    def __init__(self,value,next1):
        self.value=value
        self.next1=next1

class cardinality_estimation_structure:
    def __init__(self,cardinality,value_tuple,next1):
        self.cardinality=cardinality   
        self.value_tuple=value_tuple
        self.next1=next1
        
    def add_new_data(self,data_vec,layer_number,attr_clusters,max_layer):
        if layer_number>=max_layer:
            return
        new_tuple=tuple([data_vec[i] for i in attr_clusters[layer_number]])
        
        p=self 
        for tuple_number in new_tuple:
            if p.next1==None:
                new_one=tuple_index(tuple_number,None)
                p.next1=[new_one]
                p=new_one
            else:
                find_pos=search(p.next1,tuple_number)
                # have not found the index
                if find_pos==len(p.next1) or p.next1[find_pos].value!=tuple_number:
                    new_one=tuple_index(tuple_number,None)
                    p.next1.insert(find_pos,new_one)
                    p=new_one
                # have found the index
                else:
                    p=p.next1[find_pos]
                    
        if p.next1==None:
            p.next1=cardinality_estimation_structure(1,new_tuple,None)
            p.next1.add_new_data(data_vec,layer_number+1,attr_clusters,max_layer)
        else:
            p.next1.cardinality+=1
            p.next1.add_new_data(data_vec,layer_number+1,attr_clusters,max_layer)

def cal(tuple_index_list,screen_list,layer_number,max_layer,c_layer_number,c_max_layer):  
    #c_max_layer=len(screen_list)
    if c_layer_number>=c_max_layer:
        return
    max_layer=len(screen_list[c_layer_number])
    #print(layer_number,max_layer)
    if layer_number>=max_layer:
        global prob_list
        prob_list[c_layer_number]+=tuple_index_list.cardinality
        cal(tuple_index_list.next1,screen_list,0,max_layer,c_layer_number+1,c_max_layer)
        return
    #screen form: A B (C)
    screen=screen_list[c_layer_number][layer_number]
    
    A=screen[0] 
    if A==0:
        for i in tuple_index_list:
            cal(i.next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
        return
    B=screen[1]
    if A==1:
        find_pos=search(tuple_index_list,B)
        if find_pos==len(tuple_index_list) or tuple_index_list[find_pos].value!=B:
            return
        else:
            cal(tuple_index_list[find_pos].next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
            return
    elif A==2:
        find_pos=search(tuple_index_list,B)
        for i in tuple_index_list[find_pos:]:
            cal(i.next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
        return
    elif A==3:
        find_pos=search(tuple_index_list,B)
        if find_pos!=len(tuple_index_list) and tuple_index_list[find_pos].value==B:
            find_pos+=1
        for i in tuple_index_list[:find_pos]:
            cal(i.next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
        return
    elif A==4:
        find_pos_1=search(tuple_index_list,B)
        find_pos_2=search(tuple_index_list,screen[2])
        if find_pos_2!=len(tuple_index_list) and tuple_index_list[find_pos_2].value==screen[2]:
            find_pos_2+=1
        if find_pos_2<find_pos_1:
            print("there is something wrong about the interval")
            return
        for i in tuple_index_list[find_pos_1:find_pos_2]:
            cal(i.next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
        return
    else:
        print("something wrong must've happened"," screen[th][0]==",A)
        return      

def fill_the_structure(root=None,new_data=None,attr_clusters=None):
    max_layer=len(attr_clusters)
    if root==None:
        root=cardinality_estimation_structure(0,None,None)
    th=0
    for d in new_data:
        if (th+1)%100000==0:
            print("has finished "+str(th+1)+"/"+str(len(new_data))+" updating data")
        th+=1
        root.add_new_data(d,0,attr_clusters,max_layer)
    return root


def test_and_calc_p_value(data_quantity,real_root_nodes,root_nodes,attr_clusters,query_vec,query_label):
    probs=[]
    
    start_time=time.time()
    print("start to test on "+str(len(query_label))+" queries")
    # precision=0

    turn_to_precise_one=0
    for i in range(len(query_vec)):
        #print(i,query_vec[i])
        start_time1=time.time()
        query_clusters=[]

        start_pos=0
        flag=True
        for j in attr_clusters:
            for attr_pos in j:
                if query_vec[i][attr_pos]!=[0]:
                    flag=False
                    break
            if flag==False:
                break
            elif flag==True:
                start_pos+=1

        end_pos=len(attr_clusters)
        flag=True
        for j in range(len(attr_clusters)-1,-1,-1):
            for attr_pos in attr_clusters[j]:
                if query_vec[i][attr_pos]!=[0]:
                    flag=False
                    break
            if flag==False:
                break
            elif flag==True:
                end_pos-=1


        for j in attr_clusters[start_pos:end_pos]:
            query_clusters.append([query_vec[i][w] for w in j])
        
        c_max_layer=len(query_clusters)
        # print(c_max_layer)
        # print(i,query_clusters)

        global prob_list
        prob_list=[0 for i in range(c_max_layer)]
        cal(root_nodes[start_pos].next1,query_clusters,0,0,0,c_max_layer)

        # if the approximation of cardinality is relatively small, we choose precise and swift method
        if prob_list[-1]<(data_quantity*0.05):
            # global prob_list
            # print(prob_list[-1],data_quantity*0)
            turn_to_precise_one+=1
            prob_list=[0 for i in range(c_max_layer)]
            cal(real_root_nodes[start_pos].next1,query_clusters,0,0,0,c_max_layer)

        probs.append(prob_list[-1])

    # print("the precision is:",precision/len(query_label))
    end_time=time.time()
    print("spent "+str(end_time-start_time)+" seconds")
    # with open('result.txt', 'a') as f:
    #     f.write("spent "+str(end_time-start_time)+" seconds\n")
    #     f.write("turn to precise one:"+str(turn_to_precise_one)+"\n")
    print("turn to precise one:",turn_to_precise_one)

    return probs


def evaluate_errors(errors):
    metrics = {
        'max': np.max(errors),
        '99th': np.percentile(errors, 99),
        '95th': np.percentile(errors, 95),
        '90th': np.percentile(errors, 90),
        'mean': np.mean(errors),
    }
    print(metrics)

# class table_data:
#     def __init__(self,data,attr_name,correlation,value_to_int_dict,attr_type_dict,range_size):
#         self.data=data
#         self.attr_name=attr_name
#         self.correlation=correlation
#         self.value_to_int_dict=value_to_int_dict # value_to_int_dict[key][value] to get the int of str value
#         self.attr_type_dict=attr_type_dict  # attr_type_dict[attr] to get the type of this attribute
#         self.range_size=range_size

#TODO: input: less_histogram;more_histogram;equal_histogram;screen_type
#      output: the cardinality probability with only this screen
import bisect
def C_prob(less_histogram,more_histogram,equal_histogram,screen_type,screen_value,total):
    #0:None   ;    1=   ;   2>=   ;  3<=    ;   4[]
    if screen_type==0:
        return [-1,1]
    elif screen_type==1:
        th=bisect.bisect(equal_histogram,(screen_value,0))
        return [screen_value/(equal_histogram[-1][0]-equal_histogram[0][0]),equal_histogram[th][1]/total]
    elif screen_type==2:
        th=bisect.bisect_left(more_histogram,(screen_value,0))
        if th==len(more_histogram):
            return [screen_value/(equal_histogram[-1][0]-equal_histogram[0][0]),0]
        else:
            return [screen_value/(equal_histogram[-1][0]-equal_histogram[0][0]),more_histogram[th][1]/total]
    elif screen_type==3:
        th=bisect.bisect_right(less_histogram,(screen_value,0))
        if th==0:
            return [screen_value/(equal_histogram[-1][0]-equal_histogram[0][0]),0]
        else:
            return [screen_value/(equal_histogram[-1][0]-equal_histogram[0][0]),less_histogram[th-1][1]/total]
    elif screen_type==4:
        th1=bisect.bisect_left(more_histogram,(screen_value[0],0))
        if th1==len(more_histogram):
            more_count=0
        else:
            more_count=more_histogram[th1][1]

        th2=bisect.bisect_right(less_histogram,(screen_value[1],0))
        if th2==0:
            less_count=0
        else:
            less_count=less_histogram[th2-1][1]
        return (more_count+less_count-total)/total


import torch
import torch.nn as nn
import torch.nn.functional as F

class SetConv(nn.Module):
    def __init__(self,predicate_num,predicate_length):
        super(SetConv, self).__init__()
        self.predicate_mlp1 = nn.Linear(predicate_length, 128)
        self.predicate_mlp2 = nn.Linear(128, 128)
        self.out_mlp1 =nn.Linear(128*predicate_num,512)
        self.out_mlp2 = nn.Linear(512, 1)

    def forward(self, predicates):
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        hid_predicate=torch.flatten(hid_predicate, start_dim=1)
        # out=F.relu(self.out_mlp1(hid_predicate))
        out=F.relu(self.out_mlp1(hid_predicate))
        out = torch.sigmoid(self.out_mlp2(out))
        return out

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils import data
def Q(prediction,real,data_number):
    q_error=[]
    prediction=np.round(prediction.detach().numpy()*data_number)
    real=np.round(real.detach().numpy()*data_number)

    for i in range(len(prediction)):
        # if prediction[i]<(data_number*0.1):
        #     q_error.append(1)
        # elif prediction[i]==real[i]:
        if prediction[i]==real[i]:
            q_error.append(1)
        elif prediction[i]==0:
            q_error.append(real[i])
        elif real[i]==0:
            q_error.append(prediction[i])
        elif prediction[i]>real[i]:
            q_error.append((prediction[i]/real[i]))
        else:
            q_error.append((real[i]/prediction[i]))
    
    return q_error
    print("Max:",np.max(q_error)," 99th:",np.percentile(q_error,99)," 95th:",np.percentile(q_error,95)," 90th:",np.percentile(q_error,90)," 50th:",np.percentile(q_error,50)," mean:",np.mean(q_error))

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predications, labels, upper_bounds):
        delta=predications-upper_bounds
        return 0.5*torch.mean(torch.pow((predications - labels), 2)+0.5*torch.pow(torch.where(delta>0,delta,0),2))
        # return torch.mean(torch.pow((predications - labels), 2)+torch.where(delta>0,delta,0))
        # return torch.mean(torch.abs(predications - labels)+torch.where(delta>0,delta,0))

def CE_without_sample(dataset,version):
    print(dataset,version)
    table = load_table(dataset, version)
    for i in range(table.col_num):
        key=list(table.columns.keys())[i]
        print(i,table.columns[key].name,table.columns[key].vocab_size,table.columns[key].dtype)

    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")#class table_data
    new_data=vec_data.data
    attr_name=vec_data.attr_name
    correlation=vec_data.correlation
    attr_dict=vec_data.attr_type_dict
    
    attr_clusters=[[i] for i in range(len(new_data[0]))]
    # print(attr_clusters)
    # print(table.row_num)
    cluster_ranges=[]
    for cluster in attr_clusters:
        total_num=1
        for i in cluster:
            key = list(table.columns.keys())[i]
            total_num*=(table.columns[key].vocab_size)
        cluster_ranges.append(total_num)
    # print(attr_clusters)
    print(cluster_ranges)
    # sort the attr_clusters so that big range clusters are placed behind small ones 
    attr_clusters = [i for _,i in sorted(zip(cluster_ranges,attr_clusters))]
    for cluster in attr_clusters:
        total_num=1
        for i in cluster:
            key = list(table.columns.keys())[i]
            total_num*=(table.columns[key].vocab_size)
        # print(total_num)
    # attr_clusters.reverse()
    print(attr_clusters)
    
    print(cluster_ranges)

    # to check the data distribution
    less_histograms=[]
    more_histograms=[]
    equal_histograms=[]
    for attr in range(len(attr_clusters)):
        # attr=attr_clusters[th][0]
        # for attr in attr_clusters:
        # attr=attr[0]
        distinct={}
        for i in range(len(new_data)):
            if new_data[i][attr] not in distinct.keys():
                distinct[new_data[i][attr]]=1
            else:
                distinct[new_data[i][attr]]+=1
        sorted_tuple=sorted(distinct.items(),reverse=False)
        
        
        # print(sorted_tuple)

        less_histogram=[]
        more_histogram=[]
        less_count=0
        more_count=len(new_data)
        
        for i in range(len(sorted_tuple)):
            less_count+=sorted_tuple[i][1]
            less_histogram.append((sorted_tuple[i][0],less_count))
            more_histogram.append((sorted_tuple[i][0],more_count))
            more_count-=sorted_tuple[i][1]
        equal_histogram=sorted_tuple
        less_histograms.append(less_histogram)
        more_histograms.append(more_histogram)
        equal_histograms.append(equal_histogram)
    # for i in equal_histograms:
    #     print(i)
    # return

    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label

    for i in range(len(query_label)):
        query_label[i]/=len(new_data)
    
    # for i in range(table.col_num):
    #     key=list(table.columns.keys())[i]
    #     print(i,table.columns[key].name,table.columns[key].vocab_size,table.columns[key].dtype)  
    
    # 0:None   ;    1=   ;   2>=   ;  3<=    ;   4[]

    query_vec_for_classification=[]
    least_C=[]
    keys=list(table.columns.keys())
    for query in query_vec:
        query_vec_for_classification.append([])
        least_number=1
        
        for attr_th in range(len(query)):
            if str(table.columns[keys[attr_th]].dtype)=="category":
                if query[attr_th]==[0]:
                    # new_attr_vec=[0 for i in range(len(query)+4)]
                    new_attr_vec=[0 for i in range(len(query))]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([0,0,0,-1,1]) 
                else:
                    new_attr_vec=[0 for i in range(len(query))]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([1,0,0])
                    # new_attr_vec.append(query[attr_th][-1])
                    new_attr_vec.extend(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],1,query[attr_th][-1],len(new_data)))
                query_vec_for_classification[-1].append(new_attr_vec)
                if new_attr_vec[-1]<least_number:
                    least_number=new_attr_vec[-1]
            else:
                if query[attr_th]==[0]:
                    new_attr_vec=[0 for i in range(len(query)+3)]
                    new_attr_vec1=[0 for i in range(len(query)+3)]
                    new_attr_vec[attr_th]=1
                    new_attr_vec1[attr_th]=1
                    new_attr_vec.extend([-1,1])
                    new_attr_vec1.extend([-1,1])

                elif query[attr_th][0]==2:
                    new_attr_vec=[0 for i in range(len(query))]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([0,1,0])
                    # new_attr_vec.append(query[attr_th][-1])
                    new_attr_vec.extend(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],2,query[attr_th][-1],len(new_data)))

                    new_attr_vec1=[0 for i in range(len(query)+3)]
                    new_attr_vec1[attr_th]=1
                    new_attr_vec1.extend([-1,1])
                elif query[attr_th][0]==3:
                    new_attr_vec=[0 for i in range(len(query)+3)]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([-1,1])

                    new_attr_vec1=[0 for i in range(len(query))]
                    new_attr_vec1[attr_th]=1
                    new_attr_vec1.extend([0,0,1])
                    # new_attr_vec1.append(query[attr_th][-1])
                    new_attr_vec1.extend(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],3,query[attr_th][-1],len(new_data)))

                elif query[attr_th][0]==4:
                    new_attr_vec=[0 for i in range(len(query))]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([0,1,0])
                    # new_attr_vec.append(query[attr_th][1])
                    
                    new_attr_vec.append(query[attr_th][1]/(equal_histograms[attr_th][-1][0]-equal_histograms[attr_th][0][0]))
                    new_attr_vec.append(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],4,query[attr_th][1:3],len(new_data)))

                    new_attr_vec1=[0 for i in range(len(query))]
                    new_attr_vec1[attr_th]=1
                    new_attr_vec1.extend([0,0,1])
                    # new_attr_vec1.append(query[attr_th][2])
                    new_attr_vec1.append(query[attr_th][2]/(equal_histograms[attr_th][-1][0]-equal_histograms[attr_th][0][0]))
                    new_attr_vec1.append(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],4,query[attr_th][1:3],len(new_data)))

                query_vec_for_classification[-1].append(new_attr_vec)
                query_vec_for_classification[-1].append(new_attr_vec1)
                if new_attr_vec[-1]<least_number:
                    least_number=new_attr_vec[-1]
                if new_attr_vec1[-1]<least_number:
                    least_number=new_attr_vec1[-1]
        least_C.append(least_number)
            #print(attr_th,table.columns[keys[attr_th]].name,table.columns[keys[attr_th]].vocab_size,table.columns[keys[attr_th]].dtype)
        
    query_vec_for_classification=torch.FloatTensor(query_vec_for_classification)
    print("query_vec_for_classification shape",query_vec_for_classification.shape)
    # print(query_vec_for_classification.shape[2])
        
    # print(query_vec_for_classification)

    query_label=torch.FloatTensor(query_label)

    query_vec_for_classification=torch.unsqueeze(query_vec_for_classification,dim=1)
    query_label=torch.unsqueeze(query_label,dim=1)
    # print(query_vec_for_classification.shape)
    # print(query_label.shape)
    # print("-----------------")
    # return

    # print(query_vec_for_classification[1])
    # print(np.min(np.array(query_vec_for_classification[:,0,:,-1]),axis=1))
    # print("--------------------")
    # return


    upper_bound=np.min(np.array(query_vec_for_classification[:,0,:,-1]),axis=1)
    upper_bound=torch.FloatTensor(upper_bound)
    upper_bound=torch.unsqueeze(upper_bound,dim=1)
    # print(upper_bound.shape)
    
    # for i in range(10):
    #     print(query_vec_for_classification[i])
    #     print(upper_bound[i])
    # return

    # for i in range(len(query_vec_for_classification)):
    #     query_label[i][0]*=len(new_data)
    #     upper_bound[i][0]*=len(new_data)
    #     for j in range(len(query_vec_for_classification[i][0])):
    #         query_vec_for_classification[i][0][j][-1]*=len(new_data)
    
   
    train_data = TensorDataset(query_vec_for_classification, query_label)

    train_loader = data.DataLoader(train_data,batch_size=2000,shuffle=False)
    #test_loader = data.DataLoader(test_data,batch_size=100,shuffle=True)

    model=SetConv(query_vec_for_classification.shape[2],query_vec_for_classification.shape[3])
    # loss_func = torch.nn.L1Loss()
    # loss_func=torch.nn.MSELoss()
    loss_func=My_loss()
    lr=0.0001
    data_length=len(new_data)
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(400):
        if epoch%10==9:
            lr*=0.8
            opt = torch.optim.Adam(model.parameters(),lr=lr)
        for i,(x,y) in enumerate(train_loader):
            batch_x = Variable(x)
            batch_y = Variable(y)
            # print(batch_x.size())
            out = model(batch_x)
            
            upper_bound,_=torch.min(batch_x[:,0,:,-1],axis=1)
            # print("upper bound")
            # print(upper_bound.shape)
            # print(type(upper_bound))
            upper_bound=torch.FloatTensor(upper_bound)
            upper_bound=torch.unsqueeze(upper_bound,dim=1)


            
            # print("upper bound",upper_bound)
            # print("batch_y",batch_y)
            # return


            loss = loss_func(out,batch_y,Variable(upper_bound))
            

            # loss=Q(out,batch_y,len(new_data))
            # loss.requires_grad_(True)



            # print(type(loss))
            #print(loss)
        
            opt.zero_grad()  
            loss.backward()
            opt.step()
            
        print("epoch ",epoch,":")
        out=model(query_vec_for_classification)
        # for i in range(len(query_vec_for_classification)):
        #     if out[i]>least_C[i]:
        #         out[i]=least_C[i]
        q_error=Q(out,query_label,len(new_data))
        # upper_bound=np.min(np.array(query_vec_for_classification[:,0,:,-1]),axis=1)
        # upper_bound=torch.FloatTensor(upper_bound)
        # upper_bound=torch.unsqueeze(upper_bound,dim=1)
        # for i in range(len(q_error)):
        #     if q_error[i]>=10000:
        #         print(out[i],query_label[i],query_vec_for_classification[i],upper_bound[i:i+10])
        
        # print(len(q_error))
        # return
        # for th in range(len(q_error)):
        #     if q_error[th]>100000:
        #         print(query_label[th],out[th],q_error[th])
        print("Max:",np.max(q_error)," 99th:",np.percentile(q_error,99)," 95th:",np.percentile(q_error,95)," 90th:",np.percentile(q_error,90)," 50th:",np.percentile(q_error,50)," mean:",np.mean(q_error))
        # return