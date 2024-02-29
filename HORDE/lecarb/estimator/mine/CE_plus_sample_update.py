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
from random import sample
import random
import os
random.seed(1) 
np.random.seed(1)

class Args:
    def __init__(self, **kwargs):
        self.bs = 1024
        self.lr = 0.001
        self.epochs = 200
        self.num_samples = 1000
        self.hid_units = 256
        self.train_num = 100000

        # overwrite parameters from user
        self.__dict__.update(kwargs)

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
    def __init__(self,value,child_card_sum,next1):
        self.value=value
        self.child_card_sum=child_card_sum
        self.next1=next1
        
    def add_new_data(self,data_vec,layer_number,attr_clusters,max_layer):
        if layer_number>=max_layer:
            return
       
        new_tuple=tuple([data_vec[i[0]] for i in attr_clusters])
        p=self 
        for tuple_number in new_tuple:
            if p.next1==None:
                new_one=tuple_index(tuple_number,1,None)
                p.next1=[new_one]
                p=p.next1[0]
            else:
                find_pos=search(p.next1,tuple_number)
                # have not found the index
                if find_pos==len(p.next1) or p.next1[find_pos].value!=tuple_number:
                    new_one=tuple_index(tuple_number,1,None)
                    p.next1.insert(find_pos,new_one)
                    p=new_one
                # have found the index
                else:
                    p=p.next1[find_pos]
                    p.child_card_sum+=1
# class tuple_index:
#     def __init__(self,value,next1):
#         self.value=value
#         self.next1=next1

# class cardinality_estimation_structure:
#     def __init__(self,cardinality,value_tuple,next1):
#         self.cardinality=cardinality   
#         self.value_tuple=value_tuple
#         self.next1=next1
        
#     def add_new_data(self,data_vec,layer_number,attr_clusters,max_layer):
#         if layer_number>=max_layer:
#             return
#         new_tuple=tuple([data_vec[i] for i in attr_clusters[layer_number]])
        
#         p=self 
#         for tuple_number in new_tuple:
#             if p.next1==None:
#                 new_one=tuple_index(tuple_number,None)
#                 p.next1=[new_one]
#                 p=new_one
#             else:
#                 find_pos=search(p.next1,tuple_number)
#                 # have not found the index
#                 if find_pos==len(p.next1) or p.next1[find_pos].value!=tuple_number:
#                     new_one=tuple_index(tuple_number,None)
#                     p.next1.insert(find_pos,new_one)
#                     p=new_one
#                 # have found the index
#                 else:
#                     p=p.next1[find_pos]
                    
#         if p.next1==None:
#             p.next1=cardinality_estimation_structure(1,new_tuple,None)
#             p.next1.add_new_data(data_vec,layer_number+1,attr_clusters,max_layer)
#         else:
#             p.next1.cardinality+=1
#             p.next1.add_new_data(data_vec,layer_number+1,attr_clusters,max_layer)

# def cal(tuple_index_list,screen_list,layer_number,max_layer,c_layer_number,c_max_layer):  
#     #c_max_layer=len(screen_list)
#     if c_layer_number>=c_max_layer:
#         return
#     max_layer=len(screen_list[c_layer_number])
#     #print(layer_number,max_layer)
#     if layer_number>=max_layer:
#         global prob_list
#         prob_list[c_layer_number]+=tuple_index_list.cardinality
#         cal(tuple_index_list.next1,screen_list,0,max_layer,c_layer_number+1,c_max_layer)
#         return
#     #screen form: A B (C)
#     screen=screen_list[c_layer_number][layer_number]
    
#     A=screen[0] 
#     if A==0:
#         for i in tuple_index_list:
#             cal(i.next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
#         return
#     B=screen[1]
#     if A==1:
#         find_pos=search(tuple_index_list,B)
#         if find_pos==len(tuple_index_list) or tuple_index_list[find_pos].value!=B:
#             return
#         else:
#             cal(tuple_index_list[find_pos].next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
#             return
#     elif A==2:
#         find_pos=search(tuple_index_list,B)
#         for i in tuple_index_list[find_pos:]:
#             cal(i.next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
#         return
#     elif A==3:
#         find_pos=search(tuple_index_list,B)
#         if find_pos!=len(tuple_index_list) and tuple_index_list[find_pos].value==B:
#             find_pos+=1
#         for i in tuple_index_list[:find_pos]:
#             cal(i.next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
#         return
#     elif A==4:
#         find_pos_1=search(tuple_index_list,B)
#         find_pos_2=search(tuple_index_list,screen[2])
#         if find_pos_2!=len(tuple_index_list) and tuple_index_list[find_pos_2].value==screen[2]:
#             find_pos_2+=1
#         if find_pos_2<find_pos_1:
#             print("there is something wrong about the interval")
#             return
#         for i in tuple_index_list[find_pos_1:find_pos_2]:
#             cal(i.next1,screen_list,layer_number+1,max_layer,c_layer_number,c_max_layer)
#         return
#     else:
#         print("something wrong must've happened"," screen[th][0]==",A)
#         return      

def cal(root_node,screen_list,c_layer,max_layer):
    if c_layer>=max_layer:
        global prob_list
        prob_list+=root_node.child_card_sum
        return
    screen=screen_list[c_layer][0]
    # print("screen:",screen)
    
    root_node=root_node.next1
    A=screen[0] 
    if A==0:
        # print("wrong,not start from none-empty attribute")
        for i in root_node:
            cal(i,screen_list,c_layer+1,max_layer)
        return
    
    B=screen[1]
    if A==1:
        find_pos=search(root_node,B)
        if find_pos==len(root_node) or root_node[find_pos].value!=B:
            return
        else:
            cal(root_node[find_pos],screen_list,c_layer+1,max_layer)
            return
    elif A==2:
        find_pos=search(root_node,B)
        for i in root_node[find_pos:]:
            cal(i,screen_list,c_layer+1,max_layer)
        return
    elif A==3:
        find_pos=search(root_node,B)
        if find_pos!=len(root_node) and root_node[find_pos].value==B:
            find_pos+=1
        for i in root_node[:find_pos]:
            cal(i,screen_list,c_layer+1,max_layer)
        return
    elif A==4:
        find_pos_1=search(root_node,B)
        find_pos_2=search(root_node,screen[2])
        if find_pos_2!=len(root_node) and root_node[find_pos_2].value==screen[2]:
            find_pos_2+=1
        if find_pos_2<find_pos_1:
            print("there is something wrong about the interval")
            return
        for i in root_node[find_pos_1:find_pos_2]:
            cal(i,screen_list,c_layer+1,max_layer)
        return
    else:
        print("something wrong must've happened"," screen[th][0]==",A)
        return

def fill_the_structure(root=None,new_data=None,attr_clusters=None):
    max_layer=len(attr_clusters)
    if root==None:
        print("there is no old-version of ACCT")
        return
        # root=cardinality_estimation_structure(0,None,None)
    th=0
    for d in new_data:
        root.add_new_data(d,0,attr_clusters,max_layer)
    return root


def evaluate_errors(errors):
    metrics = {
        'max': np.max(errors),
        '99th': np.percentile(errors, 99),
        '95th': np.percentile(errors, 95),
        '90th': np.percentile(errors, 90),
        '75th': np.percentile(errors, 75),
        '50th': np.percentile(errors, 50),
        '25th': np.percentile(errors, 25),
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
        return [(screen_value-equal_histogram[0][0])/(equal_histogram[-1][0]-equal_histogram[0][0]),equal_histogram[th][1]/total]
    elif screen_type==2:
        th=bisect.bisect_left(more_histogram,(screen_value,0))
        if th==len(more_histogram):
            return [(screen_value-equal_histogram[0][0])/(equal_histogram[-1][0]-equal_histogram[0][0]),0]
        else:
            return [(screen_value-equal_histogram[0][0])/(equal_histogram[-1][0]-equal_histogram[0][0]),more_histogram[th][1]/total]
    elif screen_type==3:
        th=bisect.bisect_right(less_histogram,(screen_value,0))
        if th==0:
            return [(screen_value-equal_histogram[0][0])/(equal_histogram[-1][0]-equal_histogram[0][0]),0]
        else:
            return [(screen_value-equal_histogram[0][0])/(equal_histogram[-1][0]-equal_histogram[0][0]),less_histogram[th-1][1]/total]
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
    def __init__(self,predicate_num,predicate_length,sample_number,hid_size):
        super(SetConv, self).__init__()
        self.predicate_mlp1 = nn.Linear(predicate_length, hid_size)
        self.predicate_mlp2 = nn.Linear(hid_size, hid_size)
        
        self.sample_mlp1 = nn.Linear(sample_number,sample_number)
        self.sample_mlp2 = nn.Linear(sample_number,hid_size)
        
        self.out_mlp1 =nn.Linear(hid_size*(predicate_num+1),hid_size)
        self.out_mlp2 = nn.Linear(hid_size, 1)

    def forward(self, predicates,samples):
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)

        cat_one=torch.cat((hid_sample, hid_predicate), 1)
        cat_one=torch.flatten(cat_one, start_dim=1)
        cat_one = F.relu(self.out_mlp1(cat_one))
        cat_one = torch.sigmoid(self.out_mlp2(cat_one))
        
        return cat_one

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils import data
def Q(prediction,real):
    q_error=[]
    for i in range(len(prediction)):
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
        return torch.mean(torch.pow(predications-labels, 2)+torch.pow(torch.where(delta>0,delta,0),2))

class query_plus_sample():
    def __init__(self,query_vec_for_classification,sample_inputs):
        self.query_vec_for_classification=query_vec_for_classification
        self.sample_inputs=sample_inputs

def sample_to_vector(query_vec,sample_data):
    sample_value_on_one_attr=[]
    sample_vectors=[]
    for attr in range(len(query_vec[0])):
        sample_value_on_one_attr.append(np.array([i[attr] for i in sample_data]))
    for query in query_vec:
        sample_input=np.ones(len(sample_data), dtype=bool)
        for attr in range(len(query)):
            s=query[attr]
            if s[0]==0:
                continue
            elif s[0]==1:
                operator="="
                value=s[1]
            elif s[0]==2:
                operator=">="
                value=s[1]
            elif s[0]==3:
                operator="<="
                value=s[1]
            elif s[0]==4:
                operator='[]'
                value=[s[1],s[2]]
            sample_input&=OPS[operator](sample_value_on_one_attr[attr],value)
        sample_vectors.append(sample_input.astype(int))
    return sample_vectors

def query_to_vector(table,query_vec,less_histograms,more_histograms,equal_histograms,data_length):
    query_vec_for_classification=[]
    # least_C=[]
    keys=list(table.columns.keys())
    for query in query_vec:
        query_vec_for_classification.append([])
        # least_number=1
        
        for attr_th in range(len(query)):
            if str(table.columns[keys[attr_th]].dtype)=="category":
                if query[attr_th]==[0]:
                    # new_attr_vec=[0 for i in range(len(query)+4)]
                    new_attr_vec=[0 for i in range(len(query))]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([0,0,0,0,1]) 
                else:
                    new_attr_vec=[0 for i in range(len(query))]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([1,0,0])
                    # new_attr_vec.append(query[attr_th][-1])
                    new_attr_vec.extend(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],1,query[attr_th][-1],data_length))
                query_vec_for_classification[-1].append(new_attr_vec)
                # if new_attr_vec[-1]<least_number:
                #     least_number=new_attr_vec[-1]
            else:
                if query[attr_th]==[0]:
                    new_attr_vec=[0 for i in range(len(query)+3)]
                    new_attr_vec1=[0 for i in range(len(query)+3)]
                    new_attr_vec[attr_th]=1
                    new_attr_vec1[attr_th]=1
                    new_attr_vec.extend([0,1])
                    new_attr_vec1.extend([0,1])

                elif query[attr_th][0]==2:
                    new_attr_vec=[0 for i in range(len(query))]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([0,1,0])
                    # new_attr_vec.append(query[attr_th][-1])
                    new_attr_vec.extend(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],2,query[attr_th][-1],data_length))

                    new_attr_vec1=[0 for i in range(len(query)+3)]
                    new_attr_vec1[attr_th]=1
                    new_attr_vec1.extend([0,1])
                    new_attr_vec1[-1]=new_attr_vec[-1]
                elif query[attr_th][0]==3:
                    new_attr_vec=[0 for i in range(len(query)+3)]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([0,1])

                    new_attr_vec1=[0 for i in range(len(query))]
                    new_attr_vec1[attr_th]=1
                    new_attr_vec1.extend([0,0,1])
                    # new_attr_vec1.append(query[attr_th][-1])
                    new_attr_vec1.extend(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],3,query[attr_th][-1],data_length))
                    new_attr_vec[-1]=new_attr_vec1[-1]
                elif query[attr_th][0]==4:
                    new_attr_vec=[0 for i in range(len(query))]
                    new_attr_vec[attr_th]=1
                    new_attr_vec.extend([0,1,0])
                    # new_attr_vec.append(query[attr_th][1])
                    
                    # if query[attr_th][1]/(equal_histograms[attr_th][-1][0]-equal_histograms[attr_th][0][0])>1:
                    #     print(query[attr_th][1],equal_histograms[attr_th][-1][0],equal_histograms[attr_th][0][0],query[attr_th][1]/(equal_histograms[attr_th][-1][0]-equal_histograms[attr_th][0][0]))
                    new_attr_vec.append((query[attr_th][1]-equal_histograms[attr_th][0][0])/(equal_histograms[attr_th][-1][0]-equal_histograms[attr_th][0][0]))
                    new_attr_vec.append(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],4,query[attr_th][1:3],data_length))

                    new_attr_vec1=[0 for i in range(len(query))]
                    new_attr_vec1[attr_th]=1
                    new_attr_vec1.extend([0,0,1])
                    # new_attr_vec1.append(query[attr_th][2])
                    new_attr_vec1.append((query[attr_th][2]-equal_histograms[attr_th][0][0])/(equal_histograms[attr_th][-1][0]-equal_histograms[attr_th][0][0]))
                    new_attr_vec1.append(C_prob(less_histograms[attr_th],more_histograms[attr_th],equal_histograms[attr_th],4,query[attr_th][1:3],data_length))

                query_vec_for_classification[-1].append(new_attr_vec)
                query_vec_for_classification[-1].append(new_attr_vec1)
    return query_vec_for_classification

def evaluations(prediction,p_labels):
    # prediction=np.round(out.cpu().detach().numpy()*len(new_data))
    # p_labels=np.round(test_query_label.cpu().detach().numpy()*len(new_data))
    q_error=Q(prediction,p_labels)  
    print("q_error")
    evaluate_errors(q_error)  
    
    MAE=np.absolute(prediction-p_labels)
    print("MAE")
    evaluate_errors(MAE)

    MAPE=[]
    for i in range(len(p_labels)):
        if p_labels[i]==0:
            MAPE.append(prediction[i])
        else:
            MAPE.append(abs((prediction[i]-p_labels[i])/p_labels[i]))

    MAPE=np.array(MAPE)
    print("MAPE")
    evaluate_errors(MAPE)
    return np.max(q_error)


def sample_check(query,sample):
    for i in range(len(sample)):
        if query[i][0]==1:
            if query[i][1]!=sample[i]:
                return 0
        elif query[i][0]==2:
            if query[i][1]>sample[i]:
                return 0
        elif query[i][0]==3:
            if query[i][1]<sample[i]:
                return 0
        elif query[i][0]==4:
            if sample[i]<query[i][1] or sample[i]>query[i][2]:
                return 0
    return 1


def CE_plus_sample_update(dataset,new_version,params):
    args = Args(**params)
    # print(args.epochs)
    # return

    print_list=[]
    version="original"
    table = load_table(dataset,version)
    new_data=load_data_from_pkl_file(dataset+"_"+version+".pkl").data   #class table_data
    new_data1=load_data_from_pkl_file(dataset+"_"+new_version+".pkl").data
    print("old:",len(new_data)," tuples")
    print("new:",len(new_data1)," tuples")
    print("update:",len(new_data1)-len(new_data)," tuples")
    

    # get tuples which need to be updated
    update_data=new_data1[len(new_data):]
    # print(len(new_data1))

    print_list.append(version+" to "+new_version)
    print_list.append("need to update "+str(len(update_data))+" tuples")
    
    attr_clusters=[[i] for i in range(len(new_data[0]))]
    cluster_ranges=[]
    for cluster in attr_clusters:
        total_num=1
        for i in cluster:
            key = list(table.columns.keys())[i]
            total_num*=(table.columns[key].vocab_size)
        cluster_ranges.append(total_num)
    # sort the attr_clusters so that big range clusters are placed behind small ones 
    attr_clusters = [i for _,i in sorted(zip(cluster_ranges,attr_clusters))]
    print(attr_clusters)
    print(cluster_ranges)
    

    # to check the data distribution
    addr="./lecarb/estimator/mine/histograms_data/"+dataset+"_"+version+".pkl"
    if os.path.exists(addr):
        # the updating process of histograms
        with open(addr, 'rb') as f:
            [less_histograms,more_histograms,equal_histograms,distincts] = pickle.load(f)
        for attr in range(len(attr_clusters)):
            distinct=distincts[attr]
            for i in range(len(update_data)):
                if update_data[i][attr] not in distinct.keys():
                    distinct[update_data[i][attr]]=1
                else:
                    distinct[update_data[i][attr]]+=1
            sorted_tuple=sorted(distinct.items(),reverse=False)

            less_histogram=[]
            more_histogram=[]
            less_count=0
            more_count=len(new_data1)
            
            for i in range(len(sorted_tuple)):
                less_count+=sorted_tuple[i][1]
                less_histogram.append((sorted_tuple[i][0],less_count))
                more_histogram.append((sorted_tuple[i][0],more_count))
                more_count-=sorted_tuple[i][1]
            equal_histogram=sorted_tuple

            less_histograms[attr]=less_histogram
            more_histograms[attr]=more_histogram
            equal_histograms[attr]=equal_histogram
            distincts[attr]=distinct
    else:
        print("histogram error, you have not finished previous experiments")
        return

    # workload_data=load_data_from_pkl_file(dataset+"_"+new_version+"updated_workload.pkl")
    workload_data=load_data_from_pkl_file(dataset+"_"+new_version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label

    # test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+new_version+"updated_workload.pkl")
    test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+new_version+"_workload.pkl")
    test_query_vec=test_workload_data.query_vec
    test_query_label=test_workload_data.query_label

    valid_workload_data=load_data_from_pkl_file("valid_"+dataset+"_"+new_version+"_workload.pkl")
    valid_query_vec=valid_workload_data.query_vec
    valid_query_label=valid_workload_data.query_label
   
    # print(query_label[:5])
    # print(test_query_label[:5])
    # print(valid_query_label[:5])
    # print(len(new_data1))
    # return
    for i in range(len(query_label)):
        query_label[i]/=len(new_data1)

    for i in range(len(test_query_label)):
        test_query_label[i]/=len(new_data1)

    for i in range(len(valid_query_label)):
        valid_query_label[i]/=len(new_data1)
    
    # use reservoir sampling to update the samples
    addr="./lecarb/estimator/mine/samples_data/"+dataset+"_"+version+".pkl"
    if os.path.exists(addr):
        time1=time.time()
        #using reservoir sampling to update
        with open(addr, 'rb') as f:
            [sample_data,sample_inputs,test_sample_inputs,valid_sample_inputs] = pickle.load(f)
        sample_number=len(sample_inputs[0])
       
        change_dict={}
        for i in range(len(update_data)):
            n=random.randint(0,len(new_data)+i)
            if n<sample_number:
                u=random.randint(0,sample_number-1)
                change_dict[u]=i
        print("Using reservoir sampling, there are",len(change_dict),"samples need to be updated.")
        print_list.append("Using reservoir sampling, there are "+str(len(change_dict))+" samples need to be updated.")
        
        for j in range(len(query_vec)):
            for i in change_dict.keys():
                sample_data[i]=update_data[change_dict[i]]
        sample_inputs=sample_to_vector(query_vec,sample_data)
        
        time2=time.time()
        print("has spent",time2-time1,'seconds to update training sample vectors')        
        print_list.append("has spent "+str(time2-time1)+' seconds to update training sample vectors')

        time1=time.time()
        valid_sample_inputs=sample_to_vector(valid_query_vec,sample_data)
        time2=time.time()
        print("has spent",time2-time1,'seconds to update valid sample vectors')        
        print_list.append("has spent "+str(time2-time1)+' seconds to update valid sample vectors')

        time1=time.time()
        test_sample_inputs=sample_to_vector(test_query_vec,sample_data)
        # for j in range(len(test_query_vec)):
        #     for i in change_dict.keys():
        #         test_sample_inputs[j][i]=sample_check(test_query_vec[j],update_data[change_dict[i]])
        time2=time.time()
        print("has spent",time2-time1,'seconds to update testing sample vectors')        
        print_list.append("has spent "+str(time2-time1)+' seconds to update testing sample vectors')
    else:
        print("sample error, you have not finished previous experiments")
        return
    
    # 0:None   ;    1=   ;   2>=   ;  3<=    ;   4[]

    # query_to_vector(table,query_vec,less_histograms,more_histograms,equal_histograms)
    
    # encode queries
    time1=time.time()
    query_vec_for_classification=query_to_vector(table,query_vec,less_histograms,more_histograms,equal_histograms,len(new_data1))
    print("has spent ",time.time()-time1," seconds to encode training queries")
    print_list.append("has spent "+str(time.time()-time1)+" seconds to encode training queries")

    time1=time.time()
    test_query_vec_for_classification=query_to_vector(table,test_query_vec,less_histograms,more_histograms,equal_histograms,len(new_data1))
    print("has spent ",time.time()-time1," seconds to encode testing queries")
    print_list.append("has spent "+str(time.time()-time1)+" seconds to encode testing queries")

    time1=time.time()
    valid_query_vec_for_classification=query_to_vector(table,valid_query_vec,less_histograms,more_histograms,equal_histograms,len(new_data1))
    print("has spent ",time.time()-time1," seconds to encode valid queries")
    print_list.append("has spent "+str(time.time()-time1)+" seconds to encode valid queries")

    query_vec_for_classification=torch.FloatTensor(query_vec_for_classification)
    query_label=torch.FloatTensor(query_label)
    sample_inputs=torch.FloatTensor(sample_inputs)
    test_query_vec_for_classification=torch.FloatTensor(test_query_vec_for_classification)
    test_query_label=torch.FloatTensor(test_query_label)
    test_sample_inputs=torch.FloatTensor(test_sample_inputs)
    valid_query_vec_for_classification=torch.FloatTensor(valid_query_vec_for_classification)
    valid_query_label=torch.FloatTensor(valid_query_label)
    valid_sample_inputs=torch.FloatTensor(valid_sample_inputs)
    
    query_vec_for_classification=torch.unsqueeze(query_vec_for_classification,dim=1)
    query_label=torch.unsqueeze(query_label,dim=1)
    sample_inputs=torch.unsqueeze(sample_inputs,dim=1)
    sample_inputs=torch.unsqueeze(sample_inputs,dim=1)

    test_query_vec_for_classification=torch.unsqueeze(test_query_vec_for_classification,dim=1)
    test_query_label=torch.unsqueeze(test_query_label,dim=1)
    test_sample_inputs=torch.unsqueeze(test_sample_inputs,dim=1)
    test_sample_inputs=torch.unsqueeze(test_sample_inputs,dim=1)

    valid_query_vec_for_classification=torch.unsqueeze(valid_query_vec_for_classification,dim=1)
    valid_query_label=torch.unsqueeze(valid_query_label,dim=1)
    valid_sample_inputs=torch.unsqueeze(valid_sample_inputs,dim=1)
    valid_sample_inputs=torch.unsqueeze(valid_sample_inputs,dim=1)  
 
    train_data = TensorDataset(query_vec_for_classification,sample_inputs, query_label)
    train_loader = data.DataLoader(train_data,batch_size=args.bs,shuffle=False)
    

    model_addr="./lecarb/estimator/mine/trained_model/model_"+dataset+"_"+version+".pkl"
    with open(model_addr, 'rb') as f:
        model = pickle.load(f)
        print("load trained model from",model_addr)
    
    hid_size=model.predicate_mlp1.out_features
    print(query_vec_for_classification.shape[2],query_vec_for_classification.shape[3],sample_number,hid_size)
    
    # model=SetConv(query_vec_for_classification.shape[2],query_vec_for_classification.shape[3],sample_number,hid_size)
    # report_model(model)
    # return

    loss_func=My_loss()
    lr=0.001
    opt = torch.optim.Adam(model.parameters(),lr=lr)

    cuda = False if DEVICE == 'cpu' else True
    if cuda:
        model=model.cuda()
        test_query_vec_for_classification=test_query_vec_for_classification.cuda()
        test_sample_inputs=test_sample_inputs.cuda()
        valid_query_vec_for_classification=valid_query_vec_for_classification.cuda()
        valid_sample_inputs=valid_sample_inputs.cuda()

    #update the tree_based_structure
    print("start to update tree")
    start_time=time.time()
    addr="./lecarb/estimator/mine/filled_tree_based_structure/"+dataset+"_"+version+".pkl"
    with open(addr, 'rb') as f:
        real_root_nodes= pickle.load(f)[0]
    print("has successfully loaded data from "+addr)
    for i in range(len(attr_clusters)):
        real_root_nodes[i]=fill_the_structure(real_root_nodes[i],update_data,attr_clusters[i:])
    end_time=time.time()
    print("has spent "+str(end_time-start_time)+" seconds to update the tree-based structure")
    print_list.append("has spent "+str(end_time-start_time)+" seconds to update the tree-based structure")
    new_addr="./lecarb/estimator/mine/filled_tree_based_structure/"+dataset+"_"+new_version+".pkl"
    with open(new_addr, 'wb') as f:
        pickle.dump([real_root_nodes,end_time-start_time], f)
        print("New ACCT has been stored in "+new_addr)

    best_model=None
    best_mean_MAE=float("inf")
    best_epoch=-1
    start_time=time.time()
 
    for epoch in range(args.epochs):
        if epoch%2==1:
            lr*=0.8
            opt = torch.optim.Adam(model.parameters(),lr=lr)
        for i,(x,y,z) in enumerate(train_loader):
            upper_bound,_=torch.min(x[:,0,:,-1],axis=1)
            upper_bound=torch.FloatTensor(upper_bound)
            upper_bound=torch.unsqueeze(upper_bound,dim=1)
            if cuda:
                x=x.cuda()
                y=y.cuda()
                z=z.cuda()
                upper_bound=upper_bound.cuda()
            x = Variable(x)
            y = Variable(y)
            z = Variable(z)
            upper_bound=Variable(upper_bound)
            out = model(x,y)
            loss = loss_func(out,z,upper_bound)
            opt.zero_grad()  
            loss.backward()
            opt.step()
            
        print("epoch ",epoch,":")
        out=model(valid_query_vec_for_classification,valid_sample_inputs)
        
        prediction=np.round(out.cpu().detach().numpy()*len(new_data1))
        p_labels=np.round(valid_query_label.cpu().detach().numpy()*len(new_data1))

        MAE=np.absolute(prediction-p_labels)
        if np.mean(MAE)<best_mean_MAE:
            best_model=model
            best_mean_MAE=np.mean(MAE)
            best_epoch=epoch
            evaluations(prediction,p_labels)
    print("has spent "+str(time.time()-start_time)+" seconds to retrain model")    
    print_list.append("has spent "+str(time.time()-start_time)+" seconds to retrain model")
    print()

    start_time=time.time()
    turn_to_precise=0
    out=best_model(test_query_vec_for_classification,test_sample_inputs)
    prediction=np.round(out.cpu().detach().numpy()*len(new_data1))
    inference_time=time.time()-start_time
    info="spent "+str(inference_time)+' seconds to test raw'
    print_list.append(info)

    
    p_labels=np.round(test_query_label.cpu().detach().numpy()*len(new_data1))
    evaluations(prediction,p_labels)
    print("-------------------------------------")

    start_time=time.time()
    
    if dataset=="census13":
        eta=0.0011474853376808396 #census13
    elif dataset=="forest10":
        eta=0.0003000836881853197 #forest10
    elif dataset=='power7':
        eta=0.00025089909708382846 #power7
    elif dataset=="dmv11":
        eta=0.00022336986404429996 #dmv11

    thres=len(new_data1)*eta
    for i in range(len(prediction)):
        if prediction[i]<thres:
            turn_to_precise+=1
            query_clusters=[]
            start_pos=0
            flag=True
            for j in attr_clusters:
                for attr_pos in j:
                    if test_query_vec[i][attr_pos]!=[0]:
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
                    if test_query_vec[i][attr_pos]!=[0]:
                        flag=False
                        break
                if flag==False:
                    break
                elif flag==True:
                    end_pos-=1


            for j in attr_clusters[start_pos:end_pos]:
                query_clusters.append([test_query_vec[i][w] for w in j])
            
            max_layer=len(query_clusters)

            global prob_list
            prob_list=0
            # cal(real_root_nodes[start_pos].next1,query_clusters,0,0,0,c_max_layer)
            cal(real_root_nodes[start_pos],query_clusters,0,max_layer)

            prediction[i]=prob_list
            
    end_time=time.time()
    print_list.append("has spent "+str(end_time-start_time+inference_time)+" seconds to test Raw & Tree")
    evaluations(prediction,p_labels)

    print()
    for i in print_list:
        print(i)
    print("eta:",eta)

# get pure leanring model prediction results   
def find_best_eta(dataset,version):
    print_list=[]
    table = load_table(dataset,version)
    new_data=load_data_from_pkl_file(dataset+"_"+version+".pkl").data   #class table_data

    
    attr_clusters=[[i] for i in range(len(new_data[0]))]
    cluster_ranges=[]
    for cluster in attr_clusters:
        total_num=1
        for i in cluster:
            key = list(table.columns.keys())[i]
            total_num*=(table.columns[key].vocab_size)
        cluster_ranges.append(total_num)
    # sort the attr_clusters so that big range clusters are placed behind small ones 
    attr_clusters = [i for _,i in sorted(zip(cluster_ranges,attr_clusters))]
    print(attr_clusters)
    print(cluster_ranges)
    

    # to check the data distribution
    addr="./lecarb/estimator/mine/histograms_data/"+dataset+"_"+version+".pkl"
    if os.path.exists(addr):
        # the updating process of histograms
        with open(addr, 'rb') as f:
            [less_histograms,more_histograms,equal_histograms,distincts] = pickle.load(f)
    else:
        print("histogram error, you have not finished previous experiments")
        return

    # workload_data=load_data_from_pkl_file(dataset+"_"+new_version+"updated_workload.pkl")
    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label

    # test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+new_version+"updated_workload.pkl")
    test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_query_vec=test_workload_data.query_vec
    test_query_label=test_workload_data.query_label

    valid_workload_data=load_data_from_pkl_file("valid_"+dataset+"_"+version+"_workload.pkl")
    valid_query_vec=valid_workload_data.query_vec
    valid_query_label=valid_workload_data.query_label

    for i in range(len(query_label)):
        query_label[i]/=len(new_data)

    for i in range(len(test_query_label)):
        test_query_label[i]/=len(new_data)

    for i in range(len(valid_query_label)):
        valid_query_label[i]/=len(new_data)
    
    # samples
    addr="./lecarb/estimator/mine/samples_data/"+dataset+"_"+version+".pkl"
    if os.path.exists(addr):
        time1=time.time()
        with open(addr, 'rb') as f:
            [sample_data,sample_inputs,test_sample_inputs,valid_sample_inputs] = pickle.load(f)
        sample_number=len(sample_inputs[0])
    else:
        print("sample error, you have not finished previous experiments")
        return
    
    # 0:None   ;    1=   ;   2>=   ;  3<=    ;   4[]

    # query_to_vector(table,query_vec,less_histograms,more_histograms,equal_histograms)
    
    # encode queries
    time1=time.time()
    query_vec_for_classification=query_to_vector(table,query_vec,less_histograms,more_histograms,equal_histograms,len(new_data))
    print("has spent ",time.time()-time1," seconds to encode training queries")
    print_list.append("has spent "+str(time.time()-time1)+" seconds to encode training queries")

    time1=time.time()
    test_query_vec_for_classification=query_to_vector(table,test_query_vec,less_histograms,more_histograms,equal_histograms,len(new_data))
    print("has spent ",time.time()-time1," seconds to encode testing queries")
    print_list.append("has spent "+str(time.time()-time1)+" seconds to encode testing queries")

    time1=time.time()
    valid_query_vec_for_classification=query_to_vector(table,valid_query_vec,less_histograms,more_histograms,equal_histograms,len(new_data))
    print("has spent ",time.time()-time1," seconds to encode valid queries")
    print_list.append("has spent "+str(time.time()-time1)+" seconds to encode valid queries")

    query_vec_for_classification=torch.FloatTensor(query_vec_for_classification)
    query_label=torch.FloatTensor(query_label)
    sample_inputs=torch.FloatTensor(sample_inputs)
    test_query_vec_for_classification=torch.FloatTensor(test_query_vec_for_classification)
    test_query_label=torch.FloatTensor(test_query_label)
    test_sample_inputs=torch.FloatTensor(test_sample_inputs)
    valid_query_vec_for_classification=torch.FloatTensor(valid_query_vec_for_classification)
    valid_query_label=torch.FloatTensor(valid_query_label)
    valid_sample_inputs=torch.FloatTensor(valid_sample_inputs)
    
    query_vec_for_classification=torch.unsqueeze(query_vec_for_classification,dim=1)
    query_label=torch.unsqueeze(query_label,dim=1)
    sample_inputs=torch.unsqueeze(sample_inputs,dim=1)
    sample_inputs=torch.unsqueeze(sample_inputs,dim=1)

    test_query_vec_for_classification=torch.unsqueeze(test_query_vec_for_classification,dim=1)
    test_query_label=torch.unsqueeze(test_query_label,dim=1)
    test_sample_inputs=torch.unsqueeze(test_sample_inputs,dim=1)
    test_sample_inputs=torch.unsqueeze(test_sample_inputs,dim=1)

    valid_query_vec_for_classification=torch.unsqueeze(valid_query_vec_for_classification,dim=1)
    valid_query_label=torch.unsqueeze(valid_query_label,dim=1)
    valid_sample_inputs=torch.unsqueeze(valid_sample_inputs,dim=1)
    valid_sample_inputs=torch.unsqueeze(valid_sample_inputs,dim=1)  
 
    # train_data = TensorDataset(query_vec_for_classification,sample_inputs, query_label)
    # train_loader = data.DataLoader(train_data,batch_size=args.bs,shuffle=False)
    

    model_addr="./lecarb/estimator/mine/trained_model/model_"+dataset+"_"+version+".pkl"
    with open(model_addr, 'rb') as f:
        model = pickle.load(f)
        print("load trained model from",model_addr)
    

    cuda = False if DEVICE == 'cpu' else True
    if cuda:
        model=model.cuda()
        test_query_vec_for_classification=test_query_vec_for_classification.cuda()
        test_sample_inputs=test_sample_inputs.cuda()
        valid_query_vec_for_classification=valid_query_vec_for_classification.cuda()
        valid_sample_inputs=valid_sample_inputs.cuda()

    #tree_based_structure
    addr="./lecarb/estimator/mine/filled_tree_based_structure/"+dataset+"_"+version+".pkl"
    with open(addr, 'rb') as f:
        real_root_nodes= pickle.load(f)[0]
    print("has successfully loaded data from "+addr)
    
    out=model(valid_query_vec_for_classification,valid_sample_inputs)
    addr="./lecarb/estimator/mine/learning_model_prediction/valid_"+dataset+"_"+version+".pkl"
    with open(addr, 'wb') as f:
        pickle.dump([out,valid_query_label.cpu().detach().numpy(),len(new_data)], f)
        print("the vec data has been stored in "+addr)
    
    out=model(test_query_vec_for_classification,test_sample_inputs)
    addr="./lecarb/estimator/mine/learning_model_prediction/test_"+dataset+"_"+version+".pkl"
    with open(addr, 'wb') as f:
        pickle.dump([out,test_query_label.cpu().detach().numpy(),len(new_data)], f)
        print("the vec data has been stored in "+addr)
    
    return

    # start_time=time.time()
    # thres=len(new_data1)*0.001
    # for i in range(len(prediction)):
    #     if prediction[i]<thres:
    #         turn_to_precise+=1
    #         query_clusters=[]
    #         start_pos=0
    #         flag=True
    #         for j in attr_clusters:
    #             for attr_pos in j:
    #                 if test_query_vec[i][attr_pos]!=[0]:
    #                     flag=False
    #                     break
    #             if flag==False:
    #                 break
    #             elif flag==True:
    #                 start_pos+=1

    #         end_pos=len(attr_clusters)
    #         flag=True
    #         for j in range(len(attr_clusters)-1,-1,-1):
    #             for attr_pos in attr_clusters[j]:
    #                 if test_query_vec[i][attr_pos]!=[0]:
    #                     flag=False
    #                     break
    #             if flag==False:
    #                 break
    #             elif flag==True:
    #                 end_pos-=1


    #         for j in attr_clusters[start_pos:end_pos]:
    #             query_clusters.append([test_query_vec[i][w] for w in j])
            
    #         max_layer=len(query_clusters)

    #         global prob_list
    #         prob_list=0
    #         # cal(real_root_nodes[start_pos].next1,query_clusters,0,0,0,c_max_layer)
    #         cal(real_root_nodes[start_pos],query_clusters,0,max_layer)

    #         prediction[i]=prob_list
            
    # end_time=time.time()
    # print_list.append("has spent "+str(end_time-start_time+inference_time)+" seconds to test Raw & Tree")
    # evaluations(prediction,p_labels)

    # print()
    # for i in print_list:
    #     print(i)