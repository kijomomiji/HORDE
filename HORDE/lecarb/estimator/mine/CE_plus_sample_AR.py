import os
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

sys.setrecursionlimit(10000)
random.seed(100) 
np.random.seed(100)

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
        root=tuple_index(None,None,None)
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
        
        self.sample_mlp1 = nn.Linear(sample_number,hid_size)
        self.sample_mlp2 = nn.Linear(hid_size,hid_size)
        
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
        loss=torch.mean(torch.pow(predications-labels, 2)+torch.pow(torch.where(delta>0,delta,0),2))
        return loss
        

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

global max_node_size


class aggregate_R_tree:
    def __init__(self,restriction,nodes,blocks):
        self.restriction=restriction
        self.nodes=nodes
        self.blocks=blocks
    def split(self):
        global new_data
        global pos_index
        global max_node_size
        if len(self.nodes)>max_node_size:
            for node_th in self.nodes:
                one_tuple=new_data[node_th]
                index_value=0
                flags=[]

                for th in range(len(one_tuple)):
                    if one_tuple[th]>self.restriction[th][2]:
                        index_value+=pos_index[th]
                        flags.append(1)
                    else:
                        flags.append(0)
                if index_value in self.blocks.keys():
                    self.blocks[index_value].nodes.append(node_th)
                else:
                    restriction=[(self.restriction[i][0],self.restriction[i][2],(self.restriction[i][0]+self.restriction[i][2])/2) if flags[i]==0 \
                                 else (self.restriction[i][2],self.restriction[i][1],(self.restriction[i][1]+self.restriction[i][2])/2)\
                                  for i in range(len(flags))]
                    self.blocks[index_value]=aggregate_R_tree(restriction,[node_th],{})
            # if len(list(self.blocks.keys()))>10:
            #     print(len(list(self.blocks.keys())))
                # print(self.blocks.keys())
            for key in self.blocks.keys():
                #
                new_restriction=[]
                for i in range(len(self.blocks[key].restriction)):
                    node_value_list=[new_data[node_th][i] for node_th in self.blocks[key].nodes]
                    min_value=min(node_value_list)
                    max_value=max(node_value_list)
                
                    new_restriction.append((min_value,max_value,(min_value+max_value)/2))
                    # self.blocks[key].restriction[i][0]=min(node_value_list)
                    # self.blocks[key].restriction[i][1]=max(node_value_list)
                self.blocks[key].restriction=new_restriction
                #
                self.blocks[key].split()
                
        else:
            return

def cross_confirmation(node_ranges,query_ranges):
    flag=False
    for th in range(len(node_ranges)):
        i=node_ranges[th]
        j=query_ranges[th]
        if i[1]<j[0] or i[0]>j[1]:
            return 0 # disjoint
        elif i[0]>=j[0] and i[1]<=j[1]:
            continue
        else:
            flag=True
    if flag==False:
        return 1 #contained
    else:
        return 2 #crossed

def data_confirmation(one_data,query_range):
    for th in range(len(one_data)):
        d=one_data[th]
        r_tuple=query_range[th]
        if d<r_tuple[0] or d>r_tuple[1]:
            return False
    return True

def AR_infer(AR_tree,test_restriction):
    cross_type=cross_confirmation(AR_tree.restriction,test_restriction)
    global card_count
    global max_node_size
    global new_data
    if cross_type==0:
        return
    elif cross_type==1:
        card_count+=len(AR_tree.nodes)
        return
    elif cross_type==2:
        if len(AR_tree.nodes)>max_node_size:
            for key in AR_tree.blocks.keys():
                AR_infer(AR_tree.blocks[key],test_restriction)
        else:
            for node_th in AR_tree.nodes:
                # print(new_data[node_th],test_restriction)
                data_type=data_confirmation(new_data[node_th],test_restriction)
                if data_type==True:
                    card_count+=1


def AR_tree_inference(dataset,version):
    # print(sys.getrecursionlimit())

    print_list=[]

    sys.setrecursionlimit(10000)
    global max_node_size
    if dataset=='census13':
        max_node_size=64 #census13
    elif dataset=='forest10':
        max_node_size=1000 #forest10
    elif dataset=='power7':
        max_node_size=30000 #power7
    elif dataset=='dmv11':
        max_node_size=10000 #dmv11
    print_list.append("max_node_size:"+str(max_node_size))


    print(sys.getrecursionlimit())
    # return
    # result_addr="./lecarb/estimator/mine/tree_inference_result/"+dataset+"_"+version+".pkl"
    print(dataset,version)
    table = load_table(dataset, version)

    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")#class table_data
    global new_data
    new_data=vec_data.data
    print(len(new_data),"tuples")

    restriction=[]
    min_bound=[]
    max_bound=[]
    for th in range(len(new_data[0])):
        value_range=[d[th] for d in new_data]
        min_value=min(value_range)
        max_value=max(value_range)
        restriction.append((min_value,max_value,(min_value+max_value)/2))
        min_bound.append(min_value)
        max_bound.append(max_value)
    global pos_index
    pos_index=[pow(2,i) for i in range(len(new_data[0]))]
    # print(restriction)
    # print(pos_index)
    # return

    
    
    attr_clusters=[[i] for i in range(len(new_data[0]))]
    cluster_ranges=[]
    for cluster in attr_clusters:
        total_num=1
        for i in cluster:
            key = list(table.columns.keys())[i]
            total_num*=(table.columns[key].vocab_size)
        cluster_ranges.append(total_num)

    
    # inference
    # test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_workload_data=load_data_from_pkl_file("valid_"+dataset+"_"+version+"_workload.pkl")
    test_query_vec=test_workload_data.query_vec
    test_query_label=test_workload_data.query_label


    # vector to restriction range
    test_restrictions=[]
    for one_vec in test_query_vec:
        test_restriction=[]
        for th in range(len(test_query_vec[0])):
            if one_vec[th]==[0]:
                test_restriction.append((min_bound[th],max_bound[th]))
            elif one_vec[th][0]==1:
                test_restriction.append((one_vec[th][1],one_vec[th][1]))
            elif one_vec[th][0]==2:
                test_restriction.append((one_vec[th][1],max_bound[th]))
            elif one_vec[th][0]==3:
                test_restriction.append((min_bound[th],one_vec[th][1]))
            elif one_vec[th][0]==4:
                test_restriction.append((one_vec[th][1],one_vec[th][2]))
        test_restrictions.append(test_restriction)


    time1=time.time()
    AR_tree=aggregate_R_tree(restriction,list(range(len(new_data))),{})
    AR_tree.split()
    print_list.append("has spent "+str(time.time()-time1)+' seconds to fill AR tree')


    total_time=0
    inference_result=[]
    inference_time=[]
    # for restriction_th in range(200):
    for restriction_th in range(len(test_restrictions)):
        test_restriction=test_restrictions[restriction_th]
        global card_count
        card_count=0

        time1=time.time()
        AR_infer(AR_tree,test_restriction)
        spent_time=time.time()-time1

        total_time=total_time+spent_time
        inference_result.append(card_count)
        inference_time.append(spent_time)
        # print(spent_time,card_count,test_query_label[restriction_th])
    result_addr="./lecarb/estimator/mine/tree_inference_result/AR_tree_valid_"+dataset+"_"+version+".pkl"
    with open(result_addr, 'wb') as f:
        pickle.dump([inference_result,inference_time], f)
    print("has stored the result in",result_addr)
    # print(len(inference_result))

    for i in print_list:
        print(i)

    print(total_time)


    return



    inference_result=[]
    inference_time=[]
    for i in range(len(test_query_vec)):
        time1=time.time()
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
        cal(real_root_nodes[start_pos],query_clusters,0,max_layer)
        
        inference_result.append(prob_list)
        inference_time.append(time.time()-time1)
      
    # result_addr="./lecarb/estimator/mine/tree_inference_result/"+dataset+"_"+version+".pkl"
    # with open(result_addr, 'wb') as f:
    #     pickle.dump([inference_result,inference_time], f)
    # print("has stored the result in",result_addr)
    # print(len(inference_result))

def test_for_different_ita(dataset,version):
    ita=0.01
    print_list=[]
    print(dataset,version)
    table = load_table(dataset, version)
    for i in range(table.col_num):
        key=list(table.columns.keys())[i]
        print(i,table.columns[key].name,table.columns[key].vocab_size,table.columns[key].dtype)

    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")#class table_data
    new_data=vec_data.data
    print(len(new_data),"tuples")
        
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
    for cluster in attr_clusters:
        total_num=1
        for i in cluster:
            key = list(table.columns.keys())[i]
            total_num*=(table.columns[key].vocab_size)

    # to check the data distribution
    time1=time.time()
    print("start to get the histograms")
    addr="./lecarb/estimator/mine/histograms_data/"+dataset+"_"+version+".pkl"
    less_histograms=[]
    more_histograms=[]
    equal_histograms=[]
    distincts=[]
    for attr in range(len(attr_clusters)):
        distinct={}
        for i in range(len(new_data)):
            if new_data[i][attr] not in distinct.keys():
                distinct[new_data[i][attr]]=1
            else:
                distinct[new_data[i][attr]]+=1

        sorted_tuple=sorted(distinct.items(),reverse=False)
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
        distincts.append(distinct)
    
    time2=time.time()
    print("has spent ",time2-time1," seconds to get histograms")
    print_list.append("has spent "+str(time2-time1)+" seconds to get histograms")

    # test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"updated_workload.pkl")
    test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_query_vec=test_workload_data.query_vec
    test_query_label=test_workload_data.query_label

    for i in range(len(test_query_label)):
        test_query_label[i]/=len(new_data)

    # get the samples
    
    addr="./lecarb/estimator/mine/samples_data/"+dataset+"_"+version+".pkl"
    if os.path.exists(addr):
        print(addr," already exists")
        with open(addr, 'rb') as f:
            [sample_data,sample_inputs,test_sample_inputs,valid_sample_inputs] = pickle.load(f)
    else:
        print("wrong")
    
    sample_number=len(sample_inputs[0])

    print("start to get samples")
    
    sample_data=sample(new_data,sample_number)

    time1=time.time()
    test_sample_inputs=sample_to_vector(test_query_vec,sample_data)
    time2=time.time()
    print("has spent ",time2-time1," seconds to get testing samples")
    print_list.append("has spent "+str(time2-time1)+" seconds to get testing samples")

    
    # encode queries
   
    time1=time.time()
    test_query_vec_for_classification=query_to_vector(table,test_query_vec,less_histograms,more_histograms,equal_histograms,len(new_data))
    print("has spent ",time.time()-time1," seconds to encode testing queries")
    print_list.append("has spent "+str(time.time()-time1)+" seconds to encode testing queries")


    test_query_vec_for_classification=torch.FloatTensor(test_query_vec_for_classification)
    test_query_label=torch.FloatTensor(test_query_label)
    test_sample_inputs=torch.FloatTensor(test_sample_inputs) 
   

    test_query_vec_for_classification=torch.unsqueeze(test_query_vec_for_classification,dim=1)
    test_query_label=torch.unsqueeze(test_query_label,dim=1)
    test_sample_inputs=torch.unsqueeze(test_sample_inputs,dim=1)
    test_sample_inputs=torch.unsqueeze(test_sample_inputs,dim=1)  



    
        
    addr="./lecarb/estimator/mine/filled_tree_based_structure/"+dataset+"_"+version+".pkl"
    if os.path.exists(addr):
        print(addr," already exists")
        with open(addr, 'rb') as f:
            ACCT_set = pickle.load(f)
        real_root_nodes=ACCT_set[0]
        print("used to spent ",ACCT_set[1],' seconds to fill the ACCT')
        print_list.append("used to spent "+str(ACCT_set[1])+' seconds to fill the ACCT')
    else:
        print("wrong")


    model_addr="./lecarb/estimator/mine/trained_model/model_"+dataset+"_"+version+".pkl"
    if os.path.exists(model_addr):
        print(model_addr," already exists")
        with open(model_addr, 'rb') as f:
            best_model = pickle.load(f)
    else:
        print("wrong")
    
    cuda = False if DEVICE == 'cpu' else True
    if cuda:
        best_model=best_model.cuda()
        test_query_vec_for_classification=test_query_vec_for_classification.cuda()
        test_sample_inputs=test_sample_inputs.cuda()

    #test
    start_time=time.time()
    turn_to_precise=0
    out=best_model(test_query_vec_for_classification,test_sample_inputs)
    prediction=np.round(out.cpu().detach().numpy()*len(new_data))
    inference_time=time.time()-start_time
    print("spent",inference_time,'seconds to test raw')
    print_list.append("spent"+str(inference_time)+'seconds to test raw')

    
    p_labels=np.round(test_query_label.cpu().detach().numpy()*len(new_data))
    evaluations(prediction,p_labels)
    print("------------------------------------")

    start_time=time.time()
    thres=len(new_data)*ita
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
            
            cal(real_root_nodes[start_pos],query_clusters,0,max_layer)
            prediction[i]=prob_list
            
    end_time=time.time()
    info="spent "+str(inference_time+end_time-start_time)+" seconds to test raw & tree"
    print(info)
    print_list.append(info)
    print("turn to precise",turn_to_precise)
    evaluations(prediction,p_labels)

    for i in print_list:
        print(i)
    print("sample_number:",len(test_sample_inputs[0]),"thres:",thres,"ita:",ita)