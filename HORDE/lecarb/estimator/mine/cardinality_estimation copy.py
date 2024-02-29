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
# file_name.pkl
def save_data_to_file(data,file_name):
    save_addr="./lecarb/estimator/mine/vec_data/"+file_name
    with open(save_addr, 'wb') as f:
        pickle.dump(data, f)
    print("the vec data has been stored in "+save_addr)

def load_data_from_pkl_file(file_name):
    load_addr="./lecarb/estimator/mine/vec_data/"+file_name
    with open(load_addr, 'rb') as f:
        data = pickle.load(f)
    print("has successfully loaded data from "+load_addr)
    return data

def save_data(data,file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    print("the vec data has been stored in "+file_name)

def load_data(data,file_location,file_name):
    load_addr=file_location+file_name
    with open(load_addr, 'rb') as f:
        data = pickle.load(f)
    print("has successfully loaded data from "+load_addr)
    return data

L = logging.getLogger(__name__)

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

#------------------------------------------------------------------------
# tree structure building part
increase_threshold_per_time=0.05
divide_threshold=0.2
leaf_nodes=[]

class attr_tree():
    def __init__(self,attr_pos_list,left_child,right_child,is_leaf,frequency_model,divide_threshold):
        self.attr_pos_list=attr_pos_list
        self.left_child=left_child # highly correlated attributes
        self.right_child=right_child # weakly correlated attributes
        self.is_leaf=is_leaf
        self.frequency_model=frequency_model
        self.divide_threshold=divide_threshold

    def divide_attr(self,threshold,correlation,attr_name):
            highly_correlated_pos=[]
            weakly_correlated_pos=[]
            for i in self.attr_pos_list:
                for j in self.attr_pos_list:
                     if i!=j and correlation[attr_name[i]][attr_name[j]]>=self.divide_threshold:
                          if i not in highly_correlated_pos:
                            highly_correlated_pos.append(i)
                          if j not in highly_correlated_pos:
                            highly_correlated_pos.append(j)
            for i in self.attr_pos_list:
                if i not in highly_correlated_pos:
                    weakly_correlated_pos.append(i)
            

            return highly_correlated_pos,weakly_correlated_pos
    
    def build_the_tree(self,correlation,attr_name):
        while True:
            highly_correlated_pos,weakly_correlated_pos=self.divide_attr(self.divide_threshold,correlation,attr_name)
            
            # represents that even with a high threshold,the attributes are highly correlated 
            if self.divide_threshold>=0.8:
                self.left_child=None
                self.right_child=None
                self.is_leaf=True
                self.frequency_model=None
                leaf_nodes.append(self)
                return
            # represents that there is no highly correlated attribute so stop
            if len(highly_correlated_pos)==0:
                self.left_child=None
                self.right_child=None
                self.is_leaf=True
                self.frequency_model=None
                leaf_nodes.append(self)
                return
            # represents that there is no weakly correlated attribute so we should increase the divide_threshold
            if len(weakly_correlated_pos)==0:
                self.divide_threshold+=increase_threshold_per_time
                continue
            # represents that we find two batches of attributes, one for highly-correlated and the other for weakly
            else:
                highly_correlated_pos.sort()
                weakly_correlated_pos.sort()
                self.left_child=attr_tree(highly_correlated_pos,None,None,False,None,self.divide_threshold+increase_threshold_per_time)
                self.right_child=attr_tree(weakly_correlated_pos,None,None,False,None,self.divide_threshold+increase_threshold_per_time)
                self.left_child.build_the_tree(correlation,attr_name)
                self.right_child.build_the_tree(correlation,attr_name)
                return
    def get_the_tree_structure(self,flag=0):
        tree_structure={}
        if flag==0:
            tree_structure["name"]=str(self.attr_pos_list)+"\n"+"root"
        elif flag==1:
            tree_structure["name"]=str(self.attr_pos_list)+"\n"+"highly correlated"
        elif flag==2:
            tree_structure["name"]=str(self.attr_pos_list)+"\n"+"weakly correlated"
        if self.left_child:
            tree_structure["name"]+=str(self.divide_threshold)
            tree_structure["children"]=[]
            tree_structure["children"].append(self.left_child.get_the_tree_structure(1))
            tree_structure["children"].append(self.right_child.get_the_tree_structure(2))
        return tree_structure

    def get_the_attr_list(self,attr_list):
        if self.right_child:
            attr_list.append(self.right_child.attr_pos_list)
            attr_list=self.left_child.get_the_attr_list(attr_list)
        else:
            attr_list.append(self.attr_pos_list)
        return attr_list


def build_the_tree(dataset, version):
    
    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")
    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    
    data=vec_data.data
    attr_name=vec_data.attr_name
    correlation=vec_data.correlation
    value_to_int_dict=vec_data.value_to_int_dict

    query_vec=workload_data.query_vec
    query_label=workload_data.query_label

    #print(attr_name)
    #print(correlation)
    
    leaf_nodes=[]
    new_tree=attr_tree(list(range(len(attr_name))),None,None,False,None,0.1)
    new_tree.build_the_tree(correlation,attr_name)
    print("there are "+str(len(leaf_nodes))+" leaf nodes.")
    new_tree_structure=[new_tree.get_the_tree_structure()]
    #print(new_tree_structure)
    print(new_tree.get_the_attr_list([]))
    save_data_to_file(new_tree_structure,dataset+"_"+version+"_tree_structure.pkl")

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


def test_and_calc_p_value(root_nodes,attr_clusters,query_vec,query_label):
    probs=[]
    
    start_time=time.time()
    print("start to test on "+str(len(query_label))+" queries")
    precision=0

    for i in range(len(query_vec)):
        #print(i,query_vec[i])
        start_time1=time.time()
        
        # if (i+1)%100==0:
        #     print("spent "+str(end_time-start_time)+"s to finish "+str(i+1)+" queries")
        
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

        # end_time1=time.time()
        # print("spent "+str(end_time1-start_time1)+"s to finish "+str(i+1)+" queries")
        # print(prob_list)
        # print("   ------------------------------")



        #root_node.calc_prob(query_clusters,0,max_layer,screen_exist)
        #print(prob_list,"  ",query_label[i])
        if prob_list[-1]==query_label[i]:
            precision+=1
        probs.append(prob_list[-1])

    print("the precision is:",precision/len(query_label))
    end_time=time.time()
    print("spent "+str(end_time-start_time)+" seconds")

    return probs


def CE(dataset,version):
    table = load_table(dataset, version)
    for i in range(table.col_num):
        key=list(table.columns.keys())[i]
        print(i,table.columns[key].name,table.columns[key].vocab_size,table.columns[key].dtype)

    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")

    new_data=vec_data.data
    attr_name=vec_data.attr_name
    correlation=vec_data.correlation
    attr_dict=vec_data.attr_type_dict
    # name=test.attr_name[i]
    #     print(name,test.attr_type_dict[name],attr_count[i])

    new_tree=attr_tree(list(range(len(attr_name))),None,None,False,None,0.1)
    new_tree.build_the_tree(correlation,attr_name)
    
    attr_clusters=new_tree.get_the_attr_list([])
    
    attr_clusters=[[i] for i in range(len(new_data[0]))]
    

    print(attr_clusters)
    print(table.row_num)

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
        print(total_num)
    attr_clusters.reverse()
    print(attr_clusters)

    root_nodes=[]
    start_time=time.time()
    for i in range(len(attr_clusters)):
        root_node=fill_the_structure(None,new_data,attr_clusters[i:])
        root_nodes.append(root_node)


    end_time=time.time()
    print("has spent "+str(end_time-start_time)+" seconds to fill the structure")
    save_data(root_nodes,"./lecarb/estimator/mine/filled_model/"+"root_node_"+dataset+"_"+version+".pkl")

    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label

    #test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_query_vec=test_workload_data.query_vec
    test_query_label=test_workload_data.query_label

    # for testing data
    probs=test_and_calc_p_value(root_nodes,attr_clusters,test_query_vec[:1000],test_query_label[:1000])


    ## for training data
    probs=test_and_calc_p_value(root_nodes,attr_clusters,query_vec[:1000],query_label[:1000])

def CE_estimate(dataset,version):
    table = load_table(dataset, version)
    for i in range(table.col_num):
        key=list(table.columns.keys())[i]
        print(i,table.columns[key].name,table.columns[key].vocab_size,table.columns[key].dtype)

    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")

    new_data=vec_data.data
    attr_name=vec_data.attr_name
    correlation=vec_data.correlation
    attr_dict=vec_data.attr_type_dict
    # name=test.attr_name[i]
    #     print(name,test.attr_type_dict[name],attr_count[i])

    new_tree=attr_tree(list(range(len(attr_name))),None,None,False,None,0.1)
    new_tree.build_the_tree(correlation,attr_name)
    
    attr_clusters=new_tree.get_the_attr_list([])
    
    attr_clusters=[[i] for i in range(len(new_data[0]))]
    

    print(attr_clusters)
    print(table.row_num)

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
        print(total_num)
    # attr_clusters.reverse()
    print(attr_clusters)

    # to check the data distribution
    print(attr_clusters)
    data_distribution={}
    attr_transfer_dict={}
    for attr in attr_clusters:
        attr=attr[0]
        distinct={}
        for i in range(len(new_data)):
            if new_data[i][attr] not in distinct.keys():
                distinct[new_data[i][attr]]=1
            else:
                distinct[new_data[i][attr]]+=1
        
        if attr_dict[attr_name[attr]]!="category":
            print(attr_name[attr],attr_dict[attr_name[attr]],len(distinct))
        else:
            continue
    
        sorted_tuple=sorted(distinct.items(),reverse=False)
        
        sum_list=[]
        sums=0
        new_dict={}
        transfer_dict={}
        threshold=len(new_data)/50
        for i in sorted_tuple:
            if i[1]>=threshold:  
                new_dict[i[0]]=i[1]
                transfer_dict[i[0]]=i[0]
                if sum_list!=[]:
                    total_value=0
                    total_frequency=0
                    for j in sum_list:
                        total_value+=j[0]
                        total_frequency+=j[1]
                    for j in sum_list:
                        transfer_dict[j[0]]=(total_value/len(sum_list))
                    new_dict[(total_value/len(sum_list))]=total_frequency
                sum_list=[]
                sums=0
            else:
                sum_list.append(i)
                sums+=i[1]
                if sums>=threshold:
                    total_value=0
                    total_frequency=0
                    for j in sum_list:
                        total_value+=j[0]
                        total_frequency+=j[1]
                    for j in sum_list:
                        transfer_dict[j[0]]=(total_value/len(sum_list))
                    new_dict[(total_value/len(sum_list))]=total_frequency
                    sum_list=[]
                    sums=0
        if sum_list!=[]:
            total_value=0
            total_frequency=0
            for j in sum_list:
                total_value+=j[0]
                total_frequency+=j[1]
            for j in sum_list:
                transfer_dict[j[0]]=(total_value/len(sum_list))
            new_dict[(total_value/len(sum_list))]=total_frequency
            sum_list=[]
            sums=0
        print("new_dict:",len(new_dict))
        # print(new_dict)
        print(sum(new_dict.values()),len(new_data))

        # print(sorted(distinct.items(),reverse=False))

        # save the distribution data
        data_distribution[attr_name[attr]+"\n"+attr_dict[attr_name[attr]]]=sorted(new_dict.items(),reverse=False)
        attr_transfer_dict[attr_name[attr]]=transfer_dict

    bucket_data=[]
    # print(attr_dict.values())
    for data in new_data:
        bucket=[]
        for i in range(len(data)):
            if attr_dict[attr_name[i]]=="category":
                bucket.append(data[i])
            else:
                bucket.append(attr_transfer_dict[attr_name[i]][data[i]])
        #print(data)
        # print(bucket)
        bucket_data.append(bucket)
    
    root_nodes=[]
    start_time=time.time()
    for i in range(len(attr_clusters)):
        root_node=fill_the_structure(None,bucket_data,attr_clusters[i:])
        root_nodes.append(root_node)
    end_time=time.time()
    print("has spent "+str(end_time-start_time)+" seconds to fill the structure")
    
    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label

    #test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_query_vec=test_workload_data.query_vec
    test_query_label=test_workload_data.query_label

    # for testing data
    probs=test_and_calc_p_value(root_nodes,attr_clusters,test_query_vec[:1000],test_query_label[:1000])
    q_error_list=[]
    for i in range(len(probs)):
        if probs[i]==0 and test_query_label[i]==0:
            q_error_list.append(1)
        elif probs[i]==0:
            q_error_list.append(test_query_label[i])
        elif test_query_label[i]==0:
            q_error_list.append(probs[i])
        elif probs[i]>=test_query_label[i]:
            q_error_list.append(probs[i]/test_query_label[i])
        else:
            q_error_list.append(test_query_label[i]/probs[i])
    for i in range(len(probs)):
        with open('result.txt', 'w') as f:
            f.write(str(probs[i])+" "+str(test_query_label[i])+" "+str(q_error_list[i])+"\n")
    print(np.max(q_error_list),np.percentile(q_error_list,99),np.percentile(q_error_list,95),np.percentile(q_error_list,90))
    # for i in range(10):
    #     print(probs[i],test_query_label[i])

    ## for training data
    probs=test_and_calc_p_value(root_nodes,attr_clusters,query_vec[:1000],query_label[:1000])
    q_error_list=[]
    for i in range(len(probs)):
        if probs[i]==0 and query_label[i]==0:
            q_error_list.append(1)
        elif probs[i]==0:
            q_error_list.append(query_label[i])
        elif query_label[i]==0:
            q_error_list.append(probs[i])
        elif probs[i]>=query_label[i]:
            q_error_list.append(probs[i]/query_label[i])
        else:
            q_error_list.append(query_label[i]/probs[i])
    for i in range(len(probs)):
        with open('result.txt', 'a') as f:
            f.write(str(probs[i])+" "+str(query_label[i])+" "+str(q_error_list[i])+"\n")
    print(np.max(q_error_list),np.percentile(q_error_list,99),np.percentile(q_error_list,95),np.percentile(q_error_list,90))
    
    return
    
    save_addr="./lecarb/estimator/mine/distribution_data/"+"new_"+dataset+"_"+version+".pkl"
    with open(save_addr, 'wb') as f:
        pickle.dump(data_distribution, f)
    print("the vec data has been stored in "+save_addr)

    return



import torch
import torch.nn as nn
import torch.nn.functional as F

class SetConv(nn.Module):
    def __init__(self,predicate_feats,hid_units):
        super(SetConv, self).__init__()
        self.predicate_mlp1 = nn.Linear(12, 6)
        self.predicate_mlp2 = nn.Linear(6, 6)
        self.out_mlp1 = nn.Linear(13*6, 1)

    def forward(self, predicates):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate=torch.flatten(hid_predicate, start_dim=1)
        out = torch.sigmoid(self.out_mlp1(hid_predicate))
        # print(out.shape)
        return out

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
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

    # for i in range(len(q_error)):
    #     if q_error[i]>100000:
    #         print(prediction[i],real[i])
    print("Max:",np.max(q_error)," 99th:",np.percentile(q_error,99)," 95th:",np.percentile(q_error,95)," 90th:",np.percentile(q_error,90)," 50th:",np.percentile(q_error,50)," mean:",np.mean(q_error))

def new_model(dataset,version):

    table = load_table(dataset, version)

    test=load_data_from_pkl_file(dataset+"_"+version+".pkl")

    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label

    attr_count=[[0,0,0,0,0] for i in range(len(query_vec[0]))]
    for i in query_vec:
        for j in range(len(i)):
            attr_count[j][i[j][0]]+=1
    
    
    for i in range(len(attr_count)):
        name=test.attr_name[i]
        print(name,test.attr_type_dict[name],attr_count[i])

    print(attr_count)
    labels=list(test.attr_type_dict.keys())
    for i in range(len(labels)):
        labels[i]+=("\n"+test.attr_type_dict[labels[i]])

    print(labels)
    '''
    table = load_table(dataset, version)
    for i in range(table.col_num):
        key=list(table.columns.keys())[i]
        print(i,table.columns[key].name,table.columns[key].vocab_size,table.columns[key].dtype)

    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")

    new_data=vec_data.data
    attr_name=vec_data.attr_name
    correlation=vec_data.correlation 

    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label
    # print(len(new_data))
    print(query_label[:10])
    for i in range(len(query_label)):
        query_label[i]/=len(new_data)
    
    train_x=[]
    for i in range(len(query_vec)):
        input_x=[]
        # print(query_vec[i])
        for screen in query_vec[i]:
            if screen[0]==0:
                input_x.append([0,0,0,0]+[0,0,0,0]+[0,0,0,0])
            elif screen[0]==1:
                input_x.append([1,0,0,screen[1]]+[0,0,0,0]+[0,0,0,0])
            elif screen[0]==2:
                input_x.append([0,0,0,0]+[0,1,0,screen[1]]+[0,0,0,0])
            elif screen[0]==3:
                input_x.append([0,0,0,0]+[0,0,0,0]+[0,0,1,screen[1]])
            elif screen[0]==4:
                input_x.append([0,0,0,0]+[0,1,0,screen[1]]+[0,0,1,screen[2]])
        train_x.append(input_x)
    
    train_x=np.array(train_x)
    train_x=torch.FloatTensor(train_x)
    # train_x=torch.unsqueeze(train_x,dim=1)
    # print(train_x.shape)
    
    query_label=np.array(query_label)
    query_label=torch.FloatTensor(query_label)
    query_label=torch.unsqueeze(query_label,dim=1)
    # print(query_label.shape)
    
    model = SetConv(12,6)
    train_data = TensorDataset(train_x,query_label)
    train_loader = data.DataLoader(train_data,batch_size=100,shuffle=False)

    loss_func = torch.nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(),lr=0.000001)
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
        out=model(train_x)
        # print(out[:10])
        # print(query_label[:10])
        # return
        Q(out,query_label,False)

    return
    for i in range(10):
        print(i)
        print(query_vec[i])
        print(query_label[i])


    #test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_query_vec=test_workload_data.query_vec
    test_query_label=test_workload_data.query_label
    '''
