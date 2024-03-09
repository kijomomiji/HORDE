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

import pandas as pd
import csv
from collections import Counter
import bisect
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils import data
import copy

random.seed(1) 
np.random.seed(1)
torch.manual_seed(1)    
torch.cuda.manual_seed_all(1)


class s_node():
    def __init__(self,value,nextone):
        self.value=value
        self.nextone=nextone

def search(nums,target):
    left=0
    right=len(nums)-1
    if right<0:
        return 0
    if target>nums[right].value:
        return right+1
    while left<right:
        middle=int((right+left)/2)
        if nums[middle].value<target:
            left=middle+1
        else:
            right=middle
    return left

def fill_one(structure,table_name,attr_order,primary_pos,one_tuple):
    # print(structure)
    # node=structure[one_tuple[primary_pos]][table_name]
    node=structure[table_name]
    for i in attr_order[:len(attr_order)-1]:
        value=one_tuple[i]
        th=search(node,value)
        if th<len(node) and node[th].value==value:
            node=node[th].nextone
        else:
            new_node=s_node(value,[])
            node.insert(th,new_node)
            node=node[th].nextone
    value=one_tuple[attr_order[-1]]
    th=search(node,value)
    if th<len(node) and node[th].value==value:
        # node[th].nextone+=1
        if one_tuple[primary_pos] in node[th].nextone:
            node[th].nextone[one_tuple[primary_pos]]+=1
        else:
            node[th].nextone[one_tuple[primary_pos]]=1    
    else:
        new_node=s_node(value,{one_tuple[primary_pos]:1})
        node.insert(th,new_node)

def infer_B(restriction_vecs,node_list,current_layer,max_layer):
    global p_dict
    if current_layer>=max_layer:
        # p_dict+=Counter(node_list)
        p_dict.update(node_list)
        # print(len(node_list))
        return
    if len(node_list)==0:
        return
    restriction_symbol=list(restriction_vecs[current_layer][:3])
    restriction_value=list(restriction_vecs[current_layer][3:])
    if restriction_symbol==[0,0,0]:
        for one_node in node_list:
            infer_B(restriction_vecs,one_node.nextone,current_layer+1,max_layer)
    elif restriction_symbol==[1,0,0]:
        find_th=search(node_list,restriction_value[0])
        if find_th>=len(node_list):
            return
        elif node_list[find_th].value!=restriction_value[0]:
            return
        infer_B(restriction_vecs,node_list[find_th].nextone,current_layer+1,max_layer)
    
    elif restriction_symbol==[0,1,0]:
        find_th=search(node_list,restriction_value[0])
        if find_th>=len(node_list):
            return
        if node_list[find_th].value==restriction_value[0]:
            find_th+=1
        for one_node in node_list[find_th:]:
            infer_B(restriction_vecs,one_node.nextone,current_layer+1,max_layer)
    elif restriction_symbol==[0,0,1]:
        find_th=search(node_list,restriction_value[1])
        for one_node in node_list[:find_th]:
            infer_B(restriction_vecs,one_node.nextone,current_layer+1,max_layer)
    elif restriction_symbol==[0,1,1]:
        start_th=search(node_list,restriction_value[0])
        end_th=search(node_list,restriction_value[1])
        if start_th>=len(node_list):
            return
        if node_list[start_th].value==restriction_value[0]:
            start_th+=1 
        for one_node in node_list[start_th:end_th]:
            infer_B(restriction_vecs,one_node.nextone,current_layer+1,max_layer)
    else:
        print("something wrong must've happened")

def multi_fill_B():
    table_data_dict={}
    #load table data and get data information of six tables
    # ci usecols=[1,2,6]
    # ci.columns=['person_id','movie_id','role_id']
    ci = pd.read_pickle('./data/imdb/cast_info.pickle')

    mk = pd.read_csv('./data/imdb/movie_keyword.csv',header=None,usecols=[1,2])
    mk.columns=["movie_id",'keyword_id']
    mk=mk.astype(int)
    
    mc = pd.read_csv('./data/imdb/movie_companies.csv',header=None,usecols=[1,2,3])
    mc.columns=['movie_id','company_id','company_type_id']
    mc=mc.astype(int)

    mi = pd.read_csv('./data/imdb/movie_info.csv',header=None,usecols=[1,2])
    mi.columns=['movie_id',"info_type_id"]
    mi=mi.astype(int)

    mi_idx = pd.read_csv('./data/imdb/movie_info_idx.csv',header=None,usecols=[1,2])
    mi_idx.columns=['movie_id','info_type_id']
    mi_idx=mi_idx.astype(int)

    # t usecols=[0,3,4]
    t = pd.read_pickle('./data/imdb/title.pickle')
    t.columns=["id","kind_id","production_year"]
    t=t.astype(int)
    
    table_data_dict['ci']=ci
    table_data_dict['mk']=mk
    table_data_dict['mc']=mc
    table_data_dict['mi']=mi
    table_data_dict['mi_idx']=mi_idx
    table_data_dict['t']=t

    # for i in table_data_dict.keys():
    #     print(i,len(table_data_dict[i]))
    # return
    
    attr_dict=pd.read_pickle('./data/imdb/data_info/attr_dict.pkl')

    # get all possible primary key values and initiate the structure
    primary_key_value=[]
    for i in table_data_dict.keys():
        # primary_key_counter[i]={}
        
        if i=='t':
            primary_key_value+=list(set(table_data_dict[i].loc[:,'id']))
        else:
            primary_key_value+=list(set(table_data_dict[i].loc[:,'movie_id']))
    primary_key_value=list(set(primary_key_value))
    print(len(primary_key_value))
    print(primary_key_value[:5])

    
    s={'ci':[],'mk':[],'mc':[],'mi':[],'mi_idx':[],'t':[]}
    structure={'ci':[],'mk':[],'mc':[],'mi':[],'mi_idx':[],'t':[]}
    
    # attr_order_dict={}
    time1=time.time()
    attr_order_dict={}
    for table_name in table_data_dict.keys():
        table=table_data_dict[table_name]

        primary_pos=None
        for th in range(len(table.columns)):
            if table.columns[th]=='id' or table.columns[th]=='movie_id':
                primary_pos=th

        attr_order=sorted(range(len(table.columns)),key=lambda k:len(set(table.loc[:,table.columns[k]])))
        print(table_name)
        
        attr_order_dict[table_name]=attr_order
        
        # print(primary_pos,table.columns[primary_pos])
        attr_order.remove(primary_pos)

        print([attr_dict[table_name][table.columns[i]] for i in attr_order])
        attr_order_dict[table_name]=[attr_dict[table_name][table.columns[i]] for i in attr_order]
        print(attr_order_dict)
    
        print(table_name,len(table))
        for i in table.columns:
            print(i,len(set(table.loc[:,i])))
        print("------------------")
    
        for i in range(len(table.index)):
            if i%1000000==0:
                print(i+1,"/",len(table.index),"has spent",time.time()-time1,'seconds')
            one_tuple=list(table.loc[table.index[i],:])
            # print(table_name,attr_order,primary_pos,one_tuple,table_data_dict[table_name].columns)
            fill_one(structure,table_name,attr_order,primary_pos,one_tuple) 

    # with open('./data/imdb/filled_structure/attr_order_dict.pkl', 'wb') as file:
    #         pickle.dump(attr_order_dict,file)
    #         print('stored in ./data/imdb/filled_structure/attr_order_dict.pkl')


    # print("has spent",time.time()-time1,'seconds')
    # with open('./data/imdb/filled_structure/multi_table_B.pkl', 'wb') as file:
    #     pickle.dump(structure,file)
    #     print('stored in ./data/imdb/filled_structure/multi_table_B.pkl')

    return

def multi_ac_B():
    start_pos=int(input("input the start pos\n"))
    global_time=time.time()    
    # workload_name=['job-light','scale','synthetic','train']
    workload_name=['train']
    predicate_vec={}
    table_vec={}
    table_vec_order=['ci', 'mc', 'mi', 'mi_idx', 'mk', 't']
    label_vec={}

    cards={}
    for workload in workload_name:
        # predicate_vec[workload]=pd.read_pickle('./data/imdb/workload_vec/int_predicate-'+workload+'.pkl')
        predicate_vec[workload]=pd.read_pickle('./data/imdb/workload_vec/predicate-'+workload+'.pkl')
        table_vec[workload]=pd.read_pickle('./data/imdb/workload_vec/table-'+workload+'.pkl')
        label_vec[workload]=pd.read_pickle('./data/imdb/workload_vec/label-'+workload+'.pkl')
        cards[workload]=pd.read_pickle('./data/imdb/workload_vec/card-'+workload+'.pkl')
    
        # true_one=0
        # for i in range(len(cards[workload])):
        #     upper_bound,_=torch.min(predicate_vec[workload][i][:,:,-1],axis=1)
        #     if upper_bound*cards[workload][i]>=label_vec[workload][i]:
        #         true_one+=1
        # print(true_one,"/",len(cards[workload]))
    
    # loc_to_attr_dict=pd.read_pickle('./data/imdb/data_info/loc_to_attr_dict.pkl')
    # attr_dict=pd.read_pickle('./data/imdb/data_info/attr_dict.pkl')
    attr_order_dict=pd.read_pickle('./data/imdb/filled_structure/attr_order_dict.pkl')
    structure=pd.read_pickle('./data/imdb/filled_structure/multi_table_B.pkl')
    primary_key_counter=pd.read_pickle('./data/imdb/data_info/primary_key_counter.pkl')
    
    table_names=['mi_idx','mk','mc','ci','t','mi'] # in the order of primary key value range
    # sub_table=['ci','mk','mc','mi','mi_idx']
    # print(attr_order_dict)
    
    root_node=structure['mk']

    global p_dict
    all_count_dict={}
    # time1=time.time()
    for table_name in table_names:
        attr_num=len(attr_order_dict[table_name])
        p_dict=Counter({})
        infer_B([[0.0, 0.0, 0.0, 0.0, 0.0] for i in range(attr_num)],structure[table_name],0,attr_num)
        all_count_dict[table_name]=copy.deepcopy(p_dict)

        print(sum(all_count_dict[table_name].values()))
    
    global counts
    for workload in workload_name:
        modified_label=[]
        inference_time=[]
        # for one_th in range(len(predicate_vec[workload]))[start_pos:start_pos+20000]:
        for one_th in range(start_pos,start_pos+20000):
            if (one_th)%1000==0:
                print(one_th,'/',"("+str(start_pos)+","+str(start_pos+20000)+")","time since start:",time.time()-global_time)
            time1=time.time()
            one=predicate_vec[workload][one_th]
            table=table_vec[workload][one_th][0][0]
            primary_key_values=structure.keys()
            count_list={i:1 for i in primary_key_values}
            result_dict_list=[]
            for table_name in table_names:
                count_one_table_list={}
                restriction_vecs=[list(one[i][-6:-1]) for i in attr_order_dict[table_name]]
                
                # print(table_name,table[table_vec_order.index(table_name)])
                # print(restriction_vecs)
                if table[table_vec_order.index(table_name)]==0:
                    result_dict_list.append(None)
                    continue
                if all([list(i)==[0,0,0,0,0] for i in restriction_vecs])==False:
                    p_dict=Counter({})
                    infer_B(restriction_vecs,structure[table_name],0,len(restriction_vecs))
                    result_dict_list.append(copy.deepcopy(p_dict))
                else:
                    result_dict_list.append(all_count_dict[table_name])
            
            # print(len(result_dict_list))
            # print([len(i) if i!=None else None for i in result_dict_list])


            exist_result_dict_list=[]
            for i in range(len(result_dict_list)):
                if result_dict_list[i]!=None:
                    exist_result_dict_list.append(result_dict_list[i])

            min_index=0
            for i in range(len(exist_result_dict_list)):
                if len(exist_result_dict_list[i])<len(exist_result_dict_list[min_index]):
                    min_index=i
            
            # print([len(i) if i!=None else None for i in exist_result_dict_list])
            # print(len(exist_result_dict_list[min_index]))

            result_dict=exist_result_dict_list[min_index]
            for i in range(len(exist_result_dict_list)):
                if i==min_index:
                    continue
                else:
                    new_dict={}
                    a=exist_result_dict_list[i]
                    for key in result_dict.keys():
                        if key in a.keys():
                            new_dict[key]=result_dict[key]*a[key]
                    result_dict=new_dict
            # print(sum(result_dict.values()),label_vec[workload][one_th],'---')

            # print(one_th,time.time()-time1)     

            # print(workload,w,"result is:",sum(count_list.values()),label_vec[workload][w],"has spent",time.time()-time1)
            # print(workload,one_th,"result is:",sum(count_list.values()),label_vec[workload][one_th],"has spent",time.time()-time1)
            # print("--------------------------")
            inference_time.append(time.time()-time1)
            modified_label.append(sum(result_dict.values()))
        
         

        modified_label=torch.FloatTensor(np.array(modified_label))
        result_addr="./lecarb/estimator/mine/multi_inference_result/"+workload+"-"+str(start_pos)+".pkl"
        with open(result_addr, 'wb') as file:
            pickle.dump([modified_label,inference_time],file)
            print('stored in '+result_addr)


    print(time.time()-global_time)