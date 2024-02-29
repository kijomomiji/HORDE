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


def csv_preprocess():
    ci.to_pickle('./data/imdb/cast_info.pickle')

    a=pd.read_pickle('./data/imdb/cast_info.pickle')
    # print(len(a))
    # print(len(ci))
    # print(ci.loc[[1,2,3],:])
    # print(ci[:5])
   
    # delete rows with wrong information(ci)
    ci = pd.read_csv('./data/imdb/cast_info.csv',header=None,usecols=[1,2,6])
    ci.columns=['person_id','movie_id','role_id']
    print("the table length before deletion")
    print(len(ci))
    wrong_list=[]
    for i in range(len(ci)):
        for j in ci.columns:
            number=ci.loc[i,j]
            try:
                number=int(number)
            except:
                wrong_list.append(i)
                # print(i," ",j," ",number)
    total_set=set(list(range(len(ci))))
    ci=ci.loc[list(total_set-set(wrong_list)),:]
    ci=ci.astype(int)
    print("the table length after deletion")
    print(len(ci))

    ci.to_pickle('./data/imdb/cast_info.pickle')
    print("the table data has been stored in ",'./data/imdb/cast_info.pickle')


    # delete rows with wrong information(t)
    t = pd.read_csv('./data/imdb/title.csv',header=None,usecols=[0,3,4])
    t.columns=["id","kind_id","production_year"]
    print("the table length before deletion")
    print(len(t))
    wrong_list=[]
    for i in range(len(t)):
        for j in t.columns:
            number=t.loc[i,j]
            try:
                number=int(number)
            except:
                wrong_list.append(i)
                # print(i," ",j," ",number)
    total_set=set(list(range(len(t))))
    t=t.loc[list(total_set-set(wrong_list)),:]
    t=t.astype(int)
    print("the table length after deletion")
    print(len(t))

    t.to_pickle('./data/imdb/title.pickle')
    print("the table data has been stored in ",'./data/imdb/title.pickle')

class s_node():
    def __init__(self,value,nextone):
        self.value=value
        self.nextone=nextone
    # def __str__(self):
    #     return "value:"+str(self.value)+" nextone:"+str(self.nextone)
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
    node=structure[one_tuple[primary_pos]][table_name]
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
        node[th].nextone+=1
    else:
        new_node=s_node(value,1)
        node.insert(th,new_node)
    

def infer(restriction_vecs,start_node_list,current_layer,max_layer):
    global counts
    if current_layer>=max_layer:
        counts+=start_node_list
        return
    if len(start_node_list)==0:
        return
    restriction_symbol=list(restriction_vecs[current_layer][:3])
    restriction_value=list(restriction_vecs[current_layer][3:])
    if restriction_symbol==[0,0,0]:
        for one_node in start_node_list:
            infer(restriction_vecs,one_node.nextone,current_layer+1,max_layer)
    elif restriction_symbol==[1,0,0]:
        find_th=search(start_node_list,restriction_value[0])
        if find_th>=len(start_node_list):
            return
        elif start_node_list[find_th].value!=restriction_value[0]:
            return
        infer(restriction_vecs,start_node_list[find_th].nextone,current_layer+1,max_layer)
    
    elif restriction_symbol==[0,1,0]:
        find_th=search(start_node_list,restriction_value[0])
        if find_th>=len(start_node_list):
            return
        if start_node_list[find_th].value==restriction_value[0]:
            find_th+=1
        for one_node in start_node_list[find_th:]:
            infer(restriction_vecs,one_node.nextone,current_layer+1,max_layer)
    elif restriction_symbol==[0,0,1]:
        find_th=search(start_node_list,restriction_value[1])
        for one_node in start_node_list[:find_th]:
            infer(restriction_vecs,one_node.nextone,current_layer+1,max_layer)
    elif restriction_symbol==[0,1,1]:
        start_th=search(start_node_list,restriction_value[0])
        end_th=search(start_node_list,restriction_value[1])
        if start_th>=len(start_node_list):
            return
        if start_node_list[start_th].value==restriction_value[0]:
            start_th+=1 
        for one_node in start_node_list[start_th:end_th]:
            infer(restriction_vecs,one_node.nextone,current_layer+1,max_layer)
    else:
        print("something wrong must've happened")


def multi_ac():
    start_pos=int(input("input the start pos\n"))
        
    # workload_name=['job-light','scale','synthetic','train']
    workload_name=['job-light']
    predicate_vec={}
    table_vec={}
    table_vec_order=['ci', 'mc', 'mi', 'mi_idx', 'mk', 't']
    label_vec={}
    for workload in workload_name:
        predicate_vec[workload]=pd.read_pickle('./data/imdb/workload_vec/int_predicate-'+workload+'.pkl')
        table_vec[workload]=pd.read_pickle('./data/imdb/workload_vec/table-'+workload+'.pkl')
        label_vec[workload]=pd.read_pickle('./data/imdb/workload_vec/label-'+workload+'.pkl')
    
    # label_num=0
    # for key in label_vec.keys():
    #     print(key,len(label_vec[key]))
    #     label_num+=len(label_vec[key])

    # print(label_num)
    # return


    loc_to_attr_dict=pd.read_pickle('./data/imdb/data_info/loc_to_attr_dict.pkl')
    attr_dict=pd.read_pickle('./data/imdb/data_info/attr_dict.pkl')
    attr_order_dict=pd.read_pickle('./data/imdb/filled_structure/attr_order_dict.pkl')
    structure=pd.read_pickle('./data/imdb/filled_structure/multi_table.pkl')
    primary_key_counter=pd.read_pickle('./data/imdb/data_info/primary_key_counter.pkl')

    # global counts
    # counts=0
    # for value in structure.keys():
    #     for name in structure[value].keys():
    #         count_one_test(structure[value][name])
    # print(counts)
    # return


    # print(loc_to_attr_dict)

    # print(predicate_vec['scale'].shape)


    table_names=['mi_idx','mk','mc','ci','t','mi'] # in the order of primary key value range
    # sub_table=['ci','mk','mc','mi','mi_idx']
    # print(attr_order_dict)


    time1=time.time()
    global counts
    for workload in workload_name:
        modified_label=[]
        for one_th in range(len(predicate_vec[workload]))[start_pos:start_pos+10000]:
            one=predicate_vec[workload][one_th]
            table=table_vec[workload][one_th][0][0]
            primary_key_values=structure.keys()
            count_list={i:1 for i in primary_key_values}
            for table_name in table_names:
                count_one_table_list={}
                restriction_vecs=[list(one[i][-6:-1]) for i in attr_order_dict[table_name]]
                if table[table_vec_order.index(table_name)]==1:
                    if all([list(i)==[0,0,0,0,0] for i in restriction_vecs])==False:
                        for primary_key_value in primary_key_values:
                            start_node_list=structure[primary_key_value][table_name]
                            counts=0
                            infer(restriction_vecs,start_node_list,current_layer=0,max_layer=len(restriction_vecs))
                            result=counts*count_list[primary_key_value]
                            if result>0:
                                count_one_table_list[primary_key_value]=result
                        primary_key_values=count_one_table_list.keys()
                        count_list=copy.deepcopy(count_one_table_list)
                    else:
                        for primary_key_value in primary_key_values:
                            if primary_key_value in primary_key_counter[table_name]:
                                count_one_table_list[primary_key_value]=count_list[primary_key_value]*primary_key_counter[table_name][primary_key_value]
                        primary_key_values=count_one_table_list.keys()
                        count_list=copy.deepcopy(count_one_table_list)
                    # print(restriction_vecs)
                    # print(sum(count_list.values()))    
            # print(workload,w,"result is:",sum(count_list.values()),label_vec[workload][w],"has spent",time.time()-time1)
            print(workload,one_th,"result is:",sum(count_list.values()),label_vec[workload][one_th],"has spent",time.time()-time1)
            print(list(count_list.items())[:100])
            print("--------------------------")
            modified_label.append(sum(count_list.values()))
        return
        
        modified_label=torch.FloatTensor(np.array(modified_label))
        modified_label=torch.unsqueeze(modified_label,dim=1)
        # with open('./data/imdb/workload_vec/modified_label-'+str(start_pos)+"-"+workload+'.pkl', 'wb') as file:
        #     pickle.dump(modified_label,file)
        #     print('stored in ./data/imdb/workload_vec/modified_label-'+str(start_pos)+"-"+workload+'.pkl')



def multi_fill():
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

    #s={'ci':{},'mk':{},'mc':{},'mi':{},'mi_idx':{},'t':{}}
    s={'ci':[],'mk':[],'mc':[],'mi':[],'mi_idx':[],'t':[]}
    structure={i:copy.deepcopy(s) for i in primary_key_value}
    print(list(structure.items())[:5])
    
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
    
        for i in range(len(table.index))[:10]:
            if i%1000000==0:
                print(i+1,"/",len(table.index),"has spent",time.time()-time1,'seconds')
            one_tuple=list(table.loc[table.index[i],:])
            print(table_name,attr_order,primary_pos,one_tuple,table_data_dict[table_name].columns)
            fill_one(structure,table_name,attr_order,primary_pos,one_tuple)

        # for i in range(50):
        #     if i%1000000==0:
        #         print(i+1,"/",len(table.index),"has spent",time.time()-time1,'seconds')
        #     tuple=list(table.loc[table.index[i],:])
        #     # print(table_name,tuple)
        #     fill_one(structure,table_name,attr_order,primary_pos,tuple)
        # for i in list(structure.keys())[:5]:
        #     # print(i,structure[i])
        
        
    # global counts
    # counts=0
    
    # for value in structure.keys():
    #     for name in structure[value].keys():
    #         # counts=0
    #         count_one_test(structure[value][name])
    # print(counts)

    # with open('./data/imdb/filled_structure/attr_order_dict.pkl', 'wb') as file:
    #         pickle.dump(attr_order_dict,file)
    #         print('stored in ./data/imdb/filled_structure/attr_order_dict.pkl')


    # print("has spent",time.time()-time1,'seconds')
    # with open('./data/imdb/filled_structure/multi_table.pkl', 'wb') as file:
    #         pickle.dump(structure,file)
    #         print('stored in ./data/imdb/filled_structure/multi_table.pkl')

    return


def reservoir_sampling(total_num,sample_num):
    sample_th_list=list(range(sample_num))
    for i in range(sample_num,total_num):
        n=random.randint(0,sample_num+i)
        if n<sample_num:
            u=random.randint(0,sample_num-1)
            sample_th_list[u]=i
    return sample_th_list


def multi_table_get_sample_vec():
    loc_to_attr_dict=pd.read_pickle('./data/imdb/data_info/loc_to_attr_dict.pkl')
    attr_dict=pd.read_pickle('./data/imdb/data_info/attr_dict.pkl')
    table_data_dict={}
    #load table data and get data information of six tables
    # ci usecols=[1,2,6]
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

    
    workload_name=['job-light','scale','synthetic','train']
    # workload_name=['job-light']
    predicate_vecs=[]

    for i in range(len(workload_name)):
        with open('./data/imdb/workload_vec/predicate-'+workload_name[i]+'.pkl', 'rb') as fo: 
            predicate_vecs.append(pickle.load(fo, encoding='bytes'))

   

    sample_num=1000
    sample_data_dict={}
    sample_index_dict={}
    sample_vec_dict={}
    

    table_name_list=['ci','mc','mi','mi_idx','mk','t']
    # table_name_list=list(table_data_dict.keys())
    # print(table_name_list)
    # return

    for table_name in table_name_list:
        table_data=table_data_dict[table_name]
        sample_index=reservoir_sampling(len(table_data),sample_num)
        sample_index_dict[table_name]=sample_index
        sample_data_dict[table_name]=table_data.iloc[sample_index,:]

    for i in range(len(workload_name)):
        sample_vec_dict[workload_name[i]]=torch.ones(len(predicate_vecs[i]),6,sample_num)
        

    # =:1 0 0; >: 0 1 0; <: 0 0 1
    for i in range(len(workload_name)):
        print("start to get samples for",workload_name[i])
        for th in range(len(predicate_vecs[i])):
            for restriction in predicate_vecs[i][th][0]:
                if restriction[-1]!=1:
                    (table_name,attr_name)=loc_to_attr_dict[list(restriction).index(1)]
                    symbol=list(restriction[9:12])
                    s=sample_data_dict[table_name]
                    table_index=table_name_list.index(table_name)
                    # table_index=attr_dict[table_name][attr_name]
                    if symbol==[1,0,0]:
                        value=restriction[-3]
                        for index_th in range(len(s.index)):
                            if value!=s.loc[s.index[index_th],attr_name]:
                                sample_vec_dict[workload_name[i]][th][table_index][index_th]=0

                    elif symbol==[0,1,0]:
                        value=restriction[-3]
                        for index_th in range(len(s.index)):
                            if value>s.loc[s.index[index_th],attr_name]:
                                sample_vec_dict[workload_name[i]][th][table_index][index_th]=0
                    elif symbol==[0,0,1]:
                        value=restriction[-2]
                        for index_th in range(len(s.index)):
                            if value<s.loc[s.index[index_th],attr_name]:
                                sample_vec_dict[workload_name[i]][th][table_index][index_th]=0
                    elif symbol==[0,1,1]:
                        value1=restriction[-3]
                        value2=restriction[-2]
                        for index_th in range(len(s.index)):
                            if s.loc[s.index[index_th],attr_name]<=value1 or s.loc[s.index[index_th],attr_name]>=value2:
                                sample_vec_dict[workload_name[i]][th][table_index][index_th]=0
                    else:
                        print("there is something wrong")
    
    for key in workload_name:
        with open('./data/imdb/workload_vec/sample-'+key+'.pkl', 'wb') as file:
                pickle.dump(sample_vec_dict[key],file)
                print(key,sample_vec_dict[key].shape)
                print('stored in ./data/imdb/workload_vec/sample-'+key+'.pkl')     
        # print(sample_vec_dict[key][0])
        # print(predicate_vecs[0][0])
        # for j in sample_vec_dict[key][0]:
        #     print(sum(j))
    
    # for i in loc_to_attr_dict.items():
    #     print(i)
    return


def multi_table():
    time1=time.time()
    table_data_dict={}
    #load table data and get data information of six tables
    # ci usecols=[1,2,6]
    ci = pd.read_pickle('./data/imdb/cast_info.pickle')
    print(len(ci))
    print(ci[:5])

    mk = pd.read_csv('./data/imdb/movie_keyword.csv',header=None,usecols=[1,2])
    mk.columns=["movie_id",'keyword_id']
    mk=mk.astype(int)
    print(len(mk))
    print(mk[:5])
    

    mc = pd.read_csv('./data/imdb/movie_companies.csv',header=None,usecols=[1,2,3])
    mc.columns=['movie_id','company_id','company_type_id']
    mc=mc.astype(int)
    print(len(mc))
    print(mc[:5])

    mi = pd.read_csv('./data/imdb/movie_info.csv',header=None,usecols=[1,2])
    mi.columns=['movie_id',"info_type_id"]
    mi=mi.astype(int)
    print(len(mi))
    print(mi[:5])

    mi_idx = pd.read_csv('./data/imdb/movie_info_idx.csv',header=None,usecols=[1,2])
    mi_idx.columns=['movie_id','info_type_id']
    mi_idx=mi_idx.astype(int)
    print(len(mi_idx))
    print(mi_idx[:5])

    # t usecols=[0,3,4]
    t = pd.read_pickle('./data/imdb/title.pickle')
    t.columns=["id","kind_id","production_year"]
    t=t.astype(int)
    print(len(t))
    print(t[:5])

    # print(len(set(list(t.loc[:,'id']))))
    # print(36238972+4523930+2609129+14711168+1380035+2455878)
    # return
    
    table_data_dict['ci']=ci
    table_data_dict['mk']=mk
    table_data_dict['mc']=mc
    table_data_dict['mi']=mi
    table_data_dict['mi_idx']=mi_idx
    table_data_dict['t']=t

    # multi=1
    # for table_th in table_data_dict.keys():
    #     table=table_data_dict[table_th]
    #     for i in table.columns:
    #         print(table_th,i,len(set(table.loc[:,i])))
    #         print("------------------")
    #         multi*=len(set(table.loc[:,i]))
    # print(multi)
    # str = '{:e}'.format(multi)
    # print(str)
    # return

    primary_key_counter={}
    for i in table_data_dict.keys():
        # primary_key_counter[i]={}
        
        if i=='t':
            counter_dict=Counter(table_data_dict[i].loc[:,'id'])
            counter_dict=dict(sorted(counter_dict.items()))
            primary_key_counter[i]=counter_dict
        else:
            counter_dict=Counter(table_data_dict[i].loc[:,'movie_id'])
            counter_dict=dict(sorted(counter_dict.items()))
            primary_key_counter[i]=counter_dict
        print(i)
        print(len(counter_dict))
        # print(list(primary_key_counter[i].items())[:5])
        print("----------------")

    return
    with open('./data/imdb/data_info/primary_key_counter.pkl', 'wb') as file:
        pickle.dump(primary_key_counter,file)
        print('stored in ./data/imdb/data_info/primary_key_counter.pkl')
    

    min_max_dict={'ci':{},'mk':{},'mc':{},'mi':{},'mi_idx':{},'t':{}}
    # h_dicts={"equal_h":{},"more_than_h":{},'less_than_h':{}}
    histogram_dict={'ci':{},'mk':{},'mc':{},'mi':{},'mi_idx':{},'t':{}}
    for table_name in table_data_dict.keys():
        for attr_name in table_data_dict[table_name].columns:
            min_max_dict[table_name][attr_name]={}
            min_max_dict[table_name][attr_name]['max']=max(table_data_dict[table_name].loc[:,attr_name])
            min_max_dict[table_name][attr_name]['min']=min(table_data_dict[table_name].loc[:,attr_name])
            min_max_dict[table_name][attr_name]['difference']=min_max_dict[table_name][attr_name]['max']-min_max_dict[table_name][attr_name]['min']

            # print(table_name,attr_name)
            histogram_dict[table_name][attr_name]={}
            equal_h=Counter(table_data_dict[table_name].loc[:,attr_name])
            equal_h=dict(sorted(equal_h.items()))
            # print(len(equal_h))
            more_than_h={}
            less_than_h={}
            n_from_zero=0
            n_from_len=len(table_data_dict[table_name].loc[:,attr_name])
            keys=list(equal_h.keys())
            for i in range(len(keys)):
                key=keys[i]
                n_from_len-=equal_h[key]
                more_than_h[key]=n_from_len
                less_than_h[key]=n_from_zero
                n_from_zero+=equal_h[key]
            histogram_dict[table_name][attr_name]['equal_h']=equal_h
            histogram_dict[table_name][attr_name]['more_than_h']=more_than_h
            histogram_dict[table_name][attr_name]['less_than_h']=less_than_h


    for table_name in min_max_dict.keys():
        print(table_name)
        print(min_max_dict[table_name])

    

    joins = []
    predicates = []
    tables = []
    samples = []
    # label = []

    # Load queries
    workload_len_list=[]
    with open('./data/imdb/workloads/job-light.csv', 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        workload_len_list.append(len(data_raw))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            # label.append(int(row[3]))
    label = pd.read_pickle("./data/imdb/workload_vec/modified_label-job-light.pkl")
    # print(type(label))


    # return


    # for i in label:
    #     print(i)
    #     print(type(i))
    #     return

    # print(len(tables))

    
    with open('./data/imdb/workloads/scale.csv', 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        workload_len_list.append(len(data_raw))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
    
            # label.append(int(row[3]))
    new_label=pd.read_pickle("./data/imdb/workload_vec/modified_label-scale.pkl")
    label=torch.cat((label,new_label),0)


    with open('./data/imdb/workloads/synthetic.csv', 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        workload_len_list.append(len(data_raw))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            # label.append(int(row[3]))
    new_label=pd.read_pickle("./data/imdb/workload_vec/modified_label-synthetic.pkl")
    label=torch.cat((label,new_label),0)


    with open('./data/imdb/workloads/train.csv', 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        workload_len_list.append(len(data_raw))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            # label.append(int(row[3]))
    for i in range(10):
        new_label=pd.read_pickle('./data/imdb/workload_vec/modified_label-'+str(i*10000)+'-train.pkl')
        label=torch.cat((label,new_label),0)
    
    

    print(workload_len_list)
    print(len(tables))


    table_dict={}
    for i in tables:
        for j in i:
            table_dict[j]=1
    
    table_dict=list(sorted(table_dict))
    print(table_dict)
    return

    # table_length_dict={}
    # for key in table_dict:
    #     name=key.split(" ")[1]
    #     table_length_dict[name]=len(table_data_dict[name])

    # print(table_length_dict)

    
    join_dict={}
    for i in joins:
        for j in i:
            join_dict[j]=1
    join_dict=list(sorted(join_dict))
    print(join_dict)
    

    predicate_dict={}
    for i in predicates:
        for j in range(len(i)):
            if j%3==0:
                predicate_dict[i[j]]=1
    predicate_dict=list(sorted(predicate_dict))
    

    predicate_dict.remove('')
    print("predicate dict")
    print(len(predicate_dict))
    print(predicate_dict)
    

    # get table vec and normalized labels
    print("table vec")
    table_vec=np.zeros((len(tables),1,len(table_dict)))
    print(table_vec.shape)

    table_usage_key=[]
    table_usage_value=[]
    norm_label=[]
    card=[]
    for i in range(len(tables)):
        # print(tables[i])
        for j in tables[i]:
            # print(j)
            # print(table_dict.index(j))
            table_vec[i][0][table_dict.index(j)]=1
            # print(table_dict)
            # return
        usage_vec=list(table_vec[i][0])
        if usage_vec not in table_usage_key:
            # table_usage_list.append(list(table_vec[i][0]))
            if sum(usage_vec)==1:
                table_name=table_dict[usage_vec.index(1)].split(' ')[1]
                table_usage_key.append(usage_vec)
                table_usage_value.append(len(table_data_dict[table_name]))
            else:
                used_table_name=[]
                for w in range(len(usage_vec)-1):
                    if usage_vec[w]==1:
                        used_table_name.append(table_dict[w].split(" ")[1])
                count_num=0
                for key,value in primary_key_counter['t'].items():
                    for table_name in used_table_name:
                        try:
                            value*=primary_key_counter[table_name][key]
                        except:
                            value=0
                    count_num+=value
                table_usage_key.append(usage_vec)
                table_usage_value.append(count_num)
        new_number=label[i]/table_usage_value[table_usage_key.index(usage_vec)]
        # label[i]/=table_usage_value[table_usage_key.index(usage_vec)]
        # if new_number>1:
        #     print(tables[i],joins[i],predicates[i])
        #     print(new_number,label[i],table_usage_value[table_usage_key.index(usage_vec)],label[i]-table_usage_value[table_usage_key.index(usage_vec)])
        norm_label.append(new_number)
        card.append(table_usage_value[table_usage_key.index(usage_vec)])




    # for i in range(len(table_usage_key)):
    #     print(table_usage_key[i],table_usage_value[i])
        

        # print(table_vec[i])
        
    
    #get join vec
    print("join vec")
    join_vec=np.zeros((len(joins),1,len(join_dict)))
    print(join_vec.shape)
    for i in range(len(joins))[:5]:
        # print(joins[i])
        for j in joins[i]:
            join_vec[i][0][join_dict.index(j)]=1
        # print(join_vec[i])
        


    
    #get predicate encoding information
    attr_dict={}   
    for i in predicates:
        for j in range(len(i)):
            if j%3==0 and i[j]!='':
                table_name=i[j].split(".")[0]
                attr_name=i[j].split(".")[1]
                if table_name not in attr_dict.keys():
                    attr_dict[table_name]={}
                attr_dict[table_name][attr_name]=1
    
    attr_dict=dict(sorted(attr_dict.items()))
    for i in attr_dict.keys():
        attr_dict[i]=dict(sorted(attr_dict[i].items()))
    th=0
    for i in attr_dict.keys():
        for j in attr_dict[i].keys():
            attr_dict[i][j]=th
            th+=1
    print(attr_dict)


    with open('./data/imdb/data_info/attr_dict.pkl', 'wb') as file:
            pickle.dump(attr_dict,file)
            print('stored in ./data/imdb/data_info/attr_dict.pkl')
    
    
    loc_to_attr_dict={}
    for table_name in attr_dict.keys():
        for attr_name in attr_dict[table_name].keys():
            loc_to_attr_dict[attr_dict[table_name][attr_name]]=(table_name,attr_name)

    print(loc_to_attr_dict)
    return
    with open('./data/imdb/data_info/loc_to_attr_dict.pkl', 'wb') as file:
            pickle.dump(loc_to_attr_dict,file)
            print('stored in ./data/imdb/data_info/loc_to_attr_dict.pkl')

    
    
    
    #get predicate vec
    predicate_vec=np.zeros((len(predicates),len(predicate_dict),len(predicate_dict)+3+1+1+1))    
    print(predicate_vec.shape)
    for i in range(len(predicate_vec)):
        for j in range(len(predicate_vec[i])):
            predicate_vec[i][j][-1]=1

    print(predicate_dict)


    attr_th=0
    table_name=None
    attr_name=None
    for i in range(len(predicates)):
        # print(predicates[i])
        for j in range(len(predicates[i])):
            if j%3==0:
                if predicates[i][j]!="":
                    table_name=predicates[i][j].split(".")[0]
                    attr_name=predicates[i][j].split(".")[1]
                    attr_th=attr_dict[table_name][attr_name]
                    predicate_vec[i][attr_th][attr_th]=1 # one-hot
                else:
                    continue
            # =:1 0 0; >: 0 1 0; <: 0 0 1
            if j%3==1:
                # value=float(predicates[i][j+1])
                value=int(predicates[i][j+1])
                min_value=min_max_dict[table_name][attr_name]['min']
                max_value=min_max_dict[table_name][attr_name]['max']
                difference_value=min_max_dict[table_name][attr_name]['difference']
                if predicates[i][j]=="=":
                    predicate_vec[i][attr_th][len(predicate_dict)]=1
                    predicate_vec[i][attr_th][len(predicate_dict)+3]=value
                    predicate_vec[i][attr_th][len(predicate_dict)+4]=value
                elif predicates[i][j]==">":
                    if value>=min_value:
                        predicate_vec[i][attr_th][len(predicate_dict)+1]=1
                        predicate_vec[i][attr_th][len(predicate_dict)+3]=value
                        predicate_vec[i][attr_th][len(predicate_dict)+4]=1
                elif predicates[i][j]=="<":
                    if value<=max_value:
                        predicate_vec[i][attr_th][len(predicate_dict)+2]=1
                        predicate_vec[i][attr_th][len(predicate_dict)+4]=value
            # if j%3==2:
            #     predicate_vec[i][attr_th][len(predicate_dict)+3]=(float(predicates[i][j])-min_max_dict[table_name][attr_name]['min'])/min_max_dict[table_name][attr_name]['difference']

        # print(predicate_vec[i])

    print(predicates[0])
    print(predicate_vec[0])

    th=0
    workload_name=['job-light','scale','synthetic','train']
    for i in range(len(workload_len_list)):
        with open('./data/imdb/workload_vec/int_predicate-'+workload_name[i]+'.pkl', 'wb') as file:
                pickle.dump(predicate_vec[th:th+workload_len_list[i]],file)
                print('stored in ./data/imdb/workload_vec/int_predicate-'+workload_name[i]+'.pkl')
        th+=workload_len_list[i] 


    attr_th=None

    for i in range(len(predicate_vec)):
        # print("before")
        # print(predicate_vec[i])
        for th in range(len(predicate_vec[i])):
            predicate_flag=list(predicate_vec[i][th][len(predicate_dict):len(predicate_dict)+3])
            # print(i[th])
            # print(predicate_flag)

            if predicate_flag==[0,0,0]:
                continue
            elif predicate_flag==[1,0,0]:
                (table_name,attr_name)=loc_to_attr_dict[th]
                value=predicate_vec[i][th][len(predicate_dict)+3]
                try:
                    predicate_vec[i][th][len(predicate_dict)+5]=histogram_dict[table_name][attr_name]['equal_h'][value]/len(table_data_dict[table_name])
                except:
                    # print(predicate_vec[i][th])
                    # if value<min_max_dict[table_name][attr_name]['min'] or value>min_max_dict[table_name][attr_name]['max']:
                    print(table_name,attr_name,"=",value)
                    predicate_vec[i][th][len(predicate_dict)+5]=histogram_dict[table_name][attr_name]['equal_h'][value]/len(table_data_dict[table_name])
                    
                min_value=min_max_dict[table_name][attr_name]['min']
                difference_value=min_max_dict[table_name][attr_name]['difference']
                predicate_vec[i][th][len(predicate_dict)+3]=(value-min_value)/difference_value
                predicate_vec[i][th][len(predicate_dict)+4]=(value-min_value)/difference_value
            elif predicate_flag==[0,1,0]:
                (table_name,attr_name)=loc_to_attr_dict[th]
                value=predicate_vec[i][th][len(predicate_dict)+3]
                try:
                    predicate_vec[i][th][len(predicate_dict)+5]=histogram_dict[table_name][attr_name]['more_than_h'][value]/len(table_data_dict[table_name])
                except:
                    # if value<min_max_dict[table_name][attr_name]['min'] or value>min_max_dict[table_name][attr_name]['max']:
                    print(table_name,attr_name,">",value,min_max_dict[table_name][attr_name]['min'],min_max_dict[table_name][attr_name]['max'])
                    predicate_vec[i][th][len(predicate_dict)+5]=histogram_dict[table_name][attr_name]['more_than_h'][value]/len(table_data_dict[table_name])
                    # print(value in histogram_dict[table_name][attr_name]['more_than_h'].keys())
                    # print(int(value) in histogram_dict[table_name][attr_name]['more_than_h'].keys())
                min_value=min_max_dict[table_name][attr_name]['min']
                difference_value=min_max_dict[table_name][attr_name]['difference']
                predicate_vec[i][th][len(predicate_dict)+3]=(value-min_value)/difference_value
            elif predicate_flag==[0,0,1]:
                (table_name,attr_name)=loc_to_attr_dict[th]
                value=predicate_vec[i][th][len(predicate_dict)+4]
                try:
                    predicate_vec[i][th][len(predicate_dict)+5]=histogram_dict[table_name][attr_name]['less_than_h'][value]/len(table_data_dict[table_name])
                except:
                    # if value<min_max_dict[table_name][attr_name]['min'] or value>min_max_dict[table_name][attr_name]['max']:
                    print(table_name,attr_name,"<",value,min_max_dict[table_name][attr_name]['min'],min_max_dict[table_name][attr_name]['max'])
                    next_th=bisect.bisect_left(list(histogram_dict[table_name][attr_name]['more_than_h'].keys()),value)
                    value=list(histogram_dict[table_name][attr_name]['more_than_h'].keys())[next_th]
                    # print(value)
                    # print(value in histogram_dict[table_name][attr_name]['more_than_h'].keys())
                    predicate_vec[i][th][len(predicate_dict)+5]=histogram_dict[table_name][attr_name]['less_than_h'][value]/len(table_data_dict[table_name])
                    # print(value in histogram_dict[table_name][attr_name]['more_than_h'].keys())
                    # print(int(value) in histogram_dict[table_name][attr_name]['more_than_h'].keys())
                    # print(list(histogram_dict[table_name][attr_name]['more_than_h'].keys())[:15])
                    # print(len(histogram_dict[table_name][attr_name]['more_than_h'].keys()))
                min_value=min_max_dict[table_name][attr_name]['min']
                difference_value=min_max_dict[table_name][attr_name]['difference']
                predicate_vec[i][th][len(predicate_dict)+4]=(value-min_value)/difference_value
            elif predicate_flag==[0,1,1]:
                (table_name,attr_name)=loc_to_attr_dict[th]
                value1=predicate_vec[i][th][len(predicate_dict)+3]
                value2=predicate_vec[i][th][len(predicate_dict)+4]
                predicate_vec[i][th][len(predicate_dict)+5]= (histogram_dict[table_name][attr_name]['more_than_h'][value1]+  \
                                             histogram_dict[table_name][attr_name]['less_than_h'][value2])/   \
                                                len(table_data_dict[table_name])-1
                min_value=min_max_dict[table_name][attr_name]['min']
                difference_value=min_max_dict[table_name][attr_name]['difference']
                predicate_vec[i][th][len(predicate_dict)+3]=(value1-min_value)/difference_value
                predicate_vec[i][th][len(predicate_dict)+4]=(value2-min_value)/difference_value

    
    print("predicate vec")
    print(predicate_vec.shape)
    for j in range(5):
        for i in range(len(predicates[j])):
            if i%3==0:
                print(predicates[j][i:i+3])
        print(predicate_vec[j])
        
    
    # save the processed vectors

    table_vec=torch.FloatTensor(table_vec)
    join_vec=torch.FloatTensor(join_vec)
    predicate_vec=torch.FloatTensor(predicate_vec)
    norm_label=torch.FloatTensor(np.array(norm_label))
    card=torch.FloatTensor(np.array(card))
    
    table_vec=torch.unsqueeze(table_vec,dim=1)
    join_vec=torch.unsqueeze(join_vec,dim=1)
    predicate_vec=torch.unsqueeze(predicate_vec,dim=1)
    norm_label=torch.unsqueeze(norm_label,dim=1)
    card=torch.unsqueeze(card,dim=1)


    th=0
    workload_name=['job-light','scale','synthetic','train']
    for i in range(len(workload_len_list)):
        print("workload_name:",workload_name[i],workload_len_list[i],"queries")
        with open('./data/imdb/workload_vec/table-'+workload_name[i]+'.pkl', 'wb') as file:
            pickle.dump(table_vec[th:th+workload_len_list[i]],file)
            print('stored in ./data/imdb/workload_vec/table-'+workload_name[i]+'.pkl')
            
        with open('./data/imdb/workload_vec/join-'+workload_name[i]+'.pkl', 'wb') as file:
            pickle.dump(join_vec[th:th+workload_len_list[i]],file)
            print('stored in ./data/imdb/workload_vec/join-'+workload_name[i]+'.pkl')

        with open('./data/imdb/workload_vec/predicate-'+workload_name[i]+'.pkl', 'wb') as file:
            pickle.dump(predicate_vec[th:th+workload_len_list[i]],file)
            print('stored in ./data/imdb/workload_vec/predicate-'+workload_name[i]+'.pkl')

        with open('./data/imdb/workload_vec/norm_label-'+workload_name[i]+'.pkl', 'wb') as file:
            pickle.dump(norm_label[th:th+workload_len_list[i]],file)
            print('stored in ./data/imdb/workload_vec/norm_label-'+workload_name[i]+'.pkl')
        
        with open('./data/imdb/workload_vec/card-'+workload_name[i]+'.pkl', 'wb') as file:
            pickle.dump(card[th:th+workload_len_list[i]],file)
            print('stored in ./data/imdb/workload_vec/card-'+workload_name[i]+'.pkl')
        # join_vec[th:th+workload_len_list[i]]
        # predicate_vec[th:th+workload_len_list[i]]
        th+=workload_len_list[i]

    print("has spent "+str(time.time()-time1)+" seconds")
    return
    predicate_dict={}
    for i in predicates:
        for j in range(len(i)):
            if j%3==0:
                predicate_dict[i[j]]=1
    print(sorted(predicate_dict))

import torch
import torch.nn as nn
import torch.nn.functional as F

class SetConv(nn.Module):
    def __init__(self,predicate_num,predicate_length,table_length,join_length,sample_num,sample_length):
        super(SetConv, self).__init__()
        # self.predicate_mlp1 = nn.Linear(predicate_length, 128)
        # self.predicate_mlp2 = nn.Linear(128, 128)

        # self.table_mlp1 = nn.Linear(table_length, 128)
        # self.table_mlp2 = nn.Linear(128, 128)

        # self.join_mlp1 = nn.Linear(join_length, 128)
        # self.join_mlp2 = nn.Linear(128, 128)
        
        # self.sample_mlp1 = nn.Linear(sample_length,sample_length)
        # self.sample_mlp2 = nn.Linear(sample_length,128)
        
        # self.out_mlp1 =nn.Linear(128*(sample_num+predicate_num+2),512)
        # self.out_mlp2 = nn.Linear(512, 1)
        #-----------------------------------------

        hid_units=256
        self.predicate_mlp1 = nn.Linear(predicate_length, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)

        self.table_mlp1 = nn.Linear(table_length, hid_units)
        self.table_mlp2 = nn.Linear(hid_units,hid_units)

        self.join_mlp1 = nn.Linear(join_length, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        
        self.sample_mlp1 = nn.Linear(sample_length,sample_length)
        self.sample_mlp2 = nn.Linear(sample_length,hid_units)
        
        self.out_mlp1 =nn.Linear(hid_units*(sample_num+predicate_num+2),hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)


    def forward(self,tables,joins,predicates,samples):
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)

        hid_table = F.relu(self.table_mlp1(tables))
        hid_table = F.relu(self.table_mlp2(hid_table))
        hid_table = torch.sum(hid_table, dim=1, keepdim=False)

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)

        cat_one=torch.cat((hid_predicate,hid_table,hid_join,hid_sample), 1)

        cat_one=torch.flatten(cat_one, start_dim=1)
        cat_one = F.relu(self.out_mlp1(cat_one))
        cat_one = torch.sigmoid(self.out_mlp2(cat_one))
        
        return cat_one

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predications, labels, upper_bounds):
        delta=predications-upper_bounds
        return torch.mean(torch.pow((predications - labels), 2)+torch.pow(torch.where(delta>0,delta,0),2))
  

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

def evaluate_errors(errors):
    metrics = {
        'max': np.max(errors),
        '99th': np.percentile(errors, 99),
        '95th': np.percentile(errors, 95),
        '90th': np.percentile(errors, 90),
        'mean': np.mean(errors),
    }
    print(metrics)

def four_evaluations(prediction,p_labels):
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



from torch.utils.data import TensorDataset

def train_multi_table():
    workload_name=['job-light','scale','synthetic','train']
    table_vecs=[]
    join_vecs=[]
    predicate_vecs=[]
    labels=[]
    cards=[]
    samples=[]
    
    for i in range(len(workload_name)):
        with open('./data/imdb/workload_vec/table-'+workload_name[i]+'.pkl', 'rb') as fo: 
            table_vecs.append(pickle.load(fo, encoding='bytes'))
        with open('./data/imdb/workload_vec/join-'+workload_name[i]+'.pkl', 'rb') as fo: 
            join_vecs.append(pickle.load(fo, encoding='bytes'))
        with open('./data/imdb/workload_vec/predicate-'+workload_name[i]+'.pkl', 'rb') as fo: 
            predicate_vecs.append(pickle.load(fo, encoding='bytes'))
        with open('./data/imdb/workload_vec/norm_label-'+workload_name[i]+'.pkl', 'rb') as fo: 
            labels.append(pickle.load(fo, encoding='bytes'))
        with open('./data/imdb/workload_vec/card-'+workload_name[i]+'.pkl', 'rb') as fo: 
            cards.append(pickle.load(fo, encoding='bytes'))
        with open('./data/imdb/workload_vec/sample-'+workload_name[i]+'.pkl', 'rb') as fo: 
            samples.append(torch.unsqueeze(pickle.load(fo, encoding='bytes'),dim=1))


    table_length=table_vecs[0].shape[3]
    join_length=join_vecs[0].shape[3]
    predicate_length=predicate_vecs[0].shape[3]
    predicate_num=predicate_vecs[0].shape[2]
    sample_num=samples[0].shape[2]
    sample_length=samples[0].shape[3]
    
    model=SetConv(predicate_num,predicate_length,table_length,join_length,sample_num,sample_length)
    loss_func=My_loss()
    
    # data_length=len(new_data)
    lr=0.001
    opt = torch.optim.Adam(model.parameters(),lr=lr)

    valid_table_vecs=table_vecs[-1][90000:]
    valid_join_vecs=join_vecs[-1][90000:]
    valid_predicate_vecs=predicate_vecs[-1][90000:]
    valid_samples=samples[-1][90000:]
    valid_labels=labels[-1][90000:]
    valid_cards=cards[-1][90000:]

    train_table_vecs=table_vecs[-1][:90000]
    train_join_vecs=join_vecs[-1][:90000]
    train_predicate_vecs=predicate_vecs[-1][:90000]
    train_samples=samples[-1][:90000]
    train_labels=labels[-1][:90000]
    train_cards=cards[-1][:90000]

    cuda = False if DEVICE == 'cpu' else True
    if cuda:
        model=model.cuda()
        for i in range(len(table_vecs)-1):
            table_vecs[i]=table_vecs[i].cuda()
            join_vecs[i]=join_vecs[i].cuda()
            predicate_vecs[i]=predicate_vecs[i].cuda()
            labels[i]=labels[i].cuda()
            cards[i]=cards[i].cuda()
            samples[i]=samples[i].cuda()

            valid_table_vecs=valid_table_vecs.cuda()
            valid_join_vecs=valid_join_vecs.cuda()
            valid_predicate_vecs=valid_predicate_vecs.cuda()
            valid_samples=valid_samples.cuda()
            valid_labels=valid_labels.cuda()
            valid_cards=valid_cards.cuda()


    # train_data = TensorDataset(table_vecs[-1],join_vecs[-1],predicate_vecs[-1],samples[-1],labels[-1],cards[-1])
    train_data = TensorDataset(train_table_vecs,train_join_vecs,train_predicate_vecs,train_samples,train_labels,train_cards)
    train_loader = data.DataLoader(train_data,batch_size=200,shuffle=False)

    best_model=None
    best_mean_MAE=float("inf")
    for epoch in range(100):
        if epoch%5==4:
            lr*=0.8
            opt = torch.optim.Adam(model.parameters(),lr=lr)
        for i,(x,y,z,sample,label,card) in enumerate(train_loader):
            upper_bound,_=torch.min(z[:,0,:,-1],axis=1)
            upper_bound=torch.FloatTensor(upper_bound)
            upper_bound=torch.unsqueeze(upper_bound,dim=1)
            # print(upper_bound)
            if cuda:
                x=Variable(x.cuda())
                y=Variable(y.cuda())
                z=Variable(z.cuda())
                sample=Variable(sample.cuda())
                label=Variable(label.cuda())
                card=Variable(card.cuda())
                upper_bound=Variable(upper_bound.cuda())
            out = model(x,y,z,sample)
            loss = loss_func(out,label,upper_bound)
            opt.zero_grad()  
            loss.backward()
            opt.step()

        print("epoch ",epoch,":")
        
        # out=model(table_vecs[test_workload_th],join_vecs[test_workload_th],predicate_vecs[test_workload_th],samples[test_workload_th])
        # mul_out=np.array(torch.mul(out,cards[test_workload_th]).cpu().detach())
        # label_out=np.array(torch.mul(labels[test_workload_th],cards[test_workload_th]).cpu().detach())
        
        label_out=np.array(torch.mul(valid_labels,valid_cards).cpu().detach())
        out=model(valid_table_vecs,valid_join_vecs,valid_predicate_vecs,valid_samples)
        mul_out=np.array(torch.mul(out,valid_cards).cpu().detach())
        
        mae=np.absolute(mul_out-label_out)
        if np.mean(mae)<best_mean_MAE:
            best_model=model
            best_mean_MAE=np.mean(mae)
            # best_epoch=epoch
            four_evaluations(mul_out,label_out)
            print("--------------------------------")

    print("best model for valid")
    # label_out=np.array(torch.mul(valid_labels,valid_cards).cpu().detach())
    out=best_model(valid_table_vecs,valid_join_vecs,valid_predicate_vecs,valid_samples)
    # mul_out=np.array(torch.mul(out,valid_cards).cpu().detach())
    
    addr="./lecarb/estimator/mine/multi_learning_prediction/valid.pkl"
    with open(addr, 'wb') as f:
        pickle.dump([out,valid_labels.cpu().detach().numpy(),valid_cards.cpu().detach().numpy()], f)
        print("the vec data has been stored in "+addr)


    print("best model for test")
    threshold=0.0001639672122217379
    for test_workload_th in [0,1,2]:
        print("---------------------------")
        # test_workload_th=1
        print(workload_name[test_workload_th],len(table_vecs[test_workload_th]))
        out=best_model(table_vecs[test_workload_th],join_vecs[test_workload_th],predicate_vecs[test_workload_th],samples[test_workload_th])  
        
        addr="./lecarb/estimator/mine/multi_learning_prediction/"+str(workload_name[test_workload_th])+".pkl"
        with open(addr, 'wb') as f:
            pickle.dump([out,labels[test_workload_th].cpu().detach().numpy(),cards[test_workload_th].cpu().detach().numpy()], f)
            print("the vec data has been stored in "+addr)
        
        print()
        print("test with BND-CN")
        mul_out=np.array(torch.mul(out,cards[test_workload_th]).cpu().detach())
        label_out=np.array(torch.mul(labels[test_workload_th],cards[test_workload_th]).cpu().detach())    
        four_evaluations(mul_out,label_out)

        turn_to_count=0
        for i in range(len(out)):
            if out[i]<threshold:
                out[i]=labels[test_workload_th][i]
                turn_to_count+=1
        
        print("test with BND-CN+AC-Forest")
        mul_out=np.array(torch.mul(out,cards[test_workload_th]).cpu().detach())
        label_out=np.array(torch.mul(labels[test_workload_th],cards[test_workload_th]).cpu().detach())    
        four_evaluations(mul_out,label_out)
        print("turn to accurate:",turn_to_count)
    
    
        
