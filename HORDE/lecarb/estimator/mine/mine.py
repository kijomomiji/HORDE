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
    def __init__(self,data,attr_name,correlation,value_to_int_dict,attr_type_dict,range_size):
        self.data=data
        self.attr_name=attr_name
        self.correlation=correlation
        self.value_to_int_dict=value_to_int_dict # value_to_int_dict[key][value] to get the int of str value
        self.attr_type_dict=attr_type_dict  # attr_type_dict[attr] to get the type of this attribute
        self.range_size=range_size

class workload_data:
    def __init__(self,query_vec,query_label):
        self.query_vec=query_vec
        self.query_label=query_label


def find_the_data(dataset, version):

    # print(dataset,version,workload)

    start_time=time.time()
    # args = Args(**params)
    table = load_table(dataset, version)

    value_range_count={}
    value_to_int_dict={}
    for key in table.columns.keys():
        value_range_count[key]={}
        value_to_int_dict[key]={}
    
    
    # count the value_range for every attribute
    
    for key in table.columns.keys():
        if str(table.columns[key].dtype)!='category': 
            continue
        for i in range(table.row_num):
            value_range_count[key][table.data[key][i]]=1
    
    for attr in table.columns.keys():
        if str(table.columns[attr].dtype)!='category':
            continue
        for value in range(len(value_range_count[attr].keys())):
            value_to_int_dict[attr][list(value_range_count[attr].keys())[value]]=value
    
    attr_type_dict={}
    for attr in table.columns.keys():
        attr_type_dict[attr]=str(table.columns[attr].dtype)


    new_data=[]
    for i in range(table.row_num):
        row_data=[]
        for key in table.columns.keys():
            if str(table.columns[key].dtype)!='category':
                row_data.append(table.data[key][i])
            else:
                row_data.append(value_to_int_dict[key][table.data[key][i]])
        new_data.append(row_data)

    print("row number:",len(new_data))
    print("attribute number:",len(new_data[0]))
    attr_name=list(table.columns.keys())
    print(attr_name)


    test_data = pd.DataFrame(new_data,columns=attr_name)
    correlation=test_data.corr()
    print("correlation:")
    print(correlation)

    range_size=[table.columns[name].vocab_size for name in attr_name]

    save_data=table_data(new_data,attr_name,correlation,value_to_int_dict,attr_type_dict,range_size)
    save_data_to_file(save_data,dataset+"_"+version+".pkl")

    end_time=time.time()
    print("spent",end_time-start_time)


# def find_the_query(dataset, version, workload, params):
#     start_time=time.time()
#     queryset = load_queryset(dataset, workload)
#     labels = load_labels(dataset, version, workload)
#     args = Args(**params)
#     if args.train_num < len(queryset['train']):
#         queryset['train'] = queryset['train'][:args.train_num]
#         labels['train'] = labels['train'][:args.train_num]
#     valid_num = args.train_num // 10
#     if valid_num < len(queryset['valid']):
#         queryset['valid'] = queryset['valid'][:valid_num]
#         labels['valid'] = labels['valid'][:valid_num]
#     L.info(f"Use {len(queryset['train'])} queries for train and {len(queryset['valid'])} queries for validation")
    

#     vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")
#     #data,attr_name,correlation,value_to_int_dict
#     value_to_int_dict=vec_data.value_to_int_dict
#     #attr_type_dict=vec_data.attr_type_dict

#     # training query
#     query_vec=[]
#     query_label=[]
#     for i in range(len(queryset["train"])):
#         vec=[]
#         for attr in queryset["train"][i].predicates:
#             screen_tuple=queryset["train"][i].predicates[attr]
#             if screen_tuple==None:
#                 vec.append([0])
#             elif screen_tuple[0]=="=":
#                 vec.append([1,value_to_int_dict[attr][screen_tuple[1]]])
#             elif screen_tuple[0]==">=":
#                 vec.append([2,screen_tuple[1]])
#             elif screen_tuple[0]=="<=":
#                 vec.append([3,screen_tuple[1]])
#             elif screen_tuple[0]=="[]":
#                 vec.append([4,screen_tuple[1][0],screen_tuple[1][1]])
#         query_vec.append(vec)
#         query_label.append(labels['train'][i].cardinality)

#     save_data=workload_data(query_vec,query_label)
#     save_data_to_file(save_data,dataset+"_"+version+"_workload.pkl")

#     # testing query
#     query_vec=[]
#     query_label=[]
#     for i in range(len(queryset["valid"])):
#         vec=[]
#         for attr in queryset["valid"][i].predicates:
#             screen_tuple=queryset["valid"][i].predicates[attr]
#             if screen_tuple==None:
#                 vec.append([0])
#             elif screen_tuple[0]=="=":
#                 vec.append([1,value_to_int_dict[attr][screen_tuple[1]]])
#             elif screen_tuple[0]==">=":
#                 vec.append([2,screen_tuple[1]])
#             elif screen_tuple[0]=="<=":
#                 vec.append([3,screen_tuple[1]])
#             elif screen_tuple[0]=="[]":
#                 vec.append([4,screen_tuple[1][0],screen_tuple[1][1]])
#         query_vec.append(vec)
#         query_label.append(labels['valid'][i].cardinality)

#     save_data=workload_data(query_vec,query_label)
#     save_data_to_file(save_data,"test_"+dataset+"_"+version+"_workload.pkl")
#     '''
#     for i in range(2):
#         print(queryset["train"][i])
#         print(query_vec[i])
#         print(labels['train'][i])
#         print(query_label[i])
#         print("------------------------------------")
#     '''
#     end_time=time.time()
#     print("spent",end_time-start_time)

def find_the_query(dataset, version, workload, params):
    start_time=time.time()
    queryset = load_queryset(dataset, workload)

    labels = load_labels(dataset, version, workload)
    args = Args(**params)
    if args.train_num < len(queryset['train']):
        queryset['train'] = queryset['train'][:args.train_num]
        labels['train'] = labels['train'][:args.train_num]
    valid_num = args.train_num // 10
    if valid_num < len(queryset['valid']):
        queryset['valid'] = queryset['valid'][:valid_num]
        labels['valid'] = labels['valid'][:valid_num]

    L.info(f"Use {len(queryset['train'])} queries for train, {len(queryset['valid'])} queries for valid, {len(queryset['test'])} queries for test")
    

    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")
    #data,attr_name,correlation,value_to_int_dict
    value_to_int_dict=vec_data.value_to_int_dict
    #attr_type_dict=vec_data.attr_type_dict

    # training query
    query_vec=[]
    query_label=[]
    for i in range(len(queryset["train"])):
        vec=[]
        for attr in queryset["train"][i].predicates:
            screen_tuple=queryset["train"][i].predicates[attr]
            if screen_tuple==None:
                vec.append([0])
            elif screen_tuple[0]=="=":
                vec.append([1,value_to_int_dict[attr][screen_tuple[1]]])
            elif screen_tuple[0]==">=":
                vec.append([2,screen_tuple[1]])
            elif screen_tuple[0]=="<=":
                vec.append([3,screen_tuple[1]])
            elif screen_tuple[0]=="[]":
                vec.append([4,screen_tuple[1][0],screen_tuple[1][1]])
        query_vec.append(vec)
        query_label.append(labels['train'][i].cardinality)

    save_data=workload_data(query_vec,query_label)
    save_data_to_file(save_data,dataset+"_"+version+"_workload.pkl")

    # valid query
    query_vec=[]
    query_label=[]
    for i in range(len(queryset["valid"])):
        vec=[]
        for attr in queryset["valid"][i].predicates:
            screen_tuple=queryset["valid"][i].predicates[attr]
            if screen_tuple==None:
                vec.append([0])
            elif screen_tuple[0]=="=":
                vec.append([1,value_to_int_dict[attr][screen_tuple[1]]])
            elif screen_tuple[0]==">=":
                vec.append([2,screen_tuple[1]])
            elif screen_tuple[0]=="<=":
                vec.append([3,screen_tuple[1]])
            elif screen_tuple[0]=="[]":
                vec.append([4,screen_tuple[1][0],screen_tuple[1][1]])
        query_vec.append(vec)
        query_label.append(labels['valid'][i].cardinality)

    save_data=workload_data(query_vec,query_label)
    save_data_to_file(save_data,"valid_"+dataset+"_"+version+"_workload.pkl")

    # testing query
    query_vec=[]
    query_label=[]
    for i in range(len(queryset["test"])):
        vec=[]
        for attr in queryset["test"][i].predicates:
            screen_tuple=queryset["test"][i].predicates[attr]
            if screen_tuple==None:
                vec.append([0])
            elif screen_tuple[0]=="=":
                vec.append([1,value_to_int_dict[attr][screen_tuple[1]]])
            elif screen_tuple[0]==">=":
                vec.append([2,screen_tuple[1]])
            elif screen_tuple[0]=="<=":
                vec.append([3,screen_tuple[1]])
            elif screen_tuple[0]=="[]":
                vec.append([4,screen_tuple[1][0],screen_tuple[1][1]])
        query_vec.append(vec)
        query_label.append(labels['test'][i].cardinality)

    save_data=workload_data(query_vec,query_label)
    save_data_to_file(save_data,"test_"+dataset+"_"+version+"_workload.pkl")
    end_time=time.time()
    print("spent",end_time-start_time)

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
    
    leaf_nodes=[]
    new_tree=attr_tree(list(range(len(attr_name))),None,None,False,None,0.1)
    new_tree.build_the_tree(correlation,attr_name)
    print("there are "+str(len(leaf_nodes))+" leaf nodes.")
    new_tree_structure=[new_tree.get_the_tree_structure()]
    #print(new_tree_structure)
    print(new_tree.get_the_attr_list([]))
    save_data_to_file(new_tree_structure,dataset+"_"+version+"_tree_structure.pkl")

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

prob_list=None        

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
    c_max_layer=len(screen_list)
    if c_layer_number>=c_max_layer:
        return
    
    max_layer=len(screen_list[c_layer_number])
    print(layer_number,max_layer)
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
                    



from queue import Queue          

def fill_the_structure(root=None,new_data=None,attr_clusters=None):
    max_layer=len(attr_clusters)
    if root==None:
        root=cardinality_estimation_structure(0,None,None)
    th=0
    for d in new_data:
        if th%10000==0:
            print("has finished "+str(th)+"/"+str(len(new_data))+" updating data")
        th+=1
        root.add_new_data(d,0,attr_clusters,max_layer)
    return root

def show_structure(root,attr_clusters=None,dataset=None,version=None):
    f = open("./lecarb/estimator/mine/model_information/"+dataset+"_"+version+".txt","w")
    total_model_size=0
    max_layer=len(attr_clusters)
    out_queue_tuple=Queue()
    out_queue=Queue()
    for value_tuple,out_one in root.next1.items():
        out_queue_tuple.put(value_tuple)
        out_queue.put(out_one)
    for i in range(max_layer):
        #print("the "+str(i)+" layer:")
        f.write("the "+str(i)+" layer:\n")
        out_num=out_queue_tuple.qsize()
        for i in range(out_num):
            out_one=out_queue.get()
            out_tuple=out_queue_tuple.get()
            #print("cardinality:",out_one.cardinality," value_tuple:",out_tuple)
            f.write("cardinality:"+str(out_one.cardinality)+" value_tuple:"+str(out_tuple)+" model_size:"+str(sys.getsizeof(out_one.next1))+"\n")
            total_model_size+=sys.getsizeof(out_one.next1)
            for u,v in out_one.next1.items():
                out_queue_tuple.put(u)
                out_queue.put(v)
    f.close()
    return total_model_size

def query_info(dataset,version):
    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label    

    return


def test_and_calc_p_value(root_node,attr_clusters,query_vec,query_label):

    start_time1=time.time()
    print("start to test on "+str(len(query_label))+" queries")
    precision=0
    x_probs=[]
    for i in range(len(query_vec)):
        if i%1000==0:
            print("finished "+str(i)+" queries")
        start_time=time.time()
        query_clusters=[]
        for j in attr_clusters:
            query_clusters.append([query_vec[i][w] for w in j])
        
        screen_exist=[0 for t in range(len(attr_clusters))]
        for screen_th in range(len(attr_clusters)):
            screen=query_clusters[screen_th]
            flag=True
            for one in screen:
                if one!=[0]:
                    flag=False
                    break
            screen_exist[screen_th]=flag

        #print(query_clusters,screen_exist)
        for check_max in range(len(attr_clusters)-1,-1,-1):
            if screen_exist[check_max]==False:
                max_layer=check_max
                break
        max_layer+=1

        global prob_list
        prob_list=[0 for i in range(max_layer)]
        root_node.calc_prob(query_clusters,0,max_layer,screen_exist)
        #print(prob_list,"  ",query_label[i])
        end_time=time.time()
        
        
        print(str(query_clusters)+str(screen_exist)+" , "+str((end_time-start_time)*1000)+"ms")
        if prob_list[-1]==query_label[i]:
            precision+=1
        x_probs.append(prob_list[-1])
    print("the precision is:",precision/len(query_label))
    end_time1=time.time()
    print("spent "+str(end_time1-start_time1)+" seconds")
    print("every query spends "+str((end_time1-start_time1)*1000/len(query_vec))+"ms on average")

    return x_probs


def fill_and_show_structure(dataset,version):
    if dataset=="draft" and version=="draft":
        new_data=[[1,2,3,4,5],[1,2,3,4,6],[1,2,4,4,5],[1,2,4,4,6],[2,4,1,2,3]]
        attr_clusters=[[0,1],[2],[3,4]]
        root_node=fill_the_structure(None,new_data,attr_clusters)
        show_structure(root_node,attr_clusters,dataset,version)
    else:
        vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")
        
        new_data=vec_data.data
        attr_name=vec_data.attr_name
        correlation=vec_data.correlation

        new_tree=attr_tree(list(range(len(attr_name))),None,None,False,None,0.1)
        new_tree.build_the_tree(correlation,attr_name)
        
        attr_clusters=new_tree.get_the_attr_list([])
        print(attr_clusters)
          
        start_time=time.time()
        #print(len(new_data))

        root_nodes=[]
        for i in range(len(attr_clusters)):
            root_node=fill_the_structure(None,new_data,attr_clusters[i:i+1])
            root_nodes.append(root_node)
        end_time=time.time()

        print("has spent "+str(end_time-start_time)+" seconds to fill the structure")
        #total_model_size=show_structure(root_node,attr_clusters,dataset,version)
        #print("the model size is:",total_model_size)

        save_data(root_nodes,"./lecarb/estimator/mine/filled_model/"+dataset+"_"+version+".pkl")

        ### training data
        workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
        query_vec=workload_data.query_vec
        query_label=workload_data.query_label
        x_probs=[]
        train_x=[]
        train_y=[]
        data_num=len(new_data)
        for i in range(len(attr_clusters)):
            x_probs.append(test_and_calc_p_value(root_node,attr_clusters[i:i+1],query_vec,query_label))
        for i in range(len(query_label)):
            train_x.append([h[i]/data_num for h in x_probs])
            train_y.append(query_label[i]/data_num)

        for i in range(20):
            print(train_x[i]," ",train_y[i])
        # save_data_to_file(train_x,"train_x_"+dataset+"_"+version+".pkl")
        # save_data_to_file(train_y,"train_y_"+dataset+"_"+version+".pkl")

        ### testing data
        test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
        test_query_vec=test_workload_data.query_vec
        test_query_label=test_workload_data.query_label
        x_probs=[]
        test_x=[]
        test_y=[]
        data_num=len(new_data)
        for i in range(len(attr_clusters)):
            x_probs.append(test_and_calc_p_value(root_node,attr_clusters[i:i+1],test_query_vec,test_query_label))
        for i in range(len(test_query_label)):
            test_x.append([h[i]/data_num for h in x_probs])
            test_y.append(test_query_label[i]/data_num)
        
        for i in range(20):
            print(test_x[i]," ",test_y[i])
        # save_data_to_file(test_x,"test_x_"+dataset+"_"+version+".pkl")
        # save_data_to_file(test_y,"test_y_"+dataset+"_"+version+".pkl")

        '''
        ### testing data
        test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
        test_query_vec=test_workload_data.query_vec
        test_query_label=test_workload_data.query_label
        test_and_calc_p_value(root_node,attr_clusters,test_query_vec,test_query_label)
        '''