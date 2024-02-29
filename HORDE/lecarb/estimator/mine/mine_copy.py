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


def find_the_data(dataset, version, workload, params):

    print(dataset,version,workload)

    start_time=time.time()
    args = Args(**params)
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
        


    for key in table.columns.keys():
        print(key)
        print(value_to_int_dict[key])
        print("-----------------------------------")

    for i in range(10):
        print(new_data[i])
        print()

    print("row number:",len(new_data))
    print("attribute number:",len(new_data[0]))
    attr_name=list(table.columns.keys())
    print(attr_name)


    test_data = pd.DataFrame(new_data,columns=attr_name)
    correlation=test_data.corr()
    print("correlation:")
    print(correlation)

    save_data=table_data(new_data,attr_name,correlation,value_to_int_dict,attr_type_dict)
    save_data_to_file(save_data,dataset+"_"+version+".pkl")

    end_time=time.time()
    print("spent",end_time-start_time)


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
    L.info(f"Use {len(queryset['train'])} queries for train and {len(queryset['valid'])} queries for validation")
    

    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")
    #data,attr_name,correlation,value_to_int_dict
    value_to_int_dict=vec_data.value_to_int_dict
    #attr_type_dict=vec_data.attr_type_dict

    
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
    save_data_to_file(save_data,"test_"+dataset+"_"+version+"_workload.pkl")
    '''
    for i in range(2):
        print(queryset["train"][i])
        print(query_vec[i])
        print(labels['train'][i])
        print(query_label[i])
        print("------------------------------------")
    '''
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


from queue import Queue          

def fill_the_structure(root=None,new_data=None,attr_clusters=None):
    max_layer=len(attr_clusters)
    if root==None:
        root=cardinality_estimation_structure(0,None,None)
    th=0
    for d in new_data:
        if th%100000==0:
            print("has finished "+str(th)+"/"+str(len(new_data))+" updating data")
        th+=1
        root.add_new_data(d,0,attr_clusters,max_layer)
    return root


def query_info(dataset,version):
    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label
    return


def test_and_calc_p_value(root_node,attr_clusters,query_vec,query_label):

    probs=[]
    print(attr_clusters)
    start_time=time.time()
    print("start to test on "+str(len(query_label))+" queries")
    precision=0
    for i in range(len(query_vec)):
        end_time=time.time()
        if i%10000==0:
            print("spent "+str(end_time-start_time)+"s to finish "+str(i)+" queries")
        query_clusters=[]
        for j in attr_clusters:
            query_clusters.append([query_vec[i][w] for w in j])
        
        c_max_layer=len(query_clusters)

        global prob_list
        prob_list=[0 for i in range(c_max_layer)]
        cal(root_node.next1,query_clusters,0,0,0,c_max_layer)
        #root_node.calc_prob(query_clusters,0,max_layer,screen_exist)
        #print(prob_list,"  ",query_label[i])
        if prob_list[-1]==query_label[i]:
            precision+=1
        probs.append(prob_list[-1])
    print("the precision is:",precision/len(query_label))
    end_time=time.time()
    print("spent "+str(end_time-start_time)+" seconds")

    return probs


def fill_and_show_structure_test(dataset,version):
    if dataset=="draft" and version=="draft":
        new_data=[[1,2,3,4,5],[1,2,3,4,6],[1,2,4,4,5],[1,2,4,4,6],[2,4,1,2,3]]
        attr_clusters=[[0,1],[2],[3,4]]
        root_node=fill_the_structure(None,new_data,attr_clusters)
        #show_structure(root_node,attr_clusters,dataset,version)
    else:
        vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")
        
        new_data=vec_data.data
        attr_name=vec_data.attr_name
        correlation=vec_data.correlation

        new_tree=attr_tree(list(range(len(attr_name))),None,None,False,None,0.1)
        new_tree.build_the_tree(correlation,attr_name)
        
        attr_clusters=new_tree.get_the_attr_list([])
        print(attr_clusters)
        #attr_clusters=attr_clusters[3:4]
          
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
    
        workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
        query_vec=workload_data.query_vec
        query_label=workload_data.query_label

        #test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
        test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
        test_query_vec=test_workload_data.query_vec
        test_query_label=test_workload_data.query_label
        #print(query_vec)
        #print(query_label)

        ## get the testing data
        x_probs=[]
        for i in range(len(attr_clusters)):
            probs=test_and_calc_p_value(root_nodes[i],attr_clusters[i:i+1],test_query_vec,test_query_label)
            x_probs.append(probs)
        
        data_number=len(new_data)
        print(data_number)
        test_x=[]
        test_y=[]
        for i in range(len(test_query_label)):
            test_x.append([u[i]/data_number for u in x_probs])
            #print("the "+str(i)+"th: ",train_x,query_label[i]/data_number)
            test_y.append(test_query_label[i]/data_number)

        save_data_to_file(test_x,"test_x_"+dataset+"_"+version+".pkl")
        save_data_to_file(test_y,"test_y_"+dataset+"_"+version+".pkl")
        #     #print("the "+str(i)+"th: ",train_x,query_label[i])
        
        ## get the training data
        x_probs=[]
        for i in range(len(attr_clusters)):
            probs=test_and_calc_p_value(root_nodes[i],attr_clusters[i:i+1],query_vec,query_label)
            x_probs.append(probs)
        
        data_number=len(new_data)
        print(data_number)
        train_x=[]
        train_y=[]
        for i in range(len(query_label)):
            train_x.append([u[i]/data_number for u in x_probs])
            #print("the "+str(i)+"th: ",train_x,query_label[i]/data_number)
            train_y.append(query_label[i]/data_number)

        save_data_to_file(train_x,"train_x_"+dataset+"_"+version+".pkl")
        save_data_to_file(train_y,"train_y_"+dataset+"_"+version+".pkl")


import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

class LinerRegress(torch.nn.Module):
    def __init__(self,x_dimension):
        super(LinerRegress, self).__init__()
        self.fc = nn.Linear(x_dimension, 1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

def regression_part(dataset,version):
    train_x=load_data_from_pkl_file("train_x_"+dataset+"_"+version+".pkl")
    train_y=load_data_from_pkl_file("train_y_"+dataset+"_"+version+".pkl")
    test_x=load_data_from_pkl_file("test_x_"+dataset+"_"+version+".pkl")
    test_y=load_data_from_pkl_file("test_y_"+dataset+"_"+version+".pkl")

    for i in range(len(train_y)):
        train_y[i]=[train_y[i]]
    for i in range(len(test_y)):
        test_y[i]=[test_y[i]]

    train_x=torch.FloatTensor(train_x)
    train_y=torch.FloatTensor(train_y)
    test_x=torch.FloatTensor(test_x)
    test_y=torch.FloatTensor(test_y)

    # train_x=np.array(train_x)
    # train_y=np.array(train_y)
    # test_x=np.array(test_x)
    # test_y=np.array(test_y)

    print(len(train_x))
    print(len(test_x))
    print(len(train_x[0]))
    net = LinerRegress(len(train_x[0]))
    loss_func = torch.nn.MSELoss()
    optimzer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

    for i in range(50):
        q_error=[]
        print("start "+str(i)+"th iteration")
        for th in range(len(train_x)):
            # print(train_x[th])
            # print(train_y[th])
            optimzer.zero_grad()

            out = net(train_x[th])
            loss = loss_func(out, train_y[th])
            loss.backward()
            optimzer.step()

        for th in range(len(test_x)):
            out=net(test_x[th])
            
            if out>test_y[th]:
                if test_y[th]==0:
                    q_error.append(out)
                
                else:
                    q_error.append(out/test_y[th])
            else:
                if out==0:
                    q_error.append(test_y[th])
                else:
                    q_error.append(test_y[th]/out)
        for j in range(len(q_error)):
            q_error[j]=q_error[j].detach().numpy()
        print("Max:",np.max(q_error)," 99th:",np.percentile(q_error,99)," 95th:",np.percentile(q_error,95)," mean:",np.mean(q_error))



def get_graph_data(dataset,version):
    vec_data=load_data_from_pkl_file(dataset+"_"+version+".pkl")

    new_data=vec_data.data
    attr_name=vec_data.attr_name
    correlation=vec_data.correlation
    range_size=vec_data.range_size
    value_to_int_dict=vec_data.value_to_int_dict
    
    print(correlation)
    
    new_tree=attr_tree(list(range(len(attr_name))),None,None,False,None,0.05)
    new_tree.build_the_tree(correlation,attr_name)
    
    attr_clusters=new_tree.get_the_attr_list([])
    print(attr_clusters)

    # divide those attr_clusters whose size will increase exponentially when conbined together
    new_attr_clusters=[]
    for cluster in attr_clusters:
        product=1
        for i in cluster:
            product*=range_size[i]
        #print(product)
        if product>=(0.3*len(new_data)):
            for i in cluster:
                new_attr_clusters.append([i])
        else:
            new_attr_clusters.append(cluster)
    attr_clusters=new_attr_clusters
    print(attr_clusters)

    start_time=time.time()

    root_nodes=[]
    for i in range(len(attr_clusters)):
        root_node=fill_the_structure(None,new_data,attr_clusters[i:i+1])
        root_nodes.append(root_node)
    end_time=time.time()

    print("has spent "+str(end_time-start_time)+" seconds to fill the structure")
    save_data(root_nodes,"./lecarb/estimator/mine/filled_model/"+dataset+"_"+version+".pkl")

    

    ## get the testing data
    test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_query_vec=test_workload_data.query_vec
    test_query_label=test_workload_data.query_label
    
    x_probs=[]
    for i in range(len(attr_clusters)):
        probs=test_and_calc_p_value(root_nodes[i],attr_clusters[i:i+1],test_query_vec,test_query_label)
        x_probs.append(probs)

    attr_num=0
    for i in attr_clusters:
        attr_num+=len(i)
    #print(attr_num)
    #test_graph_x=[np.zeros((attr_num,attr_num)) for i in range(len(test_query_label))]
    test_graph_x=[np.ones((attr_num,attr_num)) for i in range(len(test_query_label))]
    data_number=len(new_data)
    for i in range(len(attr_clusters)):
        attrs=attr_clusters[i:i+1][0]
        for j in range(len(test_query_label)):
            for attr1 in attrs:
                for attr2 in attrs:
                    #print(i,j,attr1,attr2)
                    test_graph_x[j][attr1][attr2]=x_probs[i][j]/data_number

    test_y=[]
    for i in range(len(test_query_label)):
        test_y.append(test_query_label[i]/data_number)
    test_graph_x=np.array(test_graph_x)
    test_y=np.array(test_y)
    save_data_to_file(test_graph_x,"test_graph_x_"+dataset+"_"+version+".pkl")
    save_data_to_file(test_y,"test_y_"+dataset+"_"+version+".pkl")
    save_data_to_file(np.array(x_probs).T,"test_probs_"+dataset+"_"+version+".pkl")
    
    ## get the training data
    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label
    x_probs=[]
    for i in range(len(attr_clusters)):
        probs=test_and_calc_p_value(root_nodes[i],attr_clusters[i:i+1],query_vec,query_label)
        x_probs.append(probs)
    
    attr_num=0
    for i in attr_clusters:
        attr_num+=len(i)
    #print(attr_num)
    #train_graph_x=[np.zeros((attr_num,attr_num)) for i in range(len(query_label))]
    train_graph_x=[np.ones((attr_num,attr_num)) for i in range(len(query_label))]
    data_number=len(new_data)
    for i in range(len(attr_clusters)):
        attrs=attr_clusters[i:i+1][0]
        for j in range(len(query_label)):
            for attr1 in attrs:
                for attr2 in attrs:
                    #print(i,j,attr1,attr2)
                    train_graph_x[j][attr1][attr2]=x_probs[i][j]/data_number

    train_y=[]
    for i in range(len(query_label)):
        train_y.append(query_label[i]/data_number)
    train_graph_x=np.array(train_graph_x)
    train_y=np.array(train_y)
    save_data_to_file(train_graph_x,"train_graph_x_"+dataset+"_"+version+".pkl")
    save_data_to_file(train_y,"train_y_"+dataset+"_"+version+".pkl")

    for i in range(len(x_probs)):
        for j in range(len(x_probs[0])):
            x_probs[i][j]/=data_number
    save_data_to_file(np.array(x_probs).T,"train_probs_"+dataset+"_"+version+".pkl")

