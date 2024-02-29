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
import pandas as pd
import numpy as np
import csv
from collections import Counter
import bisect
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils import data
import copy

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

