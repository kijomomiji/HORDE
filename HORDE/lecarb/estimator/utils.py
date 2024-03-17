import csv
import ray
import logging
import numpy as np
import pandas as pd
import torch
from scipy.stats.mstats import gmean

#  from .lw.lw_nn import LWNN
#  from .lw.lw_tree import LWTree
from .estimator import Estimator
from ..constants import NUM_THREADS, RESULT_ROOT
from ..workload.workload import load_queryset, load_labels
from ..dataset.dataset import load_table

L = logging.getLogger(__name__)

def report_model(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    L.info(f'Number of model parameters: {num_params} (~= {mb:.2f}MB)')
    L.info(model)
    return mb

def qerror(est_card, card):
    if est_card == 0 and card == 0:
        return 1.0
    if est_card == 0:
        return card
    if card == 0:
        return est_card
    if est_card > card:
        return est_card / card
    else:
        return card / est_card

def rmserror(preds, labels, total_rows):
    return np.sqrt(np.mean(np.square(preds/total_rows-labels/total_rows)))

def evaluate(preds, labels, total_rows=-1):
    errors = []
    for i in range(len(preds)):
        errors.append(qerror(float(preds[i]), float(labels[i])))

    metrics = {
        'max': np.max(errors),
        '99th': np.percentile(errors, 99),
        '95th': np.percentile(errors, 95),
        '90th': np.percentile(errors, 90),
        'median': np.median(errors),
        'mean': np.mean(errors),
        'gmean': gmean(errors)
    }

    if total_rows > 0:
        metrics['rms'] = rmserror(preds, labels, total_rows)
    L.info(f"{metrics}")
    return np.array(errors), metrics

def evaluate_errors(errors):
    metrics = {
        'max': np.max(errors),
        '99th': np.percentile(errors, 99),
        '95th': np.percentile(errors, 95),
        '90th': np.percentile(errors, 90),
        # 'median': np.median(errors),
        'mean': np.mean(errors),
        # 'gmean': gmean(errors)
    }
    L.info(f"{metrics}")
    with open("census13_original_naru_model.txt", "a", encoding='utf-8') as f:
        f.write(f"{metrics}")
    return metrics

def report_errors(dataset, result_file):
    df = pd.read_csv(RESULT_ROOT / dataset / result_file)
    evaluate_errors(df['error'])

def report_dynamic_errors(dataset, old_new_file, new_new_file, max_t, current_t):
    '''
    max_t: Time limit for update
    current_t: Model's update time.
    old_new_path: Result file of applying stale model on new workload
    new_new_path: Result file of applying updated model on new workload
    '''
    old_new_path = RESULT_ROOT / dataset / old_new_file
    new_new_path = RESULT_ROOT / dataset / new_new_file
    if max_t > current_t:
        try:
            o_n = pd.read_csv(old_new_path)
            n_n = pd.read_csv(new_new_path)
            assert len(o_n) == len(n_n), "In current version, the workload test size should be same."
            o_n_s = o_n.sample(frac = current_t / max_t)
            n_n_s = n_n.sample(frac = 1 - current_t / max_t)
            mixed_df = pd.concat([o_n_s, n_n_s], ignore_index=True, sort=False)
            return evaluate_errors(mixed_df['error'])
        except OSError:
            print('Cannot open file.')
    return -1

def lazy_derive(origin_result_file, result_file, r, labels):
    L.info("Already have the original result, directly derive the new prediction!")
    df = pd.read_csv(origin_result_file)
    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'error', 'predict', 'label', 'dur_ms'])
        for index, row in df.iterrows():
            p = np.round(row['predict'] * r)
            l = labels[index].cardinality
            writer.writerow([int(row['id']), qerror(p, l), p, l, row['dur_ms']])
    L.info("Done infering all predictions from previous result")


#------------------------------
import time
import pickle
def load_data_from_pkl_file(file_name):
    load_addr="./lecarb/estimator/mine/vec_data/"+file_name
    with open(load_addr, 'rb') as f:
        data = pickle.load(f)
    print("has successfully loaded data from "+load_addr)
    return data

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


def test_and_calc_p_value(data_quantity,root_nodes,attr_clusters,query_vec,query_label,cards,errors):
    probs=[]
    
    # start_time=time.time()
    print("start to test on "+str(len(query_label))+" queries")
    # precision=0

    error_1=[]
    error_2=[]
    time_1=[]
    time_2=[]  
    turn_to_precise1=0
    turn_to_precise2=0
    for i in range(len(query_vec)):
        #print(i,query_vec[i])
        # start_time1=time.time()
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
        prob_list=[0 for u in range(c_max_layer)]
        

        # if the approximation of cardinality is relatively small, we choose precise and swift method
         
        if cards[i]<(data_quantity*0.1):
            # global prob_list
            # print(prob_list[-1],data_quantity*0)
            start_time=time.time()
            turn_to_precise1+=1
            prob_list=[0 for i in range(c_max_layer)]
            cal(root_nodes[start_pos].next1,query_clusters,0,0,0,c_max_layer)
            end_time=time.time()
            error_1.append(1)
            time_1.append(end_time-start_time)
        else:
            error_1.append(errors[i])
            time_1.append(0)
        

        if cards[i]<(data_quantity*0.01):
            # global prob_list
            # print(prob_list[-1],data_quantity*0)
            start_time=time.time()
            turn_to_precise2+=1
            prob_list=[0 for i in range(c_max_layer)]
            cal(root_nodes[start_pos].next1,query_clusters,0,0,0,c_max_layer)
            end_time=time.time()
            error_2.append(1)
            time_2.append(end_time-start_time)
        else:
            error_2.append(errors[i])
            time_2.append(0)
        

        probs.append(prob_list[-1])

    # print("the precision is:",precision/len(query_label))
    print("0.1 experiment")
    print("turn to precise",turn_to_precise1)
    evaluate_errors(error_1)
    print(sum(time_1))
    
    print("0.01 experiment")
    print("turn to precise",turn_to_precise2)
    evaluate_errors(error_2)
    print(sum(time_2))

    return probs

#----------------------------


def run_test(dataset: str, version: str, workload: str, estimator: Estimator, overwrite: bool, lazy: bool=True, lw_vec=None, query_async=False) -> None:
    # for inference speed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")

    L.info(f"Start loading queryset:{workload} and labels for version {version} of dataset {dataset}...")
    
    # for test or valid
    for load_type in ['valid','test']: 
        queries = load_queryset(dataset, workload)[load_type]
        labels = load_labels(dataset, version, workload)[load_type]

        if lw_vec is not None:
            X, gt = lw_vec
            #  assert isinstance(estimator, LWNN) or isinstance(estimator, LWTree), estimator
            assert len(X) == len(queries), len(X)
            assert np.array_equal(np.array([l.cardinality for l in labels]), gt)
            L.info("Hack for LW's method, use processed vector instead of raw query")
            queries = X

        # prepare file path, do not proceed if result already exists
        result_path = RESULT_ROOT / f"{dataset}"
        result_path.mkdir(parents=True, exist_ok=True)
        csv_name=f"{version}-{workload}-{estimator}.csv".replace(";","_")
        #result_file = result_path / f"{version}-{workload}-{estimator}.csv"
        result_file = result_path / csv_name
        print("the saving address is:",result_file)

        # if not overwrite and result_file.is_file():
        #     L.info(f"Already have the result {result_file}, do not run again!")
        #     exit(0)

        r = 1.0
        if version != estimator.table.version:
            test_row = load_table(dataset, version).row_num
            r = test_row / estimator.table.row_num
            L.info(f"Testing on a different data version, need to adjust the prediction according to the row number ratio {r} = {test_row} / {estimator.table.row_num}!")

            origin_result_file = RESULT_ROOT / dataset / f"{estimator.table.version}-{workload}-{estimator}.csv"
            if lazy and origin_result_file.is_file():
                return lazy_derive(origin_result_file, result_file, r, labels)

        if query_async:
            L.info("Start test estimator asynchronously...")
            for i, query in enumerate(queries):
                estimator.query_async(query, i)

            L.info('Waiting for queries to finish...')
            stats = ray.get([w.get_stats.remote() for w in estimator.workers])

            errors = []
            latencys = []
            
            
            with open(result_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'error', 'predict', 'label', 'dur_ms'])
                for i, label in enumerate(labels):
                    r = stats[i%estimator.num_workers][i//estimator.num_workers]
                    assert i == r.i, r
                    error = qerror(r.est_card, label.cardinality)
                    # print(r.est_card,label.cardinality)
                    errors.append(error)
                    latencys.append(r.dur_ms)
                    writer.writerow([i, error, r.est_card, label.cardinality, r.dur_ms])

            L.info(f"Test finished, {np.mean(latencys)} ms/query in average")
            evaluate_errors(errors)
            return

        L.info("Start test estimator on test queries...")
        errors = []
        latencys = []
        cards=[]
        queries=queries[:]
        labels=labels[:]

        prediction=[]
        p_labels=[]

        total_num=load_table(dataset, version).row_num
        
        with open(result_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'error', 'predict', 'label', 'dur_ms'])
            for i, data in enumerate(zip(queries, labels)):
                query, label = data
                
                
                est_card, dur_ms = estimator.query(query)

                # print("est_card:",type(est_card),"   label.cardinality",type(label.cardinality))
                
                prediction.append(est_card)
                p_labels.append(label.cardinality)

                est_card = np.round(r * est_card)

                error = qerror(est_card, label.cardinality)
                
                errors.append(error)
                cards.append(est_card)

                latencys.append(dur_ms)
                writer.writerow([i, error, est_card, label.cardinality, dur_ms])
                if (i+1) % 1000 == 0:
                    L.info(f"{i+1} queries finished")
        L.info(f"Raw Test finished, {np.mean(latencys)} ms/query in average")
        prediction=np.array(prediction)
        # print("prediction",prediction)
        # print("length  prediction",len(prediction))


        if 'naru' in str(estimator):
            estimator_name='naru'
        elif 'mscn' in str(estimator):
            estimator_name='mscn'
        elif 'spn' in str(estimator):
            estimator_name='deepdb'
        else:
            print('wrong')
            return

        print(estimator_name)
        
        addr="./lecarb/estimator/predict_result/"+estimator_name+"_model_prediction/"+load_type+"_"+dataset+"_"+version+".pkl"
        
        with open(addr, 'wb') as f:
            pickle.dump([prediction,p_labels,total_num], f)
            print("the vec data has been stored in "+addr)

        print("q_error original model")
        errors=[]
        for i in range(len(prediction)):
            errors.append(qerror(prediction[i],p_labels[i]))
        evaluate_errors(errors)
    
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

        print("---------------------------------------------")
        print("q_error original model+ACCT")

        if load_type=="test":
            result_addr="./lecarb/estimator/mine/tree_inference_result/"+dataset+"_"+version+".pkl"
        elif load_type=='valid':
            continue
            # result_addr="./lecarb/estimator/mine/tree_inference_result/valid_"+dataset+"_"+version+".pkl"
        
        with open(result_addr, 'rb') as f:
            [inference_result,inference_time] = pickle.load(f)


    
        errors=[]
        thres=None
        if estimator_name=="mscn":
            if dataset=='census13':
                eta=0.0021732061805777433
            elif dataset=='forest10':
                eta=0.0
            elif dataset=='power7':
                eta=1.0484484391781734e-09
            elif dataset=='dmv11':
                eta=0.0004951753423341053
        elif estimator_name=='deepdb':
            if dataset=='census13':
                eta=0.008443939625411012
            elif dataset=='forest10':
                eta=0.0
            elif dataset=='power7':
                eta=0.00046214956543683133
            elif dataset=='dmv11':
                eta=0.0015720891266910683
        elif estimator_name=='naru':
            if dataset=='census13':
                eta=0.0
            elif dataset=='forest10':
                eta=3.247180302423658e-12
            elif dataset=='power7':
                eta=1.4526335689879488e-10
            elif dataset=='dmv11':
                eta=0.0

        
        thres=total_num*eta
        
        forest_increase_time=[0 for i in range(len(inference_time))]
        turn_to_precise=0
        for i in range(len(prediction)):
            if prediction[i]<thres:
                turn_to_precise+=1
                prediction[i]=inference_result[i]
                latencys[i]+=inference_time[i]*1000 # s->ms

                forest_increase_time[i]+=inference_time[i]*1000 # s->ms
        print("eta:",eta," turn to precise:",turn_to_precise)
        L.info(f"forest increase time, {np.mean(forest_increase_time)} ms/query in average")
        
        L.info(f"Raw & Tree Test finished, {np.mean(latencys)} ms/query in average")

        for i in range(len(prediction)):
            errors.append(qerror(prediction[i],p_labels[i]))
        evaluate_errors(errors)
    
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

