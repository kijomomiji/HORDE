import pickle
import time
from ...dataset.dataset import load_table
def load_data_from_pkl_file(file_name):
    load_addr="./lecarb/estimator/mine/vec_data/"+file_name
    with open(load_addr, 'rb') as f:
        data = pickle.load(f)
    print("has successfully loaded data from "+load_addr)
    return data

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
    # th=0
    for d in new_data:
        # if (th+1)%100000==0:
        #     print("has finished "+str(th+1)+"/"+str(len(new_data))+" updating data")
        # th+=1
        root.add_new_data(d,0,attr_clusters,max_layer)
    return root



def update_label(test_query_vec,test_query_label,attr_clusters,real_root_nodes):
    update_c=[]
    for i in range(len(test_query_vec)):
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
        
        c_max_layer=len(query_clusters)
        # print(c_max_layer)
        # print(i,query_clusters)

        global prob_list
        prob_list=[0 for w in range(c_max_layer)]
        cal(real_root_nodes[start_pos].next1,query_clusters,0,0,0,c_max_layer)
        # print("prediction",prediction[i],"problist[-1]",prob_list[-1],"p_labels[i]",p_labels[i])
        update_c.append(test_query_label[i]+prob_list[-1])
    return update_c

class workload:
    def __init__(self,query_vec,query_label):
        self.query_vec=query_vec
        self.query_label=query_label

def get_updated_workload(dataset,version):
    new_version='original+original_cor_0.5'
    table = load_table(dataset, version)

    for i in range(table.col_num):
        key=list(table.columns.keys())[i]
        print(i,table.columns[key].name,table.columns[key].vocab_size,table.columns[key].dtype)

    
    old_data=load_data_from_pkl_file(dataset+"_"+version+".pkl").data #class table_data
    print(len(old_data),"tuples")
    new_data=load_data_from_pkl_file(dataset+"_"+new_version+".pkl").data
    print(len(new_data),'tuples')

    update_data=new_data[len(old_data):]
    print(len(update_data),'tuples')

    # return
    
    attr_clusters=[[i] for i in range(len(old_data[0]))]
    cluster_ranges=[]
    for cluster in attr_clusters:
        total_num=1
        for i in cluster:
            key = list(table.columns.keys())[i]
            total_num*=(table.columns[key].vocab_size)
        cluster_ranges.append(total_num)
    print(cluster_ranges)
    # sort the attr_clusters so that big range clusters are placed behind small ones 
    attr_clusters = [i for _,i in sorted(zip(cluster_ranges,attr_clusters))]
    for cluster in attr_clusters:
        total_num=1
        for i in cluster:
            key = list(table.columns.keys())[i]
            total_num*=(table.columns[key].vocab_size)
    print(attr_clusters)
    
    print(cluster_ranges)
    
    workload_data=load_data_from_pkl_file(dataset+"_"+version+"_workload.pkl")
    query_vec=workload_data.query_vec
    query_label=workload_data.query_label

    test_workload_data=load_data_from_pkl_file("test_"+dataset+"_"+version+"_workload.pkl")
    test_query_vec=test_workload_data.query_vec
    test_query_label=test_workload_data.query_label
    


    real_root_nodes=[]
    start_time=time.time()
    for i in range(len(attr_clusters)):
        real_root_node=fill_the_structure(None,update_data,attr_clusters[i:])
        real_root_nodes.append(real_root_node)
    end_time=time.time()
    print("has spent "+str(end_time-start_time)+" seconds to fill the structure")

    
    update_test_label=update_label(test_query_vec,test_query_label,attr_clusters,real_root_nodes)
    save_data=workload(test_query_vec,update_test_label)
    addr="./lecarb/estimator/mine/vec_data/"+"test_"+dataset+"_"+new_version+"updated_workload.pkl"
    with open(addr, 'wb') as f:
        pickle.dump(save_data, f)
    print("the vec data has been stored in "+addr)


    update_train_label=update_label(query_vec,query_label,attr_clusters,real_root_nodes)
    save_data=workload(query_vec,update_train_label)
    addr="./lecarb/estimator/mine/vec_data/"+dataset+"_"+new_version+"updated_workload.pkl"
    with open(addr, 'wb') as f:
        pickle.dump(save_data, f)
    print("the vec data has been stored in "+addr)

    # for i in range(5):
    #     print(query_vec[i],query_label[i])
    #     print(query_vec[i],update_train_label[i])
    #     print("---------------")
