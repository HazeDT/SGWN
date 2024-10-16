import torch
from math import sqrt
import numpy as np
from torch_geometric.data import Data
from scipy.spatial.distance import pdist
import copy

def KNN_classify(k,X_set,x):
    """
    k:number of neighbours
    X_set: the datset of x
    x: to find the nearest neighbor of data x
    """

    distances = [sqrt(np.sum((x_compare-x)**2)) for x_compare in X_set]
    nearest = np.argsort(distances)
    node_index  = [i for i in nearest[1:k+1]]
    topK_x = [X_set[i] for i in nearest[1:k+1]]
    return  node_index,topK_x


def KNN_weigt(x,topK_x):
    distance = []
    v_1 = x
    data_2 = topK_x
    for i in range(len(data_2)):
        v_2 = data_2[i]
        combine = np.vstack([v_1, v_2])
        likely = pdist(combine, 'euclidean')
        distance.append(likely[0])
    beata = np.mean(distance)
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))
    return w


def KNN_attr(data):
    '''
    for KNNgraph
    :param data:
    :return:
    '''
    edge_raw0 = []
    edge_raw1 = []
    edge_fea = []
    for i in range(len(data)):
        k = 3 #Define the k for cluster
        x = data[i]
        node_index, topK_x = KNN_classify(k,data,x)
        loal_weigt = KNN_weigt(x,topK_x)
        local_index = np.zeros(k)+i

        edge_raw0 = np.hstack((edge_raw0,local_index))
        edge_raw1 = np.hstack((edge_raw1,node_index))
        edge_fea = np.hstack((edge_fea,loal_weigt))

    edge_index = [edge_raw0, edge_raw1]

    return edge_index, edge_fea



def cal_sim(data,s1,s2):
    edge_index = [[],[]]
    edge_feature = []
    if s1 != s2:
        v_1 = data[s1]
        v_2 = data[s2]
        combine = np.vstack([v_1, v_2])
        likely = 1- pdist(combine, 'cosine')
#         w = np.exp((-(likely[0]) ** 2) / 30)
        if likely.item() >= 0:
            w = 1
            edge_index[0].append(s1)
            edge_index[1].append(s2)
            edge_feature.append(w)
    return edge_index,edge_feature



def Radius_attr(data):
    '''
    for RadiusGraph
    :param feature:
    :return:
    '''
    s1 = range(len(data))
    s2 = copy.deepcopy(s1)
    edge_index = np.array([[], []])  # 一个故障样本与其他故障样本匹配生成一次图
    edge_fe = []
    for i in s1:
        for j in s2:
            local_edge, w = cal_sim(data, i, j)
            edge_index = np.hstack((edge_index, local_edge))
            if any(w):
                edge_fe.append(w[0])
    return edge_index,edge_fe


def Path_attr(data):

    node_edge = [[], []]

    for i in range(len(data) - 1):
        node_edge[0].append(i)
        node_edge[1].append(i + 1)

    distance = []
    for j in range(len(data) - 1):
        v_1 = data[j]
        v_2 = data[j + 1]
        combine = np.vstack([v_1, v_2])
        likely = pdist(combine, 'euclidean')
        distance.append(likely[0])

    beata = np.mean(distance)
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))  #Gussion kernel高斯核

    return node_edge, w


def Gen_graph(graphType, data, label,task):
    data_list = []
    if graphType == 'KNNGraph':
        for i in range(len(data)):
            graph_feature = data[i]
            if task == 'Node':
                labels = np.zeros(len(graph_feature)) + label
            elif task == 'Graph':
                labels = [label]
            else:
                print("There is no such task!!")
            node_edge, w = KNN_attr(data[i])
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)

    elif graphType == 'RadiusGraph':
        for i in range(len(data)):
            graph_feature = data[i]
            if task == 'Node':
                labels = np.zeros(len(graph_feature)) + label

            elif task == 'Graph':
                labels = [label]
            else:
                print("There is no such task!!")
            node_edge, w = Radius_attr(graph_feature)
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)

    elif graphType == 'PathGraph':
        for i in range(len(data)):
            graph_feature = data[i]
            if task == 'Node':
                labels = np.zeros(len(graph_feature)) + label
            elif task == 'Graph':
                labels = [label]
            else:
                print("There is no such task!!")
            node_edge, w = Path_attr(graph_feature)
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)  # 获得图标签
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)

    else:
        print("This GraphType is not included!")
    return data_list
