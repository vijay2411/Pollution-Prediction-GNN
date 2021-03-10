import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import random
import datetime
from haversine import haversine, Unit
import argparse
import pickle

#gives you the weighted edges, 3xn
def getEdges(data, spaceTH = 0.5, timeTH = 7, spacePower = 2, timePower = 0.5, minEdges = 4, verbose = False):
#----------------------------------------------2-3Km in space, 1-2.5hrs in time----------------------------------------

    spaceD = data[:,1:3]
    timeD = data[:,0]
    dist = 1.0 #in Km, Time
    edges = []
    for i in range(len(data)):
        if(verbose):
            print(i)
        lim = spaceTH
        count = 0
        #keep track of nodes that have been added from a certain location onto the target node
        notebook ={}
        while(count<minEdges):
            #considering nodes of same timeZone
            for j in range(i+1,len(data)):
                hString = np.array_str(spaceD[j])
                if hString in notebook:
                    continue
                di = haversine(spaceD[i],spaceD[j])
                ti = int((timeD[i]-timeD[j])/14.9)
                w = ((1+di)**spacePower)*(1+ti**timePower)
                if ti<0:
                    break
                if di<=lim:
                    edges.append([j,i,1.0/w])
                    count+=1
                    notebook[hString] = True
                    
            #nodes from past
            for j in range(i-1,-1,-1):
                hString = np.array_str(spaceD[j])
                if hString in notebook:
                    continue
                di = haversine(spaceD[i],spaceD[j])
                ti = int((timeD[i]-timeD[j])/14.9)
                w = ((1+di)**spacePower)*(1+ti**timePower)
                if ti>timeTH:
                    break
                if di<=lim:
                    edges.append([j,i,1.0/w])
                    count+=1
                    notebook[hString] = True
            lim+=0.1
            
    edges = np.array(edges).T
    return edges


# Find Normalization coefficients for edges in weighted graph sage
#-----------------------------------------------------------------------------------------------------------------------
def findNorm(tempEdges, x_size):
    weightSum = np.ones(x_size)
    ei = pd.DataFrame(tempEdges, columns = ['from','to','val'])
    ei = ei.groupby(['to']).agg('sum').reset_index()
    ei = ei[['from','to','val']]
    weightSum[ei.to.to_numpy().astype(int)] = ei.val.to_numpy()
    weightSum = torch.from_numpy(weightSum)
    normalize = torch.zeros(tempEdges.shape[0])
    for i,edge in enumerate(tempEdges):
        if(weightSum[int(edge[1])] == 0):
            print("Error in WeightedSageConv, as one of the nodes has no incoming edges to it.")
        normalize[i] = edge[2]*1.0/weightSum[int(edge[1])]    
    return normalize


# Create Dataset/Subgraphs for training. Partition nodes into known/unknown.
#------------------------------------------ Unknown Samples = train Data * 1/10 ---------------------------------------------
def createDataset(d: np.array, edges: np.array, numGraphs:int = 50, device = 'cpu', percent = 0.1):
    #d is PM10 values of train set, edges are edge list on this train set
    dataset = []
    trainSamples = int(len(d)*percent)
    count = 0
    while(count<numGraphs):
        trainSet = random.sample(range(len(d)), trainSamples)
        trainSet = np.sort(trainSet)
        trainSet = torch.from_numpy(trainSet)
        #proper subset making and reshaping
        #removing edges coming out of validation nodes
        tempEdges = edges[:,[i not in trainSet for i in edges[0]]]
        #tempEdges = tempEdges[:, tempEdges[1].argsort()]
        subWeights = torch.reshape(torch.tensor(tempEdges[2,:], dtype = torch.float),(tempEdges.shape[1],1))
        subEdges = torch.tensor(tempEdges[:2,:], dtype =torch.long)
        
        #2 features of nodes, 1) PM10 value, 2)Presence
        subNodes = np.ones((d.shape[0],d.shape[1]+1))
        subNodes[:,:d.shape[1]] = d.copy()
        subNodes[trainSet,0] = 0.0
        subNodes[trainSet,-1] = 0.0
        subNodes = torch.tensor(subNodes, dtype = torch.float)
        
        #print(d.shape[0])
        #normalization calculation
        norm = findNorm(tempEdges.T, d.shape[0])
        sample = Data(x = subNodes, edge_index = subEdges, train_set = trainSet, edge_attr = subWeights, norm = norm)
        sample.to(device)
        if((not sample.contains_isolated_nodes())):
            dataset.append(sample)
            print(count, end=" ")
            count+=1
        else:
            print("waste")
    return dataset


#Mean Finder, First One is weighted Mean, Zeroth one is Mean Model, takes an object of torch_geometric.data
#------------------------------------------------------------------------------------------------------------------------
def Meaner1(dt):
    #return torch tensor
    ei = dt.edge_index.to('cpu')
    if(type(ei) ==torch.Tensor):
        ei = ei.numpy()
    w = dt.norm.to('cpu')
    if(type(w) == torch.Tensor):
        w = w.numpy()
    n = dt.x.to('cpu')
    if(type(n) == torch.Tensor):
        n = n.numpy()
    #print(ei.shape, w.shape, n.shape)
    if(n.ndim == 2):
        n = n[:,0]
    predVal = torch.zeros(n.shape[0], dtype = float)
    for i in range(ei.shape[1]):
        predVal[ei[1][i]] += n[ei[0][i]]*w[i]
    return predVal

def Meaner0(dt):
    #return torch tensor
    ei = dt.edge_index.to('cpu')
    if(type(ei) ==torch.Tensor):
        ei = ei.numpy()
    w = dt.norm.to('cpu')
    if(type(w) == torch.Tensor):
        w = w.numpy()
    n = dt.x.to('cpu')
    if(type(n) == torch.Tensor):
        n = n.numpy()
    #print(ei.shape, w.shape, n.shape)
    if(n.ndim == 2):
        n = n[:,0]
    predVal = torch.zeros(n.shape[0], dtype = float)
    numEdges = torch.zeros(n.shape[0], dtype = int)
    for i in range(ei.shape[1]):
        predVal[ei[1][i]] += n[ei[0][i]]
        numEdges[ei[1][i]] += 1
    predVal = predVal/numEdges
    return predVal  

