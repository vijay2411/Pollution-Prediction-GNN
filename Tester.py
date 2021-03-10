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

from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing

#trainAccuracy is determined using the net and dataset.
def trainAccuracyCheck(net, dataset,verbose = False):
    sumNN, sumMean, sumMean0 = 0, 0, 0
    #graph = Data(x = torch.from_numpy(X), edge_index = torch.from_numpy(edges[:2,:]).type(torch.LongTensor), norm = findNorm(torch.from_numpy(edges.T),len(X)))
    outMean = Meaner1(graph)
    outMean0 = Meaner0(graph)
    for i in range(len(dataset)):
        out = net(dataset[i])
        out = torch.reshape(out,(-1,))
        lossOut = F.mse_loss(out, X1)**0.5
        lossOutVal = F.mse_loss(out[dataset[i].train_set.type(torch.LongTensor)], X1[dataset[i].train_set.type(torch.LongTensor)])**0.5
        meanLossOut = F.mse_loss(outMean, X2)**0.5
        meanLossOutVal = F.mse_loss(outMean[dataset[i].train_set.type(torch.LongTensor)], X2[dataset[i].train_set.type(torch.LongTensor)])**0.5
        mean0LossOutVal = F.mse_loss(outMean0[dataset[i].train_set.type(torch.LongTensor)], X2[dataset[i].train_set.type(torch.LongTensor)])**0.5
        sumNN+=lossOutVal.item()
        sumMean+=meanLossOutVal.item()
        sumMean0+=mean0LossOutVal.item()
        if(verbose):
            print("Iter ", i)
            print("On Whole Set: ", lossOut.item(),"(NN), ", meanLossOut.item(),"(Just Mean)")
            print("On Train Set: ", lossOutVal.item(),"(NN), ", meanLossOutVal.item(),"(Just Mean)")
            print(" ")
    print("Mean Loss by NN on validation sets: ", sumNN/len(dataset))
    print("Mean Loss by Weighted MeanEstimate on validation sets: ", sumMean/len(dataset))
    print("Mean Loss by Mean Estimate on validation sets: ", sumMean0/len(dataset))


#just a helper function
def findBS(data, val):
    l,r = np.searchsorted(data[:,0],val[0],side='left'), np.searchsorted(data[:,0],val[0],side = 'right')
    ll, rr = np.searchsorted(data[l:r][:,1], val[1], side='left'), np.searchsorted(data[l:r][:,1], val[1], side='right')
    lll = np.searchsorted(data[l:r][ll:rr][:,2], val[2], side = 'left')
    return l+ll+lll

#Same as getEdges, which work for traingraph, the difference is it connects only to the nodes of trian graph, test nodes are not interlinked.
# the test node value is obtained by passing the neighbour nodes value thorough the net that we jsut trained.
def prepareTestData(trainData, testData, edges, spaceTH = 0.5, timeTH = 7, spacePower = 2, timePower = 0.5, minEdges = 4, verbose = False):
    finalData = np.concatenate((trainData,testData), axis = 0)
    spaceD = finalData[:,1:3]
    timeD = finalData[:,0]
    finalEdges = edges.T
    trainLen = len(trainData)
    newEdges = []
    for i in range(len(testData)):
        if(verbose):
            print(i)
        lim = spaceTH
        count = 0
        #keep track of nodes that have been added from a certain location onto the target node
        notebook ={}
        I = findBS(trainData, testData[i])
        while(count<minEdges):
            #considering nodes of same timeZone
            for j in range(I+1,len(trainData)):
                hString = np.array_str(spaceD[j])
                if hString in notebook:
                    continue
                di = haversine(spaceD[i+trainLen],spaceD[j])
                ti = int((timeD[i+trainLen]-timeD[j])/14.9)
                w = ((1+di)**spacePower)*(1+ti**timePower)
                if ti<0:
                    break
                if di<=lim:
                    newEdges.append([j,i+trainLen,1.0/w])
                    count+=1
                    notebook[hString] = True
                    
            #nodes from past
            for j in range(I,-1,-1):
                hString = np.array_str(spaceD[j])
                if hString in notebook:
                    continue
                di = haversine(spaceD[i+trainLen],spaceD[j])
                ti = int((timeD[i+trainLen]-timeD[j])/14.9)
                w = ((1+di)**spacePower)*(1+ti**timePower)
                if ti>timeTH:
                    break
                if di<=lim:
                    newEdges.append([j,i+trainLen,1.0/w])
                    count+=1
                    notebook[hString] = True
            lim+=0.1
    
    newEdges = np.array(newEdges)
    finalEdges = np.concatenate((finalEdges, newEdges), axis = 0)

    #in the following the pollutant value is brought forward, i.e. data change looks something like this. time, lat, long, pollutant -> pollutant, time, lat, long
    return np.concatenate((finalData[:,3].reshape((finalData.shape[0],1)), finalData[:,:3].reshape((finalData.shape[0],3))), axis = 1),finalEdges.T

#This one preparest the whole test set, remember 
def prepareTestGraph(data, testData, edges, device, minEdges = 4):
    #set verbose false to stop printing
    #Xf is the final data(with pollutant value in first column from left), ef is edges for test graph. 
    Xf, ef = prepareTestData(data, testData, edges, spaceTH=args.EdgeR, timeTH= args.EdgeTR, spacePower= args.SpacePower, timePower= args.TimePower, verbose = False, minEdges = minEdges )
    #Xf = np.reshape(Xf,(Xf.shape[0],1))
    Xf = np.concatenate((Xf,np.ones((Xf.shape[0],1))), axis = 1)
    #declaring test nodes as test nodes(last index from left) and setting their pollutatnt value as zero, so that they would be filled with predicted value(first index from left)
    Xf[-(len(testData)):,0] = 0.0
    Xf[-(len(testData)):,-1] = 0.0
    test_set = range(data.shape[0], Xf.shape[0])
    ei = torch.tensor(ef[:2,:], dtype= torch.long)
    ef = ef.astype('float64')
    nm = findNorm(ef.T, Xf.shape[0])
    testGraph = Data(x = torch.from_numpy(Xf), edge_index = ei,
                test_set = test_set, norm = nm)
    testGraph.to(device)
    return testGraph