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

import * from WeightedSAGEConv
import * from ModelCreator
import * from Trainer
import * from Tester
import * from DataProcessor

parser = argparse.ArgumentParser(description='GNN')
parser.add_argument('-f', '--file', type = str, required = True, help="Input Data file name, strictly adhering to specific inpput files relevant to this project.")
parser.add_argument('-nr', '--NodeR', type=float, default=0.2,  help="Node Radius in Km.")
parser.add_argument('-ntr', '--NodeTR', type=int, default=15, help="Node Time Radius in Minutes")
parser.add_argument('-er', '--EdgeR', type= float, default=0.5, help="Edge Condition Distance in Km.")
parser.add_argument('-etr', '--EdgeTR', type=int, default=7, help="Edge Upto how much multiples of time, an int value {* will be multiplied to Node Time Radius *}")
parser.add_argument('-tdb', '--TDBSize', type= int, default=100, help="Random samples of Train database(random walk) that should be considered.")
parser.add_argument('-sp', '--SpacePower', type=float, default=3, help="Power of distance in Weight function. A +ve real.")
parser.add_argument('-tp', '--TimePower', type= float, default=0.5, help="Power of time in Weight function. A +ve real.")
parser.add_argument('-tf', '--TrainRatio', type= float, default=0.7, help="Ratio of Train data/Total data we have")
parser.add_argument('-m', '--ModelName', type= str, required = True, help="Name of the model to be saved.")
parser.add_argument('-epoch','--epoch', type = int, default = 4000, help = "Number of epoch to train.")
parser.add_argument('-d','--SpecificData', type = str, required= False, help ="[IGNORED] Would contain test indices when we were tryin to match with another model.")

#OFC this code takes only one day of data, while training
#IF you want code that can train on multiple days, contact author.
args = parser.parse_args(['-f','data28.csv','-m','E5_7_28TimeXY'])

df = readPollutionData(args.file, nr = args.NodeR)
hourlyDistribution(df, True)

# The following is there when we have indices of train and test data.
# # with open(args.SpecificData) as handle:
# #     df_dict = pickle.load(handle)
# testIndices = np.load(args.SpecificData)
# print("TestIndices shape: ", testIndices.shape)

# data = df.copy()
# testData = data.iloc[testIndices.tolist()]
# data = data.drop(testIndices.tolist())
# data, testData = roundOff(data, roundTime=str(args.NodeTR)+'min'), roundOff(testData, roundTime=str(args.NodeTR)+'min')
# data, testData = timeShredding(data, -1, 1500), timeShredding(testData, -1, 1500)
# print("data, testData prepared")

#-------------------------
#[GENERAL]The following is the routine where we are not comparing with other model, and we get to decide our indices.
meaned = roundOff(df, roundTime=str(args.NodeTR)+'min')
#print(meaned)
data = timeShredding(meaned)
#print(len(data))
testData, data = testSampling(data, 1.0 - args.TrainRatio, takeFromCopy=False)
#print(len(data))


#--------------------------
#set verbose false to stop printing
edges = getEdges(data, spaceTH=args.EdgeR, timeTH= args.EdgeTR, spacePower= args.SpacePower, timePower= args.TimePower, verbose = True, minEdges = 6)
edges = edges.astype('float64')
#print("Edges Ready: ", edges.shape)


#--------------------------
#Bringing pollutant value up in front(numpy array), four columns, pollutant, time(in minutes), lat, long
#As you can see the code is not feasible for multiple pollutants, you might need help from author if that's what you are trying to do.
X = np.concatenate((data[:,3].reshape((data.shape[0],1)),data[:,:3].reshape((data.shape[0],3))), axis = 1)
print(X),print(len(X))
#tensor version of same data
X2 = torch.from_numpy(X)
#what is norm? norm is the normalization value of edges. Basically edge weights are not normalized for each node. They were absolute value uptil now. Now, they will be normalized.
graph = Data(x = X2, edge_index = torch.from_numpy(edges[:3,:]).type(torch.LongTensor), norm = findNorm(edges.T,len(X)))

#Remember "CUDA IS MANDATORY", the graph would not train if there will be no cuda(100x slower, it already takes 4hours+ on cuda.)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("Data graph ready, Device: ", dev)


#-----------------------------
#debugging
mg = Meaner1(graph)
mg0 = Meaner0(graph)
#print("Rmse of Weighted mean on complete data: ", F.mse_loss(mg,X2)**0.5)
#print("Rmse of mean on complete data: ", F.mse_loss(mg0,X2)**0.5)

#debugging 2
ei = graph.edge_index*1
wi = graph.norm
ei,wi = edges[:, ei[1,:] == 1],wi[ei[1,:] == 1]
ei.T,wi


#------------------------------------Assuming Data is nx4 where first column is pollutant value-------------------------------
# [SKIPPABLE ISSUE] so here is a "NOT SO IMPORTANT" issue, in dataset, a random walk is created and outgoing edges from selected validation nodes(which are percent= 1.0 - args.TrainRatio in percent)
# are closed. Now, that creates sparsity even more and some of the nodes become so empty that they have no incoming nodes coming to them. Such random walk data is discared
# Now as the ratio goes less and less(not feasible below 0.6), occcurences of such cases would increase tremendously, there are solutions to this, which can be discussed.
# [IF TAKES infite TIME] -> the above issue has come into play.
dataset = createDataset(X,edges,device = dev, numGraphs = args.TDBSize, percent= 1.0 - args.TrainRatio)


#----------------------------
device = dev
X2 = torch.from_numpy(X[:,0])
X1 = X2.to(device)
X1.shape
device


#-----------------------------
testGraph = prepareTestGraph(data,testData,edges, device, minEdges = 6)
test_set = range(data.shape[0], data.shape[0]+testData.shape[0])
#print("Test graph ready")



#----------------------------
#set verbose -1 to stop printing
model,loss = iterate(dataset, device, testGraph, testData,  epochs = 5000, verbose = 100, valCheckEpochs= 1000)
model.train()
torch.save(model.state_dict(), args.ModelName+"time_lat_long.pt")
net = model



#-----------------------------
model.eval()
out = model(testGraph)
outMean = Meaner1(testGraph)
outMean0 = Meaner0(testGraph)
tdt = torch.from_numpy(testData[:,3])
print("Loss by GNN model on test Set: ", F.mse_loss(out[test_set], tdt.to(device))**0.5)
print("Loss by weighted mean model on test Set: ", F.mse_loss(outMean[test_set], tdt)**0.5)
print("Loss by mean model on test Set: ", F.mse_loss(outMean0[test_set], tdt)**0.5)



#-------------------------------------EXTRAS-------------------------------------
out.to(device)
outPy = out.to('cpu').detach().numpy()
outPy = np.resize(outPy,(outPy.shape[0],1))
prediction = np.concatenate((testData,outPy[test_set]),axis=1)
p2 = np.concatenate((prediction,abs(prediction[:,3]-prediction[:,4]).reshape((prediction.shape[0],1))), axis = 1)
pdf = pd.DataFrame(data=p2, columns = ['time','lat','long','pol','predPol','diff'])
pdff = pdf.groupby(['lat','long'])
pdff = pdff.mean().reset_index()

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import stats
pdff2 = pdff
pdff2 = pdff.drop(np.where((pdff["pol"]>500))[0])
plt.figure(figsize=(20,10))
slope, intercept, r_value, p_value, std_err = stats.linregress(pdff2.pol,pdff2.predPol)
plt.scatter(pdff2.pol, pdff2.predPol, c='royalblue')
#plt.plot(pdff2.predPol, slope*pdff2.predPol + intercept, c = 'navy')
plt.plot(range(int(min(pdff2.pol))-1,int(max(pdff2.pol))+1), range(int(min(pdff2.pol))-1,int(max(pdff2.pol))+1), c='black')

for i in range(24):
    plt.figure(figsize=(20,10))
    lp = pdf[(pdf["time"]>=i*60) & (pdf["time"]<=i*60+60)]
    lpf = lp.groupby(['lat','long'])
    lpf = lpf.mean().reset_index()
    plt.scatter(lpf.lat, lpf.long, c = lpf["predPol"], s = 50, cmap = 'winter', vmin = 0, vmax = 500)
    plt.xlim((min(pdf.lat), max(pdf.lat)))
    plt.ylim((min(pdf.long), max(pdf.long)))
    plt.colorbar()
    plt.savefig("ext\\gphTimeConnectionPredPol_"+str(i)+".png")
#plt.scatter(pdff.lat, pdff.long, c =  pdff["diff"]/pdff["pol"], s = 50, cmap = 'gray')


