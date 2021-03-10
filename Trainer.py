import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
#Net of 5 x 28 x 14 x 8 x 5 x 1
#input 5-> mean GraphSage 20-> mean GraphSage 10-> Linear Layer 8-> Linear Layer 5-> Output 1
#       ->  max GraphSage  8   max  GraphSage 4
#This is the real net, that includes layers as described above.
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1Mean = WeightedSAGEConv(5,20)
        self.conv1Max = SAGEConv(5,8)
        self.conv1Max.aggr = 'max'
        self.conv2Mean = WeightedSAGEConv(28,10)
        self.conv2Max = SAGEConv(28,4)
        self.conv2Max.aggr = 'max'
        self.conv3 = nn.Linear(14, 8)
        self.conv4 = nn.Linear(8, 5)
        self.conv5 = nn.Linear(5, 1)

    def forward(self,data):
        l, edge_index, norm = data.x.float(), data.edge_index, data.norm
        y = F.relu(self.conv1Mean(l, edge_index, norm))
        z = F.relu(self.conv1Max(l, edge_index))
        x = torch.cat((y,z),1)
        
        y = F.relu(self.conv2Mean(x, edge_index, norm))
        z = F.relu(self.conv2Max(x, edge_index))
        x = torch.cat((y,z,l),1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return torch.squeeze(x)

#the function which run one iteration of it, calculate loss and bakcpropogates.
#Now basically there are certain num_graphs which are like random walk for the model. There number is 
# in between 50-200 and they are simultaneously used to train this model.(That's the general idea of a random walk)
# .backward is applied on 'loss' which is sum of mse loss of all models, returned value is rmse 'loss' i.e. 'mloss' of this function.
def train(net, opt, dataset):
    net.train()
    opt.zero_grad()
    fv = dataset[0]
    output = net(fv)
    output = torch.reshape(output,(-1,))
    loss = F.mse_loss(output.float(),X1.float())
    for i in dataset[1:]:
        output = net(i)
        output = torch.reshape(output,(-1,))
        indLoss = F.mse_loss(output.float(),X1.float())
        loss = loss + indLoss
    mloss = loss/len(dataset)
    loss.backward()
    opt.step()
    return mloss.item()


#This one is iterator and runs the number of iteration, be careful it reinitializes the net, incase the loss reduction is slow in intial epochs,
# it also considers validation/test graph to train, it will stop either if number of epochs are over or if validation accuracy starts decreasing
# (validation check starts after atleast first valCheckEpochs number of epochs is completed)
import torch.optim as optim
def iterate(dataset, device, valGraph, valData, net = None, verbose = 10, epochs = 2000, minDef = 0.00001, valCheckEpochs = 2000):
    if(net == None):
        net = Net()
        net.to(device)
        net = net.float()
        net.train()
    optimizer1 = torch.optim.Adam(net.parameters(), lr = 0.1)
    optimizer01 = torch.optim.Adam(net.parameters(), lr = 0.01)
    optimizer001 = torch.optim.Adam(net.parameters(), lr = 0.001)     
    oz = optimizer001
    prevloss = 100000000
    valLoss = 100000000
    tdt = torch.from_numpy(valData[:,3])
    test_set = range(data.shape[0], data.shape[0]+valData.shape[0])
    valCount = 0
    count = 0
    epoch = 0
    loss = 100000
    while(epoch<epochs):
        if(epoch>500) & (loss>200):
            net = Net()
            net.to(device)
            net = net.float()
            net.train()
            oz = torch.optim.Adam(net.parameters(), lr = 0.001)
            print("Reintializing Weights")
            prevloss = 100000000
            epoch = 0
            count = 0
        
        loss = train(net, oz, dataset)
        if(epoch%50 == 0) and (epoch>valCheckEpochs):
            net.eval()
            out = net(valGraph)
            net.train()
            tempValLoss = F.mse_loss(out[test_set], tdt.to(device))**0.5
            if(tempValLoss > valLoss):
                valCount+=1
            else:
                valCount = 0
            valLoss = tempValLoss
            if(valCount>2):
                print("Breaking due to reduced val accuracy")
                print("Terminating at epoch: ",epoch)
                print("Termination Loss: ", loss)
                break

        epoch+=1
        if(prevloss - loss<minDef):
            count+=1
        prevloss = loss
        if(count>500) | (epoch == epochs):
            print("Terminating at epoch: ",epoch)
            print("Termination Loss: ", loss)
            break
        if(verbose>0) & (epoch%verbose == 0):
            print(epoch,": ",loss)
    return net,loss