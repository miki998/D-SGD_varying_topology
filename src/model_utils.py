import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
from .helper import *


############ MODEL ############

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# use standard rather shallow network for simplicity
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
############ TRAINING FUNC ############
    
def train(nodedata, nodemodel, nodeoptimizer, loss_fn, epoch, verbose=True):
    """
    desc: training loop that we apply to nodes, names are self-explanatory
    """
    running_loss = 0.0
    for i in range(len(nodedata[0])):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = nodedata[0][i], nodedata[1][i]

        # zero the parameter gradients
        nodeoptimizer.zero_grad()

        # forward + backward + optimize
        outputs = nodemodel(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        nodeoptimizer.step()

        # print statistics
        running_loss += loss.item()
        if verbose:
            if i % 10 == 9:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
                

def mixing_matrix(graph, loop=False):
    """
    desc: generate mixing matrix from a graph (we implemented our experiments without self-loops for nodes)
    """
    x = nx.to_numpy_matrix(graph)
    # with self loops
    if loop:
        x = x + np.eye(x.shape[0])
        
    for i in range(x.shape[0]):
        n_neighbors   = np.count_nonzero(x[i, :])
        mixing_matrix = 1 / n_neighbors
        x[i, np.array(x[i, :], dtype=bool)[0]] = mixing_matrix
    
    return x


def subopt(nodedata, nodemodel, loss_fn):
    """
    desc: record the loss states for each node, we evaluate using the nodemodel (this function is also alternately used\
    in evaluating the validation and test score per node)
    """
    nodemodel.eval()
    total, total_loss, correct = 0, 0, 0
    for i in range(len(nodedata[0])):
        inputs, labels = nodedata[0][i], nodedata[1][i]

        with torch.no_grad():
            outputs = nodemodel(inputs)
            loss    = loss_fn(outputs, labels)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item() 
    
    return total_loss / total, correct / total    


def computeDecentralize(nodesData, topology, K, device='cpu', max_epoch=10, validation=None):
    """
    desc: compute a full training of D-SGD while logging on train set (and possibly validation set)
    args: 
        - nodesData ::[list(tuple<<array>,<array>>)](pairs of images and labels for each nodes)
        - topology  ::[Topology](topology class object used for its mixing matrix)
        - K         ::[int] (number of nodes)
        - device    ::[str] ('cpu'/'cuda:{idx}')
        - max_epoch ::[int]
        - validation::[tuple<<array>,<array>>](images and labels belonging to validation set, we give the same validation set\
        to all nodes when evaluating)
    ret :
        - resultdfTrain::[dataframe](logging training accuracies and losses for all nodes and epochs)
        - resultdfVal  ::[dataframe](logging validation accuracies and losses for all nodes and epochs)
        - cDists       ::[list](list of consensus distances for each epochs)
        - models       ::[CNN](models and its updated weights, used for test set prediction)
    """    
    graph = topology.to_networkx()
    W = nx.adjacency_matrix(graph)
    # topology.draw_graph()

    W = mixing_matrix(graph, loop=False)

    # we define model and optimizers for each nodes
    # reset the models and optimizers
    models     = []
    optimizers = []
    losses     = []

    for i in range(K):
        model     = CNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
        loss_fn   = nn.CrossEntropyLoss()

        models.append(model)
        optimizers.append(optimizer)
        losses.append(loss_fn)

    train_stats = {str(node):[] for node in graph.nodes()}
    val_stats   = {str(node):[] for node in graph.nodes()}
    cDists      = []

    for epoch in tqdm(range(max_epoch)):
        # we train separately the nodes with their respective data 
        # we update implicitely models and optimizers
        for node_idx in graph.nodes():

            train(nodesData[node_idx], models[node_idx], optimizers[node_idx], losses[node_idx], epoch, device, verbose=False)

        # we share the parameters weights among the neighbours
        for node_idx in graph.nodes():
            neighbors = topology.get_neighbors(node_idx)

            # without self loops
            ret = zeroParam(getParams(models[node_idx]))

            # with self loops
    #         ret = getParams(models[node_idx])
    #         ret = prod(ret, W[node_idx, node_idx])

            for p in neighbors:
                p_params = getParams(models[p])
                tmp = prod(p_params, W[node_idx, p])
                ret = addition(ret, tmp)

            # copying the parameters to inplace operation for each nodes
            with torch.no_grad():
                for idx, param in enumerate(models[node_idx].parameters()):
                    param.copy_(ret[idx])


            loss, acc   = subopt(nodesData[node_idx], models[node_idx], loss_fn, device)
            curTrainDir = {'Epoch': epoch + 1, 'loss': loss, 'acc': acc}
            train_stats[str(node_idx)].append(curTrainDir)
            
            if not validation is None:
                lossVal, accVal = subopt(validation, models[node_idx], loss_fn, device)
                curValDir       = {'Epoch': epoch + 1, 'loss': lossVal, 'acc': accVal}
                val_stats[str(node_idx)].append(curValDir)                

        cDist  = consensusDistance(models)
        cDists.append(cDist)
    
    resultdfTrain = getLogs(train_stats)
    resultdfVal   = None
    if not validation is None:
        resultdfVal = getLogs(val_stats)

    return resultdfTrain, resultdfVal, cDists, models