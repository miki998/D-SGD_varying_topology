import torch
import numpy as np
import pandas as pd
import networkx as nx

################ OPERATION ON WEIGHTS FOR D-SGD ################
def getParams(model):
    return [param for param in model.parameters()]

def prod(params, scale):
    ret = []
    for param in params:
        tmp = (param * scale).detach()
        tmp.requires_grad = True
        ret.append(tmp)
    return ret

def addition(params1,params2):
    ret = []
    for i in range(len(params1)):
        tmp = (params1[i] + params2[i]).detach()
        tmp.requires_grad = True
        ret.append(tmp)
    return ret

def substraction(params1, params2):
    return addition(params1, prod(params2, -1))

def zeroParam(params):
    return [torch.zeros_like(param, requires_grad=True) for param in params]

def norm(params):
    with torch.no_grad():
        ret = 0
        for i in range(len(params)):
            ret += torch.norm(params[i],'fro') ** 2
        return ret.item()
    
################ UTILS FUNCTION ################

def consensusDistance(trainModels):
    """
    desc: compute consensus distance between weights from nodes' model
    args: 
        - trainModels::[list](list of nodes'weights)
    ret :
        - cDist      ::[float](consensus distance)
    
    """
    # 1. compute mean parameters values
    n = len(trainModels)
    curMean = getParams(trainModels[0])
    for nodeModel in trainModels[1:]:
        curMean = addition(curMean, getParams(nodeModel))
    curMean = prod(curMean, 1/len(trainModels))
    # 2. expand the mean parameters
    expanded = []
    for k in range(len(trainModels)):
        expanded.append(curMean)
    
    # 3. node weights minus mean weights 
    Cmat = []
    for k in range(len(trainModels)):
        Cmat.append(substraction(getParams(trainModels[k]),  expanded[k]))
    
    # 4. compute frobenius norm of the minus meaned
    cDist = 0
    for k in range(len(trainModels)):
        cDist += norm(Cmat[k])
    
    return cDist


def getLogs(stats):
    """
    desc: save logs of the training to check for minimum convergence (sanity check)
    args: 
        - stats::[list](dataframe with logs per nodes)
    ret : 
        - ret  ::[dataframe](concatenated dataframes with node labelled)
    """
    ret = []
    for node in stats:
        nodeLogs    = stats[node]
        df          = pd.concat([pd.DataFrame(d, index=[0]) for d in nodeLogs])
        df['node#'] = node
        ret.append(df)
    ret = pd.concat(ret)
    return ret.reset_index(drop=True)