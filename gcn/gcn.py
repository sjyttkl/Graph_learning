# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     gcn.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/3/8 15:44
   Description :
   gcn图卷积网络，全称：graph convolutional networks 图卷积网络，提出于2017年，GCN的出现标志这图神经网络的出现，要说深度学习最常用的网络结构就是
        CNN，RNN。GCN与CNN不仅名字相似，其实理解起来也很类似，都是特征提取器。不同的是，CNN提取的张量数据特征，而GCN提出的是图数据特征

==================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv  # 这是一个图 DGL库
from dgl.data import CoraGraphDataset


class GCN(nn.Module):
    def __init__(self, g, in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        #
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # 输出层：
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i!=0:
                h = self.dropout(h)
            h = layer(self.g,h)
        return h
# @torch.no_grad()
def evalutate(model,features,labels,mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        # print(logits)
        logits = logits[mask]
        labels = labels[mask]
        _,indices  = torch.max(logits,dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train( n_epochs=100,lr=1e-2,
           weight_decay=5e-4,n_hidden=16,
           n_layers=1,activation=F.relu,
           dropout=0.5):
    data = CoraGraphDataset()
    print(data)
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask'] #是true，fasle。其实就是样本提取
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_classes

    model  = GCN(g,in_feats,n_hidden,n_classes,n_layers,activation,dropout)

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

    for epoch in range(n_epochs):
        model.train()
        logits =model(features)
        loss = loss_fcn(logits[train_mask],labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc =evalutate(model,features,labels,val_mask)
        print("Epoch {} | Locc {:.4f} | accuracy {:.4f}".format(epoch,loss.item(),acc))
    print()
    acc =evalutate(model,features,labels,test_mask)
    print("Test accuracy {:.2%}".format(acc))

if __name__ == "__main__":
    train()

