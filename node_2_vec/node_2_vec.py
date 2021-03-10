# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     node2vec.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2021/3/8 13:51
   Description :   https://arxiv.org/abs/1607.00653
   node2vec 在2016年发布的，与deepwalk的区别就是控制了游走方向的参数.按照Deepwalk的思想，
   所有邻居节点游走的概率都是相等的。而Node2vec可以通过调整方向的参数来控制模型更倾向宽带有限的游走还是深度优先的游走
   https://www.bilibili.com/video/BV15o4y1R7nC?from=search&seid=6653090908098469268
   https://github.com/dmlc/dgl/blob/master/examples/tensorflow/gat/gat.py
==================================================
"""

import networkx as nx
from node2vec import Node2Vec  #非常简单的node2vec api,基于 networkx与gensim进行的封装

graph = nx.fast_gnp_random_graph(n =100,p =0.5) #快速 随机生成一个无向图
node2vec = Node2Vec(graph,dimensions=64,walk_length=30,num_walks=100,p=0.3,q=0.7,workers=4) #初始化模型 这里的workers是设置同时游走的线程数

model = node2vec.fit() #训练模型
print(model.wv.most_similar("2",topn=3)) #观察与节点2最相近的三个节点

