# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import matplotlib.pyplot as plt
import numpy as np

num_of_nodes = 500

cycles = 5

g = dgl.graph((np.arange(num_of_nodes-1), np.arange(1,num_of_nodes)))
g = dgl.add_reverse_edges(g)

length = np.pi * 2 * cycles
feat = torch.Tensor(np.arange(0,num_of_nodes))
feat = feat.unsqueeze(1)
labels = torch.Tensor(10*np.sin(np.arange(0, length, length / num_of_nodes)))

plt.plot(labels)

class GATv2(nn.Module):
    def __init__(self, num_features, num_classes, num_heads=1):
        super(GATv2, self).__init__()
        self.layer1 = dgl.nn.GATv2Conv(in_feats=num_features, out_feats=8, num_heads=num_heads)
        self.layer2 = dgl.nn.GATv2Conv(in_feats=8 * num_heads, out_feats=num_classes, num_heads=1)

    def forward(self, g, x):
        h = self.layer1(g, x).flatten(1, -1)
        h = F.elu(h)
        h = self.layer2(g, h).flatten(1, -1)
        return h

class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.layer1 = dgl.nn.GraphConv(in_feats=num_features, out_feats=16)
        self.layer2 = dgl.nn.GraphConv(in_feats=16, out_feats=num_classes)

    def forward(self, g, x):
        h = self.layer1(g, x)
        h = F.elu(h)
        h = self.layer2(g, h)
        return h

modelgat = GATv2(num_features=1, num_classes=1)
modelgcn = GCN(num_features=1, num_classes=1)

criterion = nn.MSELoss()
optimizergat = optim.Adam(modelgat.parameters(), lr=0.00001)
optimizergcn = optim.Adam(modelgcn.parameters(), lr=0.00001)

num_epochs = 2*10**4

for epoch in range(num_epochs):
    modelgat.train()
    optimizergat.zero_grad()
    output = modelgat(g, feat)
    loss = criterion(output, labels)
    if (epoch+1)%1000==0:
        print(f'Epoch : {epoch+1},\t,Loss : {loss.item()}')
    loss.backward()
    optimizergat.step()

for epoch in range(num_epochs):
    modelgcn.train()
    optimizergcn.zero_grad()
    output = modelgcn(g, feat)
    loss = criterion(output, labels)
    if (epoch+1)%1000==0:
        print(f'Epoch : {epoch+1},\t,Loss : {loss.item()}')
    loss.backward()
    optimizergcn.step()

with torch.no_grad():
    y_hat_gat = modelgat(g, feat)
    y_hat_gat = np.array(y_hat_gat)
    y_hat_gat = y_hat_gat.squeeze()

with torch.no_grad():
    y_hat_gcn = modelgcn(g, feat)
    y_hat_gcn = np.array(y_hat_gcn)
    y_hat_gcn = y_hat_gcn.squeeze()

plt.plot(y_hat_gat, label='Predicted GAT')
plt.plot(y_hat_gat, label='Predicted GCN')
plt.plot(labels, label='Ground Truth')
plt.ylabel('label')
plt.xlabel('datapoint')
plt.legend()
plt.title("GAT vs GCN on High Frequency Sine Wave")
plt.savefig('prediction.jpeg')

