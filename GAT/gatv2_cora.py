# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.data import CoraGraphDataset

class GATv2(nn.Module):
    def __init__(self, num_features, num_classes, num_heads=1, feat_drop=0.6, attn_drop=0.6, layers = 1):
        super(GATv2, self).__init__()
        self.input = dgl.nn.GATv2Conv(in_feats=num_features, out_feats=8, num_heads=num_heads, feat_drop=feat_drop)
        self.hidden = [dgl.nn.GATv2Conv(in_feats=8 * num_heads, out_feats=8, num_heads=num_heads, feat_drop=feat_drop) for _ in range(layers-1)]
        self.output = dgl.nn.GATv2Conv(in_feats=8 * num_heads, out_feats=num_classes, num_heads=1, feat_drop=feat_drop)


    def forward(self, g, x):
        h = self.input(g, x).flatten(1, -1)
        h = F.elu(h)
        if self.hidden:
            for hidden in self.hidden:
                h = hidden(g, h).flatten(1, -1)
                h = F.elu(h)
        h = self.output(g, h).flatten(1, -1)
        return h

dataset = CoraGraphDataset(raw_dir='drive/MyDrive/')
g = dataset[0]

train_mask = g.ndata['train_mask']
test_mask = g.ndata['test_mask']

print(f'Total Samples : {len(train_mask)}')

train_mask[0:(2708*5)//100] = True

print(f'Total Training Samples : {sum(train_mask)}')

test_mask[:] = False
test_mask[(2708*9)//10:] = True

print(f'Total Training Samples : {sum(test_mask)}')

steps = 200

model = GATv2(num_features=g.ndata['feat'].shape[1], num_classes=dataset.num_classes, num_heads=1, layers=1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-5)

for epoch in range(steps):
    model.train()
    optimizer.zero_grad()
    logits = model(g, g.ndata['feat'])
    loss = criterion(logits.softmax(dim=1)[train_mask], g.ndata['label'][train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Testing loop
    if (epoch+1)%20==0:
        model.eval()
        with torch.no_grad():
            logits = model(g, g.ndata['feat'])
            pred = logits[test_mask].max(1)[1]
            acc = pred.eq(g.ndata['label'][test_mask]).sum().item() / test_mask.sum().item()
            logits = model(g, g.ndata['feat'])
            pred = logits[train_mask].max(1)[1]
            acc_train = pred.eq(g.ndata['label'][train_mask]).sum().item() / train_mask.sum().item()
            print(f'Epoch {epoch + 1}, Test Accuracy: {acc * 100:.2f}%, Train Accuracy: {acc_train * 100:.2f}%')

