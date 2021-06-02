#!/usr/bin/env python

import sys
from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np
np.set_printoptions(suppress=True, linewidth=1000)

import torch_geometric
import torch_cluster 
import torch_scatter

import e3nn
import e3nn.nn
from e3nn import o3

from libdata import AAset
from liblayer import loss_f_soft as loss_f
from liblayer import TestModel, test

def main():
    trainloader = torch_geometric.data.DataLoader(AAset(), batch_size=4, shuffle=True)
    testloader = torch_geometric.data.DataLoader(AAset(), batch_size=20, shuffle=False)
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestModel(layer='SE3Transformer').to(device)
    #model = TestModel(layer='ConvLayer').to(device)
    model.device = device
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    test(model, testloader)
    for epoch in range(250):
        loss_sum = 0.
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            #
            batch = batch.to(device)
            output = model(batch)
            target = batch.output_pdbIndex
            #
            loss = loss_f(output, target)
            loss.backward()
            loss_sum += loss.item()
            #
            optimizer.step()

        if (epoch+1)%50 == 0:
            print ("LOSS ", epoch, loss_sum)
            test(model, testloader)
    print (epoch, loss_sum)
    test(model, testloader)

if __name__ == '__main__':
    main()

