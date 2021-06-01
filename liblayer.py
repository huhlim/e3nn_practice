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

class ConvLayer(nn.Module):
    def __init__(self, in_Irreps: str, out_Irreps: str, radius: float, \
            l_max: Optional[int] = 2, \
            mlp_num_neurons: Optional[List[int]] = [20, 20], \
            ) -> None:
        #
        super().__init__()
        #
        self.in_Irreps = o3.Irreps(in_Irreps)
        self.out_Irreps = o3.Irreps(out_Irreps)
        self.sh_Irreps = o3.Irreps.spherical_harmonics(l_max)
        self.radius = radius
        self.mlp_num_basis = mlp_num_neurons[0]
        #
        self.tensor_product = o3.FullyConnectedTensorProduct(\
                self.in_Irreps, self.sh_Irreps, self.out_Irreps, shared_weights=False)
        #
        neurons = mlp_num_neurons + [self.tensor_product.weight_numel]
        self.mlp = e3nn.nn.FullyConnectedNet(neurons, act=torch.relu)
        self.bn = e3nn.nn.BatchNorm(self.out_Irreps)
        #
    def forward(self, data: torch_geometric.data.Data, \
            f_in: torch.Tensor, \
            ) -> torch.Tensor:
        n_node = data.pos.size(0)
        edge_src, edge_dst = torch_cluster.radius_graph(data.pos, self.radius, batch=data.batch)
        edge_vec = data.pos[edge_dst] - data.pos[edge_src]
        #
        sh = o3.spherical_harmonics(self.sh_Irreps, edge_vec, \
                                    normalize=True, normalization='component')
        #
        edge_length_embedding = e3nn.math.soft_one_hot_linspace(
                edge_vec.norm(dim=1), \
                start=0., end=self.radius, number=self.mlp_num_basis, \
                basis='smooth_finite', cutoff=True)
        edge_length_embedding = edge_length_embedding.mul(self.mlp_num_basis**0.5)
        weight = self.mlp(edge_length_embedding)
        #
        f_out = self.tensor_product(f_in[edge_src], sh, weight)
        f_out = torch_scatter.scatter(f_out, edge_dst, dim=0, dim_size=n_node, reduce='sum')
        f_out = self.bn(f_out)
        return f_out

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = ConvLayer("5x0e", "10x0e + 10x1o", radius=0.4, l_max=2)
        self.conv_2 = ConvLayer("10x0e + 10x1o", "20x0e", radius=0.4, l_max=2)
    def forward(self, x):
        out = self.conv_1(x, x.input_element)
        out = self.conv_2(x, out)
        out = torch_scatter.scatter(out, x.batch, dim=0, reduce='sum')
        out = torch.nn.functional.softmax(out, dim=-1)
        return out

def test(model, dataloader):
    random_rotation = o3.rand_matrix()
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            batch.pos = batch.pos@random_rotation.T
            batch = batch.to(model.device)
            output = model(batch).cpu().detach().numpy()
            target = batch.output_pdbIndex.cpu().detach().numpy()
        print (output.round(2))
        predict = np.argmax(output, axis=-1)
        accuracy = np.sum(predict == target) / target.shape[0] * 100.
        print ("PREDICT  ", predict)
        print ("ANSWER   ", target)
        print ("ACCURACY" , accuracy)
        print ("")

def loss_f_hard(output, target):
    return torch.nn.CrossEntropyLoss()(output, target)
def loss_f_soft(output, target):
    soft_label = 0.95
    target_soft = torch.zeros_like(output)
    target_soft += (1-soft_label)/19.
    for i,label in enumerate(target):
        target_soft[i,label] = 0.9
    #
    loss = -torch.sum(target_soft * output.log(), dim=1)
    return loss.mean()
