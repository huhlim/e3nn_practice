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
            activation=torch.relu, \
            norm: Optional[bool] = True, \
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
        self.mlp = e3nn.nn.FullyConnectedNet(neurons, act=activation)
        if norm:
            self.norm = e3nn.nn.BatchNorm(self.out_Irreps)
        else:
            self.norm = None
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
        if self.norm is not None:
            f_out = self.norm(f_out)
        return f_out

class SE3Transformer(nn.Module):
    def __init__(self, in_Irreps: str, out_Irreps: str, attn_Irreps: str, \
            radius: float, l_max: Optional[int] = 2, \
            mlp_num_neurons: Optional[List[int]] = [20, 20], \
            activation=torch.relu, \
            norm: Optional[bool] = True, \
            return_attn: Optional[bool] = False, \
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
        self.attn_Irreps = o3.Irreps(attn_Irreps)   # for the query and key
        #
        # Query
        self.h_q = o3.Linear(self.in_Irreps, self.attn_Irreps)
        #
        # Key
        self.tensor_product_k = o3.FullyConnectedTensorProduct(\
                self.in_Irreps, self.sh_Irreps, self.attn_Irreps, shared_weights=False)
        neurons_k = mlp_num_neurons + [self.tensor_product_k.weight_numel]
        self.mlp_k = e3nn.nn.FullyConnectedNet(neurons_k, act=activation)
        #
        # Value
        self.tensor_product_v = o3.FullyConnectedTensorProduct(\
                self.in_Irreps, self.sh_Irreps, self.out_Irreps, shared_weights=False)
        neurons_v = mlp_num_neurons + [self.tensor_product_v.weight_numel]
        self.mlp_v = e3nn.nn.FullyConnectedNet(neurons_v, act=activation)
        #
        self.dot_product = o3.FullyConnectedTensorProduct(self.attn_Irreps, self.attn_Irreps, "0e")
        if norm:
            self.norm = e3nn.nn.BatchNorm(self.out_Irreps)
        else:
            self.norm = None
        self.return_attn = return_attn

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
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = e3nn.math.soft_one_hot_linspace(
                edge_length,
                start=0., end=self.radius, number=self.mlp_num_basis, \
                basis='smooth_finite', cutoff=True)
        edge_length_embedding = edge_length_embedding.mul(self.mlp_num_basis**0.5)
        edge_weight_cutoff = e3nn.math.soft_unit_step(10. * (1. - edge_length / self.radius))
        #
        # compute the queries (per node), keys (per edge), and values (per edge)
        q = self.h_q(f_in)
        k = self.tensor_product_k(f_in[edge_src], sh, self.mlp_k(edge_length_embedding))
        v = self.tensor_product_v(f_in[edge_src], sh, self.mlp_v(edge_length_embedding))
        #
        # compute the softmax (per edge)
        exp = edge_weight_cutoff[:, None] * self.dot_product(q[edge_dst], k).exp()
        z = torch_scatter.scatter(exp, edge_dst, dim=0, dim_size=len(f_in))
        z[z == 0] = 1.
        alpha = exp / z[edge_dst]
        #
        f_out = torch_scatter.scatter(alpha.sqrt() * v, edge_dst, dim=0, dim_size=len(f_in))
        if self.norm is not None:
            f_out = self.norm(f_out)
        if self.return_attn:
            attn = torch.zeros((n_node, n_node), dtype=torch.float)
            attn[edge_src, edge_dst] = alpha[:,0]
            return f_out, attn
        else:
            return f_out

class TestModel(nn.Module):
    def __init__(self, layer='ConvLayer'):
        super().__init__()
        if layer == 'ConvLayer':
            self.layer_1 = ConvLayer("5x0e", "10x0e + 10x1o", radius=0.4, l_max=2)
            self.layer_2 = ConvLayer("10x0e + 10x1o", "20x0e", radius=0.4, l_max=2)
        elif layer == 'SE3Transformer':
            self.layer_1 = SE3Transformer("5x0e", "10x0e + 10x1o", "12x0e + 12x1o", \
                    radius=0.4, l_max=2)
            self.layer_2 = SE3Transformer("10x0e + 10x1o", "20x0e", "12x0e + 12x1o", \
                    radius=0.4, l_max=2)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.layer_1(x, x.input_element)
        out = self.layer_2(x, out)
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
