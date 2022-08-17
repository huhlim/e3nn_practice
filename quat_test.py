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
from liblayer import ConvLayer, SE3Transformer

torch.autograd.set_detect_anomaly(True)


class TestModel(nn.Module):
    def __init__(self, layer="ConvLayer", loop=False, self_interaction=True):
        super().__init__()
        self.irreps_input = o3.Irreps("5x0e")
        self.irreps_output = o3.Irreps("1x0e+1x1o")
        if layer == "ConvLayer":
            self.layer_1 = ConvLayer(
                self.irreps_input,
                "10x0e + 10x1o",
                radius=0.4,
                l_max=2,
                loop=loop,
                self_interaction=self_interaction,
            )
            self.layer_2 = ConvLayer(
                "10x0e + 10x1o",
                self.irreps_output,
                radius=0.4,
                l_max=2,
                loop=loop,
                self_interaction=self_interaction,
            )
        elif layer == "SE3Transformer":
            self.layer_1 = SE3Transformer(
                self.irreps_input,
                "10x0e + 10x1o",
                "12x0e + 12x1o",
                radius=0.4,
                l_max=2,
                loop=loop,
                self_interaction=self_interaction,
            )
            self.layer_2 = SE3Transformer(
                "10x0e + 10x1o",
                self.irreps_output,
                "12x0e + 12x1o",
                radius=0.4,
                l_max=2,
                loop=loop,
                self_interaction=self_interaction,
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.layer_1(x, x.input_element)
        out = self.layer_2(x, out)
        out = torch_scatter.scatter(out, x.batch, dim=0, reduce="sum")
        out[:, 0] = 2.0 * np.pi * (torch.sigmoid(out[:, 0]) - 0.5)
        v = out[:, 1:].clone()
        out[:, 1:] = v / torch.linalg.norm(v, dim=1)[:, None]
        return out

    def test_equivariance(self, x):
        random_rotation = o3.rand_matrix().to(self.device)
        #
        # after
        out_0 = []
        out = self.layer_1(x, x.input_element)
        out_0.append(out @ self.layer_1.out_Irreps.D_from_matrix(random_rotation).T)
        out = self.layer_2(x, out)
        out_0.append(out @ self.layer_2.out_Irreps.D_from_matrix(random_rotation).T)
        #
        # before
        x.pos = x.pos @ random_rotation.T
        out_1 = []
        out = self.layer_1(x, x.input_element)
        out_1.append(out)
        out = self.layer_2(x, out)
        out_1.append(out)
        #
        for step, (o0, o1) in enumerate(zip(out_0, out_1)):
            status = torch.allclose(o0, o1, rtol=1e-4, atol=1e-4)
            print(step, status)
            if not status:
                print(o0, o1)
                raise ValueError


def test(model, dataloader):
    random_rotation = o3.rand_matrix()
    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            batch.pos = batch.pos @ random_rotation.T
            batch = batch.to(model.device)
            output = model(batch).cpu().detach().numpy()
            target = batch.output_q @ model.irreps_output.D_from_matrix(random_rotation).T
            target = target.cpu().detach().numpy()
        mse = np.mean((output - target) ** 2)
        print("PREDICT\n", output.round(2))
        print("ANSWER\n", target.round(2))
        print("ACCURACY", mse)
        print("")


def test_equivariance(model, dataloader):
    batch = next(iter(dataloader))
    model.eval()
    model.test_equivariance(batch)
    #
    random_rotation = o3.rand_matrix().to(model.device)
    #
    batch = batch.to(model.device)
    output_0 = model(batch) @ model.irreps_output.D_from_matrix(random_rotation).T
    #
    batch.pos = batch.pos @ random_rotation.T
    output_1 = model(batch)
    print(torch.allclose(output_0, output_1, rtol=1e-4, atol=1e-4))
    raise


loss_f = torch.nn.MSELoss()


def main():
    trainloader = torch_geometric.loader.DataLoader(AAset(), batch_size=4, shuffle=True)
    testloader = torch_geometric.loader.DataLoader(AAset(), batch_size=20, shuffle=False)
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestModel(layer="ConvLayer", loop=True, self_interaction=True).to(device)
    # model = TestModel(layer="ConvLayer").to(device)
    model.device = device
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # test_equivariance(model, testloader)
    test(model, testloader)
    for epoch in range(500):
        loss_sum = 0.0
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            #
            batch = batch.to(device)
            output = model(batch)
            target = batch.output_q
            #
            loss = loss_f(output, target)
            loss.backward()
            loss_sum += loss.item()
            #
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print("LOSS ", epoch, loss_sum)
            test(model, testloader)
    print(epoch, loss_sum)
    test(model, testloader)


if __name__ == "__main__":
    main()
