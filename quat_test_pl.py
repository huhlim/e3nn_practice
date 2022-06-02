#!/usr/bin/env python

import sys
from typing import Optional, List

import torch
import torch.nn as nn
import pytorch_lightning as pl
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


class TestModel(pl.LightningModule):
    def __init__(self, layer="ConvLayer"):
        super().__init__()
        self.save_hyperparameters()
        self.irreps_input = o3.Irreps("5x0e")
        self.irreps_output = o3.Irreps("1x0e+1x1o")
        if layer == "ConvLayer":
            self.layer_1 = ConvLayer(
                self.irreps_input, "10x0e + 10x1o", radius=0.4, l_max=2
            )
            self.layer_2 = ConvLayer(
                "10x0e + 10x1o", self.irreps_output, radius=0.4, l_max=2
            )
        elif layer == "SE3Transformer":
            self.layer_1 = SE3Transformer(
                self.irreps_input, "10x0e + 10x1o", "12x0e + 12x1o", radius=0.4, l_max=2
            )
            self.layer_2 = SE3Transformer(
                "10x0e + 10x1o",
                self.irreps_output,
                "12x0e + 12x1o",
                radius=0.4,
                l_max=2,
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
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    def training_step(self, batch, batch_idx):
        loss = torch.nn.MSELoss()(self.forward(batch), batch.output_q)
        return loss
    def test_step(self, batch, batch_idx):
        random_rotation = o3.rand_matrix()
        batch.pos = batch.pos @ random_rotation.T
        output = self.forward(batch).cpu().detach().numpy()
        target = (
            batch.output_q @ self.irreps_output.D_from_matrix(random_rotation).T
        )
        target = target.cpu().detach().numpy()
        #
        mse = np.mean((output - target) ** 2)
        print("PREDICT\n", output.round(2))
        print("ANSWER\n", target.round(2))
        print("ACCURACY", mse)
        print("")
    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)

def main():
    trainloader = torch_geometric.loader.DataLoader(AAset(), batch_size=4, shuffle=True)
    testloader = torch_geometric.loader.DataLoader(
        AAset(), batch_size=20, shuffle=False
    )
    #
    trainer = pl.Trainer(accelerator='cpu', max_epochs=500, check_val_every_n_epoch=50)
    #
    # model = TestModel(layer='SE3Transformer')
    model = TestModel(layer="ConvLayer")
    #
    trainer.test(model, testloader)
    trainer.fit(model, trainloader, testloader)


if __name__ == "__main__":
    main()
