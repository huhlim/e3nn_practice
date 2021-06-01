#!/usr/bin/env python

import sys
import glob
import mdtraj

import torch
import torch.nn as nn
import numpy as np

import torch_geometric

ELEMENTs = ['H', 'C', 'N', 'O', 'S']
one_hot_encoding = lambda x: np.eye(len(ELEMENTs), dtype=np.float)[x]
dtype = torch.get_default_dtype()

class AAset(torch_geometric.data.Dataset):
    def __init__(self, selection='element != H'):
        self.pdb_s = []
        pdb_fn_s = sorted(glob.glob("pdb/*.pdb"))
        self.n_pdb = len(pdb_fn_s)
        for pdbIndex,pdb_fn in enumerate(pdb_fn_s):
            pdb = mdtraj.load(pdb_fn, standard_names=False)
            xyz = pdb.xyz[0]
            #
            top = pdb.topology
            element = []
            for atom in top.atoms:
                element.append(ELEMENTs.index(atom.element.symbol))
            element = np.array(element, dtype=np.int)
            element = one_hot_encoding(element)
            #
            inputIndex = top.select(selection)
            #
            data = torch_geometric.data.Data(pos=torch.tensor(xyz[inputIndex], dtype=dtype))
            data.input_element = torch.tensor(element[inputIndex], dtype=dtype)
            data.output_element = torch.tensor(element, dtype=dtype)
            data.output_xyz = torch.tensor(xyz, dtype=dtype)
            data.output_pdbIndex = pdbIndex
            #
            print (pdbIndex, top.residue(0).name)
            self.pdb_s.append(data)
    def __len__(self):
        return self.n_pdb
    def __getitem__(self, index):
        return self.pdb_s[index]

def batch_to_device(batch, device, ignore=[]):
    for key in batch.keys:
        if key in ignore: continue
        batch[key] = batch[key].to(device)
    return batch

def main():
    dataloader = torch_geometric.data.DataLoader(AAset(selection='all'), batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print (batch.input_element.device)
    batch.to("cuda")
    #out = batch_to_device(batch, "cuda")
    print (batch.input_element.device)
    #print (out.input_element.device)

if __name__ == '__main__':
    main()

