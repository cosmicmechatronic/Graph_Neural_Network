import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import GraphConv
import dgl
from dgl.data import DGLDataset
import torch
import torch as th
import os
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)
from rdkit import Chem
import numpy as np
import pandas as pd


def smiles2graph(smiles_string):
 """
 Converts SMILES string to graph Data object
 :input: SMILES string (str)
 :return: graph object
 """

 mol = Chem.MolFromSmiles(smiles_string)

 A = Chem.GetAdjacencyMatrix(mol)
 A = np.asmatrix(A)
 nnodes = len(A)
 nz = np.nonzero(A)
 edge_list = []
 src = []
 dst = []

 for i in range(nz[0].shape[0]):
  src.append(nz[0][i])
  dst.append(nz[1][i])

 u, v = src, dst
 g = dgl.graph((u, v))
 bg = dgl.to_bidirected(g)

 return bg


def feat_vec(smiles_string):
 """
 Returns atom features for a molecule given a smiles string
 """
 # atoms
 mol = Chem.MolFromSmiles(smiles_string)
 atom_features_list = []
 for atom in mol.GetAtoms():
  atom_features_list.append(atom_to_feature_vector(atom))
 x = np.array(atom_features_list, dtype=np.int64)
 return x


def transform_smile(string):
 graph = smiles2graph(string)
 graph.ndata['feat'] = torch.tensor(feat_vec(string))
 return graph



class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


model = GCN(9, 8, 2)

#device = torch.device('cpu')
PATH = 'GraphNN.pt'
model.load_state_dict(torch.load(PATH))
model.eval()


def get_prediction(string):
 foo = transform_smile(string)
 pred = model(foo, foo.ndata['feat'].float())
 result = int(pred.argmax(1))
 return result


