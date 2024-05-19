import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class MLPLayer(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()

        if hidden_layers < 1 and hidden_features != out_features:
            raise ValueError('hidden_layers must be at least 1 if hidden_features != out_features')

        self.net = [
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            ]
        if hidden_layers > 1:
            self.net += [
                nn.Linear(hidden_features, hidden_features),
                nn.GELU()]*(hidden_layers-1)
        self.net += [
            nn.Linear(hidden_features, out_features),
            nn.GELU()]
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)


class Solver(nn.Module):
    def __init__(self,
                 window_length,
                 latent_features,
                 patch_length,
                 MLP_hidden_layers,
                 MLP_hidden_features,
                 ):
        super().__init__()
        
        self.L = window_length
        self.D = latent_features
        self.S = patch_length
        self.hidden_layers = MLP_hidden_layers
        self.hidden_features = MLP_hidden_features

        self.mlp_dudt = MLPLayer(
            in_features=self.D,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            out_features=self.D
            )
        
        self.mlp_u = MLPLayer(
            in_features=self.D,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            out_features=self.D
            )

        self.mlp_integrate = MLPLayer(
            in_features=self.D,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            out_features=self.D
            )
    def forward(self, tau):
        batch_size = tau.shape[0]
        dudt = self.mlp_dudt(tau) * -1
        u = self.mlp_u(tau)
        u = u.view(batch_size,-1,self.S,self.D)
        dudt = dudt.view(batch_size,-1,self.S,self.D)
        dudt = torch.flip(dudt, [2])
        dudt = dudt[:,:,:-1,:]
        dudt = torch.cat((u[:,:,-1:,:], dudt), dim=-2)
        intdudt = torch.cumsum(dudt, dim=-2)
        intdudt = torch.flip(intdudt, [2])
        intdudt = intdudt.reshape(batch_size, self.L, self.D)

        intdudt = self.mlp_integrate(intdudt)
        return intdudt
