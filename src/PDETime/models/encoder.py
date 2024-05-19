import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional



class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    # def forward(self, coords):
    #     coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
    #     output = self.net(coords)
    #     return output#, coords

    def forward(self, x):
        return self.net(x)



# from https://github.com/jmclong/random-fourier-features-pytorch/
def sample_b(s: float, size: tuple) -> Tensor:
    r"""Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, 2^s)`

    Args:
        s (float): standard deviation
        size (tuple): size of the matrix sampled

    See :class:`~rff.layers.GaussianEncoding` for more details
    """
    return torch.randn(size) * (2 ** s)


# from https://github.com/jmclong/random-fourier-features-pytorch/
def fourier_encoding(
        v: Tensor,
        b: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`

    See :class:`~rff.layers.GaussianEncoding` for more details.
    """
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class CFF(nn.Module):
    """Layer for mapping coordinates using random concatenated Fourier features"""

    def __init__(self, s: int = None,
                 in_features: int = None,
                 encoded_size: int = None):
        r"""
        Args:
            s (int): number of random Fourier features to create per input feature
            input_size (int): the number of input dimensions
            encoded_size (int): the number of dimensions the `b` matrix maps to
        Raises:
            ValueError:
                If :attr:`b` is provided and one of :attr:`sigma`, :attr:`input_size`,
                or :attr:`encoded_size` is provided. If :attr:`b` is not provided and one of
                :attr:`sigma`, :attr:`input_size`, or :attr:`encoded_size` is not provided.
        """
        super().__init__()
        self.s = s
        if s is None or in_features is None or encoded_size is None:
            raise ValueError(
                'Arguments "sigma," "input_size," and "encoded_size" are required.')

        b_s = [sample_b(s_i, (encoded_size, in_features)) for s_i in range(1,s+1)]

        for i,b in enumerate(b_s):
             self.register_buffer('b'+str(i+1), b)

    def forward(self, v: Tensor) -> Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: Tensor mapping using random fourier features of shape :math:`(N, *, 2 \cdot \text{encoded_size})`
        """
        ff = [fourier_encoding(v, getattr(self, 'b'+str(i+1))) for i in range(self.s)]
        return torch.cat(ff, dim=-1)


class Tau_INR(nn.Module):
    """Implicit Neural Representation with concatenated Fourier features for time index feature"""
    def __init__(self, in_features, hidden_features, s_cff, num_layers, out_features):
        super().__init__()
        if hidden_features % 2 != 0:
            raise ValueError('out_features (d) must be divisible by 2: got %d' % out_features)
        self.net = []
        self.net.append(CFF(s_cff, in_features=in_features, encoded_size=out_features//2))
        self.net.append(nn.Linear(out_features*s_cff, hidden_features, bias=True))
        self.net.append(nn.GELU())
        if num_layers > 2:
            for i in range(num_layers-2):
                self.net.append(nn.Linear(hidden_features, hidden_features, bias=True))
                self.net.append(nn.GELU())
        self.net.append(nn.Linear(hidden_features, out_features, bias=True))
        self.net.append(nn.GELU())
        
        self.net = nn.Sequential(*self.net).float()
    
    def forward(self, tau):
        return self.net(tau)


class CrossAttention(nn.Module):
    def __init__(self, input_dim, kq_dim, output_dim):
        super().__init__()
        self.d_out_kq = kq_dim
        self.W_q = nn.Linear(input_dim, kq_dim)
        self.W_k = nn.Linear(input_dim, kq_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
    
    def forward(self, x_1, x_2):
        queries_1 = self.W_q(x_1)
        keys_2 = self.W_k(x_2)
        values_2 = self.W_v(x_2)
        attn_scores=queries_1.matmul(keys_2.mT)
        attn_weights=torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )
        
        return attn_weights.matmul(values_2)


class AggLayer(nn.Module):
    def __init__(self,
                 latent_features,
                 temporal_latent_features,
                 spatial_dim,
                 lookback,
                 horizon,
                 attn_dim):
        super().__init__()
        self.D = latent_features
        self.T = temporal_latent_features
        self.L = lookback
        self.H = horizon
        self.C = spatial_dim
        self.attn_dim = attn_dim
        
        self.tau_linear = nn.Linear(self.D, self.D)
        self.t_linear = nn.Linear(self.T, self.T)
        self.x_linear = nn.Linear(self.D, self.D)

        self.gelu = nn.GELU()
        self.tau_norm_1 = nn.LayerNorm((self.L+self.H, self.D))
        self.t_norm_1 = nn.LayerNorm((self.L+self.H, self.T))
        self.x_norm_1 = nn.LayerNorm((self.C, self.D))

        self.attention = CrossAttention(input_dim=self.D, kq_dim=self.attn_dim, output_dim=self.D)
        self.tau_norm_2 = nn.LayerNorm((self.L+self.H, self.D))
        self.t_tau_linear = nn.Linear(self.T + self.D, self.D)
        self.tau_norm_3 = nn.LayerNorm((self.L+self.H, self.D))


    def forward(self, tau, t, x):
        tau = self.tau_linear(tau)
        t = self.t_linear(t)
        x = self.x_linear(x)
        tau = self.tau_norm_1(self.gelu(tau))
        t = self.t_norm_1(t.sin())
        x = self.x_norm_1(self.gelu(x))
        tau = self.tau_norm_2(tau + self.attention(tau, x))
        t = torch.cat((t, tau), dim=-1)
        tau = self.tau_norm_3(tau + self.t_tau_linear(t))

        return tau


class Encoder(nn.Module):
    def __init__(self,
                 spatial_dim,
                 temporal_features,
                 temporal_latent_features,
                 lookback,
                 horizon,
                 s_cff,
                 hidden_features,
                 INR_layers,
                 aggregation_layers,
                 latent_features,
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.
                 ):
        super().__init__()
        
        # letters are the same as the paper
        self.L = lookback
        self.H = horizon
        self.s_cff = s_cff if s_cff is not None else 1
        self.C = spatial_dim
        self.T = temporal_latent_features
        self.D = latent_features
        self.hidden_features = hidden_features
        self.INR_layers = INR_layers
        self.temporal_features = temporal_features
        
        self.tau_INR = Tau_INR(
            in_features=1,
            hidden_features=self.hidden_features,
            s_cff=self.s_cff,
            num_layers=self.INR_layers,
            out_features=self.D
            )

        self.t_siren = Siren(
            in_features=self.temporal_features,
            hidden_features=self.hidden_features,
            hidden_layers=self.INR_layers,
            out_features=self.T,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
            )

        self.x_siren = Siren(
            in_features=self.L,
            hidden_features=self.hidden_features,
            hidden_layers=self.INR_layers,
            out_features=self.D,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
            )

        self.agg_layers = [
            AggLayer(
                latent_features=self.D,
                temporal_latent_features=self.T,
                spatial_dim=self.C,
                lookback=self.L,
                horizon=self.H,
                attn_dim=64)
            ]
        
        self.agg_layers *= aggregation_layers
    def forward(self, tau, x, t):
        tau = self.tau_INR(tau.unsqueeze(-1))
        x = self.x_siren(torch.transpose(x, 1, 2))
        t = self.t_siren(t)

        for agg_layer in self.agg_layers:
            tau = agg_layer(tau, t, x)
        return tau