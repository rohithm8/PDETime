from torch import roll
import torch.nn as nn
from src.PDETime.models.encoder import Encoder
from src.PDETime.models.solver import Solver
from src.PDETime.models.decoder import Decoder

class PDETime(nn.Module):
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
                 patch_length,
                 MLP_hidden_layers,
                 MLP_hidden_features,
                 ):
        super().__init__()
        
        self.encoder = Encoder(
            spatial_dim=spatial_dim,
            temporal_features=temporal_features,
            temporal_latent_features=temporal_latent_features,
            lookback=lookback,
            horizon=horizon,
            s_cff=s_cff,
            hidden_features=hidden_features,
            INR_layers=INR_layers,
            aggregation_layers=aggregation_layers,
            latent_features=latent_features
            )

        self.solver = Solver(
            window_length=lookback+horizon,
            latent_features=latent_features,
            patch_length=patch_length,
            MLP_hidden_layers=MLP_hidden_layers,
            MLP_hidden_features=MLP_hidden_features
            )

        self.decoder = Decoder(
            lookback=lookback,
            horizon=horizon,
            lambda_=0.1
            )
    
    def forward(self, x, t, tau):
        tau = self.encoder(tau, x, t)
        tau = self.solver(tau)
        target = self.decoder(tau, x)
        return target
    

class PDETimeLoss(nn.Module):
    def __init__(self, loss_fn, lookback, horizon):
        super().__init__()
        self.loss_fn = loss_fn
        self.lookback = lookback
        self.horizon = horizon
    def forward(self, outputs, labels):
        fo_diff = outputs - roll(outputs, 1, dims=-2)
        fo_diff_labels = labels - roll(labels, 1, dims=-2)
        x_tau_0 = outputs[:,self.lookback-1,:].unsqueeze(-2).repeat(1, self.horizon + self.lookback, 1)
        L_r = self.loss_fn(outputs[:,:-self.horizon,:], labels[:,:-self.horizon,:]) / (self.lookback/self.horizon)
        L_p = self.loss_fn(outputs[:,-self.horizon:,:], labels[:,-self.horizon:,:] - x_tau_0[:,-self.horizon:,:])
        L_f = self.loss_fn(fo_diff[:,-self.horizon:,:], fo_diff_labels[:,-self.horizon:,:])
        return L_r + L_p + L_f
