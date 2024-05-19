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
        return target#.squeeze(-1)