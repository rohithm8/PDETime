import torch.nn as nn
import torch

class RidgeRegression(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        if lambda_ < 0:
            raise ValueError("Ridge regressor regularization strength must be non-negative")
        self.lambda_ = lambda_
    def forward(self, x, y):
        batch_size, window_length, _ = x.shape
        xb = torch.cat([x, torch.ones(batch_size, window_length, 1)], dim=-1)
        xb_L = xb.mT @ xb
        xb_L.diagonal(dim1=-2, dim2=-1).add_(self.lambda_)
        wb = torch.linalg.solve(xb_L, xb.mT @ y)
        return wb[:,:-1,:], wb[:,-1,:]

    
class Decoder(nn.Module):
    def __init__(self, lookback, horizon, lambda_=0.):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.ridge = RidgeRegression(lambda_=lambda_)
    
    def forward(self, tau, x):
            tau_r, tau_p = torch.split(tau, [self.lookback, self.horizon], dim=-2)
            w, b = self.ridge(tau_r, x)
            preds = torch.einsum('bwl,bld->bwd', tau, w) 
            preds += b.unsqueeze(-2).repeat(1, self.horizon + self.lookback, 1)
            return preds
