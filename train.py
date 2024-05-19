# In[]:
from tqdm import tqdm
import torch
from torch import Tensor
import torch.utils.data
import pandas as pd
import numpy as np
from src.PDETime.models.PDETime import PDETime
from src.PDETime.preprocess.dataloader import ETTm1Dataset, make_temporal_features

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float)


horizon=96
lookback=horizon*5

model = PDETime(
    spatial_dim=7,
    temporal_features=5,
    temporal_latent_features=32,
    lookback=lookback,
    horizon=horizon,
    s_cff=1,
    hidden_features=64,
    INR_layers=5,
    aggregation_layers=1,
    latent_features=64,
    patch_length=32,
    MLP_hidden_layers=3,
    MLP_hidden_features=64
    )


class CustomLoss(torch.nn.Module):
    def __init__(self, loss_fn, lookback, horizon):
        super().__init__()
        self.loss_fn = loss_fn
        self.lookback = lookback
        self.horizon = horizon
    def forward(self, outputs, labels):
        fo_diff = outputs - torch.roll(outputs, 1, dims=-2)
        fo_diff_labels = labels - torch.roll(labels, 1, dims=-2)
        x_tau_0 = outputs[:,self.lookback-1,:].unsqueeze(-2).repeat(1, self.horizon + self.lookback, 1)
        L_r = self.loss_fn(outputs[:,:-self.horizon,:], labels[:,:-self.horizon,:]) / (self.lookback/self.horizon)
        L_p = self.loss_fn(outputs[:,-self.horizon:,:], labels[:,-self.horizon:,:] - x_tau_0[:,-self.horizon:,:])
        L_f = self.loss_fn(fo_diff[:,-self.horizon:,:], fo_diff_labels[:,-self.horizon:,:])
        return L_r + L_p + L_f


# criterion = CustomLoss(torch.nn.SmoothL1Loss(), lookback, horizon)
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

df = pd.read_csv('data/ETTm1.csv')
total_length = len(df)
train_ratio = 0.7
test_ratio = 0.2
val_ratio = 0.1
train_length = int(total_length * train_ratio)
test_length = int(total_length * test_ratio)
val_length = total_length - train_length - test_length


train_data = df[:train_length]
test_data = df[train_length:train_length+test_length]
val_data = df[train_length+test_length:]

print("Train Length:", len(train_data))
print("Test Length:", len(test_data))
print("Validation Length:", len(val_data))


dataset = ETTm1Dataset(make_temporal_features(train_data))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, generator=torch.Generator(device='cuda'))


# In[]:

model.train()
for epoch in range(5):
    for i, data in enumerate(tqdm(dataloader)):
        data = [data.to('cuda', dtype=torch.float) for data in data]
        x, t, tau, labels = data
        optimizer.zero_grad()
        outputs = model(x, t, tau)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
    print("epoch:", epoch, "    loss:", loss.item())
model.eval()
torch.save(model, 'model.pt')
# In[]:
x_tau_0 = outputs[:,lookback-1,:].unsqueeze(-2).repeat(1, horizon + lookback, 1)

labelss = labels.detach().cpu().numpy()
outputsss = (outputs).detach().cpu().numpy()
import matplotlib.pyplot as plt

# In[]:

batch_no = 1
plt.plot(labelss[batch_no,:,-1], label='ground truth')
plt.plot(outputsss[batch_no,:,-1], label='preds')
plt.axvline(x=lookback, color='r')
plt.legend()
plt.show()

# In[]:

labels[:, lookback-1].repeat(horizon + lookback, 1).shape
outputs[:,:,-1].shape

