# In[]:
from tqdm import tqdm
import torch
from torch import Tensor
import torch.utils.data
import pandas as pd
import numpy as np
from PDETime.models.pdetime_model import PDETime, PDETimeLoss
from src.PDETime.preprocess.dataloader import PDETimeDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float)


horizon=96
lookback=horizon*5

model = PDETime(
    dimension=7,
    temporal_features=7,
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



criterion = PDETimeLoss(torch.nn.SmoothL1Loss(), lookback, horizon)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

df = pd.read_csv('data/ETTm1.csv')

train_data, test_data, val_data = PDETimeDataset.train_test_val_split(df, horizon=horizon, lookback=lookback, val_ratio=0.1, test_ratio=0.2)

print("Train Length:", len(train_data))
print("Test Length:", len(test_data))
print("Validation Length:", len(val_data))



dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, generator=torch.Generator(device='cuda'))

# In[]:

model.train()
for epoch in range(10):
    epoch_loss = 0
    for i, data in enumerate(tqdm(dataloader)):
        data = [data.to('cuda', dtype=torch.float) for data in data]
        x, t, tau, labels = data
        optimizer.zero_grad()
        outputs = model(x, t, tau)
        loss = criterion(outputs, labels.float())
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    epoch_loss /= len(dataloader)
    print("epoch:", epoch, "    training loss:", epoch_loss.item())
model.eval()
# torch.save(model, 'model.pt')
# In[]:
x_tau_0 = outputs[:,lookback-1,:].unsqueeze(-2).repeat(1, horizon, 1)
x_tau_0 = torch.cat((torch.zeros_like(outputs[:,:lookback,:]), x_tau_0), dim=-2)

labelss = labels.detach().cpu().numpy()
outputsss = (outputs + x_tau_0).detach().cpu().numpy()
import matplotlib.pyplot as plt

# In[]:

batch_no = 2
plt.plot(labelss[batch_no,:,-1], label='ground truth')
plt.plot(outputsss[batch_no,:,-1], label='preds')
plt.axvline(x=lookback, color='r')
plt.legend()
plt.show()
