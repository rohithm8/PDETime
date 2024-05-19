import torch.utils.data
import numpy as np
import pandas as pd

def make_temporal_features(df, time_index='date'):
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['date'] = pd.to_numeric(df['date'])
    df['date'] = (df['date'] - df['date'].min()) / (df['date'].max() - df['date'].min())
    df['day_of_year'] = (df['day_of_year'] - df['day_of_year'].min()) / (df['day_of_year'].max() - df['day_of_year'].min())
    df['month'] = (df['month'] - df['month'].min()) / (df['month'].max() - df['month'].min())
    df['day_of_week'] = (df['day_of_week'] - df['day_of_week'].min()) / (df['day_of_week'].max() - df['day_of_week'].min())
    df['hour'] = (df['hour'] - df['hour'].min()) / (df['hour'].max() - df['hour'].min())
    df['minute'] = (df['minute'] - df['minute'].min()) / (df['minute'].max() - df['minute'].min())
    return df



class ETTm1Dataset(torch.utils.data.Dataset):
    def __init__(self, df, horizon=96, lookback=None, target='OT', temporal_features=['day_of_year', 'month', 'day_of_week', 'hour', 'minute'], time_index='date'):
        self.df = df
        self.horizon = horizon
        self.lookback = lookback if lookback else horizon*5
        self.target = df[target].values
        self.features = self.df.copy().drop(columns=temporal_features).drop(columns=time_index).values
        self.temporal_features = self.df[temporal_features].values
        self.time_index = self.df[time_index].values
        self.labels = self.df[target].values
    def __len__(self):
        return len(self.df) - self.lookback - self.horizon + 1
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds (valid indices are [0:{}]".format(len(self)-1) + " but got index {}".format(idx) + " instead)")
        x = self.features[idx : idx + self.lookback]
        t = self.temporal_features[idx : idx + self.lookback + self.horizon]
        tau = self.time_index[idx : idx + self.lookback + self.horizon]
        labels = self.features[idx : idx + self.lookback + self.horizon]
        return x, t, tau, labels

