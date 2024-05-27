from typing import Optional, Union, List
from torch.utils.data import Dataset

from torch import Tensor
from pandas import DataFrame, Series, to_datetime, to_numeric
from sklearn.preprocessing import MinMaxScaler

class PDETimeDataset(Dataset):
    def __init__(self, df:DataFrame, horizon:int=96, lookback:Optional[int]=None, time_index_name:str='date'):
        self.df = df
        self.horizon = horizon
        self.lookback = lookback if lookback else horizon*5
        self.time_index_name = time_index_name
        # self.features = self.df.copy().drop(columns=time_index).values
        # self.time_index = self.df[time_index].values
        self.scaler = MinMaxScaler()
        self.scaler_time_index = MinMaxScaler()
        self.temporal_feature_names = ['year', 'quarter', 'month', 'day_of_year', 'day_of_week', 'hour', 'minute', 'second']

        self.make_temporal_features(self.time_index_name)
        self.scale_data()

        self.features = self.df.copy().drop(columns=self.time_index_name).drop(columns=self.temporal_feature_names).values
        self.time_index = self.df.copy()[self.time_index_name].values
        self.temporal_features = self.df.copy()[self.temporal_feature_names].values

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
    
    def make_temporal_features(self, time_index_name:Optional[str]=None) -> None:
        """
        Adds temporal features to the given DataFrame based on the specified time index column.

        Parameters:
            time_index (str): The name of the column containing the time index. Default is 'date'.
        """
        time_index_name = time_index_name if time_index_name else self.time_index_name
        for feature in self.temporal_feature_names:
            self.df[feature] = to_datetime(self.df[time_index_name]).dt.__getattribute__(feature)
            if len(self.df[feature].values.unique()) == 1:
                self.df.drop(columns=feature, inplace=True)
                self.temporal_feature_names.remove(feature)

    def scale_data(self) -> None:
        """
        Scales the input DataFrame using MinMaxScaler, in place.
        """
        self.df = to_numeric(self.df)
        self.df[self.temporal_feature_names + [self.time_index_name]] = self.scaler.fit_transform(self.df[self.temporal_feature_names + [self.time_index_name]])

    def unscale_data(self, df: Optional[Union[DataFrame, Tensor]]) -> DataFrame:
        """
        Unscales the input DataFrame using MinMaxScaler.

        Parameters:
            df (DataFrame): The input DataFrame to be unscaled. In the order of temporal features and time index.

        Returns:
            DataFrame: The unscaled DataFrame.
        """
        df = df if df else self.df[self.temporal_feature_names + [self.time_index_name]]
        return self.scaler.inverse_transform(df)
