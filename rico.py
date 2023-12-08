
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from typing import Optional, List
import numpy as np
import os

class RICOFullDataset(Dataset):
    """Implements a Dataset class on all elements from the specified dataframe"""
    def __init__(self, data: pd.DataFrame, ori_seq_len:int, tgt_seq_len: Optional[int] = None, stride: int = 1, channels:List[str] =[], standardize: bool = True):
        self.ori_seq_len = ori_seq_len
        self.tgt_seq_len = tgt_seq_len if tgt_seq_len else ori_seq_len
        self.stride = stride
        self.data = data[channels] if channels else data
        self.num_series, self.series_length = data.shape
        self.scaler = StandardScaler()
        self.channels = self.data.columns
        self.n_channels = len(self.data.columns)

        if standardize: self.data = torch.tensor(self.scaler.fit_transform(self.data.values), dtype=torch.float32)

        assert self.ori_seq_len <= len(data), ValueError(f'Seq_len {self.ori_seq_len} should be lower than series length {len(data)}')
        self._len = int(len(data) / self.ori_seq_len) * ((self.ori_seq_len - self.tgt_seq_len )//self.stride + 1)
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        series_index = idx // ((self.ori_seq_len - self.tgt_seq_len )//self.stride + 1) 
        pos_in_series = idx - series_index*((self.ori_seq_len - self.tgt_seq_len)//self.stride + 1)

        return self.__getserie__(series_index)[pos_in_series:pos_in_series + self.tgt_seq_len]

    def __getserie__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for length {len(self)}. Please ensure idx < length.")
        start = self.ori_seq_len * idx
        end = start + self.ori_seq_len
        return self.data[start:end]

class RICODataset:
    """
    A dataset similar to RicoFullDataset with a split specified by its kind ('train', 'test, 'val', 'full') and og size specified by the `ranges` dictionary ('default' -> 0.7, 0.15, 0.15)

    Returns:
    'RICODataset' of specified type
    """
    ranges = {
        'train':0.7,
        'test': 0.15,
        'val': 0.15
              }
    def __init__(self, dataset:RICOFullDataset, kind:str, ranges='default', get_every=1) -> None:
        self.kind = kind
        self.get_every = get_every
        if ranges != 'default':
            self.ranges = ranges
        
        if kind == 'train':
            start = 0
            end = int(self.ranges['train']* len(dataset))
        elif kind == 'val':
            start = int(self.ranges['train'] * len(dataset))
            end = int((self.ranges['train'] + self.ranges['val']) * len(dataset))
        elif kind == 'test':
            start = int((self.ranges['train'] + self.ranges['val']) * len(dataset))
            end = len(dataset)
        elif kind == 'full':
            start = 0
            end = len(dataset)
        else:
            raise ValueError(f'kind should be one of "train", "val", "test" or "full" but got {kind}')

        self.data = torch.stack([dataset[i][::self.get_every] for i in range(start, end)])
        self._len = len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return self._len
    def get_kind(self) -> str:
        return self.kind
    
def create_dummy_data(n_series: int, seq_length: int, amp_range: tuple = (0.1, 0.9), freq_range: tuple = (1, 1)) -> np.ndarray:
    """
    Create a series of sinusoids each with a random phase, an amplitude contained within a range and of specified sequence length.

    Parameters:
    n_series (int): Number of series to generate.
    seq_length (int): Length of each series.
    amp_range (tuple): Tuple containing the minimum and maximum amplitude.
    freq_range (tuple): Tuple containing the minimum and maximum frequency.

    Returns:
    np.ndarray: A numpy array of shape (n_series, seq_length).
    """
    phases = np.random.uniform(0, 2*np.pi, size=n_series)  # random phase
    amplitudes = np.random.uniform(amp_range[0], amp_range[1], size=n_series)  # random amplitude
    frequencies = np.random.uniform(freq_range[0], freq_range[1], size=n_series)  # random frequency
    x = np.linspace(0, 2*np.pi, seq_length)  # x values
    # create 2D array of shape (n_series, seq_length)
    series = torch.from_numpy(np.array([amplitudes[i]*np.sin(frequencies[i]*x + phases[i]) for i in range(n_series)])).float()
    return series.unsqueeze(2)

def data_to_tsv(data: RICODataset, dir:str, name: str, split=(0.7, 0.85)) -> None:
    """
    Save a numpy array to two tsv file (0.7 train, 0.15 test) by default

    Parameters:
    data (np.ndarray): Numpy array to save.
    path (str): Path to save the csv file.
    split (tuple):
        Numbers between 0 and 1 defining the splitting coefficient (eg. `(0.7, 1.0)`)
        If split = 1, then only one full dataset will be exported
    """
    
    array = [row.squeeze().numpy() for row in data]
    df = pd.DataFrame(array)
    if split == 1:
        df.to_csv(os.path.join(dir, name + '_full.tsv'))
        return

    train_end = split[0]
    test_end = split[1]

    train_df = df.iloc[:int(train_end*len(df))]
    test_df = df.iloc[int(train_end*len(df)):int(test_end*len(df))]

    # Managing paths
    if not os.path.exists(dir):
        os.makedirs(dir)

    train_path = os.path.join(dir, name + '_TRAIN.tsv')
    test_path = os.path.join(dir, name + '_TEST.tsv')

    # Saving
    train_df.to_csv(train_path, header=None, sep='\t', index=False)
    test_df.to_csv(test_path, header=None, sep='\t', index=False)
