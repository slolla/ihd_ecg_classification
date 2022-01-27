from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ECGDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, leads, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with paths.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [0, 1]
        self.leads = leads

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        ecg = np.load(filename)
        ecg = ecg[self.leads]
        ecg = ecg.astype(np.float32)
        label = self.data.iloc[idx, 1]
        # label = label.astype(np.float64)
        return ecg, label


ecg_dataset = ECGDataset(
    csv_file="WFDB_StPetersburgtrain_df.csv",
    root_dir="/om2/user/sadhana/time-series-data/",
    leads=[1, 10],
)
dataloader = DataLoader(ecg_dataset, batch_size=1, shuffle=True, num_workers=0)
print(dataloader.next())

