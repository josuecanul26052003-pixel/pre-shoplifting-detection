from src.modules.dataset import HandCraftedDataset
from torch.utils.data import DataLoader
from dataclasses import asdict
from torch.utils.data import Dataset
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv('.config')

class SuspiciousDataModule(LightningDataModule):
    
    def __init__(self,
                 dataset: dict,
                 k: int,
                 seed: int,
                 n_workers: int = 0,
                 batch_size: int = 8):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.dataset = dataset
        self.k = k
        self.samples = dataset['samples']
        self.sources = np.array(self.dataset['sources'])
        self.labels = np.array(self.dataset['labels'])
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.seed=seed
    def setup(self, stage=None):
        if not self.train_dataset and not self.val_dataset:

            n_splits_env = int(os.getenv("N_FOLDS"))
            
            n_splits = int(n_splits_env) if n_splits_env else 5
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            
            all_splits = list(skf.split(self.sources, self.labels))
            train_idxs, val_idxs = all_splits[self.k]
            
            train_sources = self.sources[train_idxs]
            val_sources = self.sources[val_idxs]
            
            train_sources_set = set(train_sources)
            val_sources_set = set(val_sources)
            
            train_samples = [sample for sample in self.samples if sample['source'] in train_sources_set]
            val_samples = [sample for sample in self.samples if sample['source'] in val_sources_set]
                        
            self.train_dataset = HandCraftedDataset(train_samples, train=True)
            self.val_dataset = HandCraftedDataset(val_samples, train=False)
            
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)
        
    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)
        
    def test_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)