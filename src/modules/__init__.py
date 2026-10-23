from torch.utils.data import DataLoader
from dataclasses import asdict
from torch.utils.data import Dataset
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from typing import List, Optional, Any

class DatasetConfig:
    dataset_path: str
    n_frames: int
    modalities: List[str] 
    extraction: Optional[Any]

class SuspiciousDataModule(LightningDataModule):
    
    def __init__(self, 
                 dataset_config: DatasetConfig,
                 dataset_info: str,
                 augmentation: list,
                 k: int,
                 n_folds: int,
                 seed: int = 42,
                 n_workers: int = 0,
                 batch_size: int = 8):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.augmentation = augmentation
        self.dataset_config = dataset_config
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.dataset_info = pd.read_csv(dataset_info)
        
    def setup(self, stage=None):
        if not self.train_dataset and not self.val_dataset:
            labels = self.dataset_info['label'].to_numpy()
            sources = self.dataset_info['source'].to_numpy()
            
            skf = StratifiedKFold(n_splits=self.hparams.n_folds, shuffle=True, random_state=self.hparams.seed)
            
            all_splits = [k for k in skf.split(sources, labels)]
            train_idxs, val_ixds = all_splits[self.hparams.k]
            train_sources, val_sources = sources[train_idxs], sources[val_ixds]
            
            base_kwargs = asdict(self.dataset_config)
            self.train_dataset = SuspiciousDataset(train=True, idxs=train_sources, **base_kwargs, augmentation=self.augmentation)
            
            self.val_dataset = SuspiciousDataset(train=True, idxs=val_sources, **base_kwargs, augmentation=None)
            
    def train_dataloader(self): return DataLoader(self.train_dataset, self.hparams.batch_size, True)
        
    def val_dataloader(self): return DataLoader(self.val_dataset, self.hparams.batch_size, False)
        
    def test_dataloader(self): return DataLoader(self.val_dataset, self.hparams.batch_size, False)