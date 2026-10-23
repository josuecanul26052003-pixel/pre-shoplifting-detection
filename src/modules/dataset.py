from abc import ABC
from torch.utils.data import Dataset
from dotenv import load_dotenv
import os
import src.stratergy.sampler as S
load_dotenv('.config')
import numpy as np

class BaseDataset(ABC, Dataset):
    def __init__(self, samples: list, train: bool):
        self.samples = samples
        self.train = train
    
    def __len__(self):
        return len(self.samples)
    
    def get_sampler(self):
        stratergy = os.getenv("SAMPLER_STRATERGY")
        
        if stratergy == 'LAST': 
            return S.LastIndicesSampler()
        elif stratergy == 'RANDOM' and not self.train: 
            return S.RandomUniformSampler()
        elif stratergy == 'RANDOM' and self.train: 
            return S.UniformSampler()
        elif stratergy == 'UNIFORM': 
            return S.UniformSampler()
        else:
            raise ValueError(f"Estrategia de muestreo no reconocida o inválida: '{stratergy}'. " 
                            f"Asegúrate de que 'SAMPLER_STRATERGY' esté correctamente definida en tu .env.")
        

class HandCraftedDataset(BaseDataset):
    def __init__(self, samples, train: bool):
        super().__init__(samples, train)
        self.sample_size = int(os.getenv("SAMPLE_SIZE"))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        sample_stratergy = self.get_sampler() 
        
        available_frames = sample['handcrafted'].shape[0] 
        
        idxs = sample_stratergy.apply(available_frames, self.sample_size)
        
        handcrafted = sample['handcrafted'][idxs]
        descriptor = sample['descriptor'][idxs]
        
        features = np.concatenate([handcrafted, descriptor], axis=1)
        
        return {
            'features': features,        # Revisa cómo lo llamas en model.py, usualmente 'features' o 'data'
            'label': sample['label']     # Asumiendo que tu dict original tiene la llave 'label'
        }
