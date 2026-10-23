from abc import abstractmethod
import numpy as np
from abc import ABC

        
class Sampler(ABC):
    def apply(self, segment_size: int, sample_size: int) -> list[int]:
        if sample_size > segment_size:
            raise ValueError(f"El tamaño de la muestra ({sample_size}) no puede ser mayor al tamaño del segmento ({segment_size}).")
        
        return self._apply(segment_size, sample_size)

    @abstractmethod
    def _apply(self, segment_size: int, sample_size: int) -> list[int]: 
        pass

class RandomUniformSampler(Sampler):
    def _apply(self, segment_size: int, sample_size: int) -> list[int]:
        seg_size = segment_size / sample_size
        idxs = [int(i * seg_size + np.random.uniform(0, seg_size)) for i in range(sample_size)]
        return np.clip(idxs, 0, segment_size - 1).astype(int).tolist()

class UniformSampler(Sampler):
    def _apply(self, segment_size: int, sample_size: int) -> list[int]:
        return np.linspace(0, segment_size - 1, sample_size, dtype=int).tolist()
    
class LastIndicesSampler(Sampler):
    def _apply(self, segment_size: int, sample_size: int) -> list[int]:
        start_idx = segment_size - sample_size
        return list(range(start_idx, segment_size))

class SampleSegmenter:
    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def process_segment(self, target_size: int, sample: np.ndarray, axis: int = 0) -> np.ndarray:
        available_frames = sample.shape[axis]
        
        indices = self.sampler.apply(segment_size=available_frames, sample_size=target_size)
        
        sampled_data = np.take(sample, indices, axis=axis)
        
        return sampled_data