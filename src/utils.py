import h5py
import numpy as np
import pickle
from pathlib import Path

def save_dataset(save_path: str, dataset: list[dict], filename: str = 'dataset.pkl'):
    save_path = Path(save_path)
    final_path = save_path / filename
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_path, 'wb') as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_database(dt_path: str) -> list[dict]:
    with open(dt_path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset

def load_keypoint_data_by_id(file_path: str, target_person: int) -> np.ndarray:
    with h5py.File(file_path, 'r') as f:
        if 'frames' not in f:
            raise KeyError("El archivo no contiene el grupo 'frames'.")
            
        frames_group = f['frames']
        frame_names = sorted(frames_group.keys())
        T = len(frame_names)
        
        if T == 0:
            return np.zeros((0, 0, 0)) # T, V, C
            
        V, C = 0, 0
        for frame_name in frame_names:
            for person_name in frames_group[frame_name].keys():
                if 'keypoints' in frames_group[frame_name][person_name]:
                    shape = frames_group[frame_name][person_name]['keypoints'].shape
                    V, C = shape[0], shape[1]
                    break
            if V > 0:
                break
                
        if V == 0 or C == 0:
            return np.zeros((T, 0, 0)) 
            
        tensor = np.full((T, V, C), np.nan)
        
        target_person_name = f"person_{target_person}" 
        for t_idx, frame_name in enumerate(frame_names):
            frame_data = frames_group[frame_name]
            
            if target_person_name in frame_data and 'keypoints' in frame_data[target_person_name]:
                tensor[t_idx, :, :] = np.array(frame_data[target_person_name]['keypoints'])

    return tensor