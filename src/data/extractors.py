import numpy as np
from abc import abstractmethod
from abc import ABC
from typing import List, Tuple, Dict, Optional,Union, Any
import copy

class FeatureExtractor(ABC):
    
    @abstractmethod
    def extract(self, sample: np.ndarray) -> np.ndarray:
        pass

class FlatKeypointsFeatureExtractor(FeatureExtractor):
    
    def __init__(self, root_index: Optional[int]):
        self.root_index = root_index if root_index else None
    
    def extract(self, sample: np.ndarray):
        if self.root_index:
            sample = np.delete(sample, self.root_index, axis=1)
            
        F, K, C = sample.shape
        return sample.reshape(F, K * C) 
    
    
class TemporalFeatureExtractor(FeatureExtractor): 
    root_index = 9
    
    def extract(self, sample: np.ndarray) -> np.ndarray:
        filtered_sample = self._delete_keypoint_data(sample)
        return self._extract_temporal_features(filtered_sample)
    
    @abstractmethod
    def _extract_temporal_features(self, filtered_sample: np.ndarray) -> np.ndarray:
        pass    
    
    def _delete_keypoint_data(self, data: np.ndarray) -> np.ndarray:
        return np.delete(data, self.root_index, axis=1)
    
    
    
    
class LimbLengthExtractor(FeatureExtractor):
    angle_pairs = [
        (0, 1),  # 1. Neck to Right shoulder
        (0, 4),  # 2. Neck to Left shoulder
        (1, 2),  # 3. Right shoulder to Right elbow
        (4, 5),  # 4. Left shoulder to Left elbow
        (2, 3),  # 5. Right elbow to Right wrist
        (5, 6),  # 6. Left elbow to Left wrist
        (3, 7),  # 7. Right wrist to Right hip
        (6, 8)   # 8. Left wrist to Left hip
    ]
    
    def extract(self, sample: np.ndarray) -> np.ndarray:
        F, K, C = sample.shape
        L = len(self.distance_pairs)
        
        all_distances = np.zeros((F, L))
        
        for i, (j, k) in enumerate(self.distance_pairs):
            # Coordenadas del punto j
            xj = sample[:, j, 0]
            yj = sample[:, j, 1]
            
            # Coordenadas del punto k
            xk = sample[:, k, 0]
            yk = sample[:, k, 1]
            
            # Diferencias (xk - xj) y (yk - yj)
            delta_x = xk - xj
            delta_y = yk - yj
            
            # Fórmula de la distancia euclidiana (Ecuación 6 de tu imagen)
            distances = np.sqrt((delta_x ** 2) + (delta_y ** 2))
            
            all_distances[:, i] = distances
            
        return all_distances
        
    def __str__(self):
        return "DistanceBetweenKeypoints"
    
class AngleBetweenLimbsExtractor(FeatureExtractor):
    '''
    Abbreviations:
        C: Channels
        F: Frames
        L: Limbs
    '''
    def __init__(self, angle_triples):
        self.angle_triples = angle_triples 
        

    def compute_body_part_angle(self, limb_lens: np.ndarray):
        """
        limb_lens shape: (L, 3) -> [l1, l2, vl]
        returns: (L,)
        """
        l1 = limb_lens[:, 0]
        l2 = limb_lens[:, 1]
        vl = limb_lens[:, 2]

        # Ley de cosenos
        cos_theta = (l1**2 + l2**2 - vl**2) / (2 * l1 * l2)

        angles = np.arccos(cos_theta)  # en radianes
        return angles
    

    def compute_limb_len(self, keypoints: np.ndarray, angle_triplet: Tuple[int, int, int]):
        """
        keypoints shape: (K, C)
        returns: (3,)
        """
        A, B, C = angle_triplet

        return np.array([
            np.linalg.norm(keypoints[A] - keypoints[B]),
            np.linalg.norm(keypoints[B] - keypoints[C]),
            np.linalg.norm(keypoints[A] - keypoints[C])
        ])
    

    def is_triangle(self, sides: np.ndarray):
        """
        sides shape: (3,)
        """
        s1, s2, s3 = np.sort(sides)
        return s1 < s2 + s3
    

    def extract(self, sample: np.ndarray) -> np.ndarray:
        """
        sample shape: (F, K, C)
        
        returns:
            angles shape: (F, L)
        """
        F = sample.shape[0]
        L = len(self.angle_triples)

        all_angles = []

        for f in range(F):
            keypoints = sample[f]  # (K, C)
            frame_limb_lens = []

            # calcular longitudes
            for triple in self.angle_triples:
                lens = self.compute_limb_len(keypoints, triple)
                
                # validar triángulo
                if self.is_triangle(lens):
                    frame_limb_lens.append(lens)
                else:
                    raise ValueError(f"Error at frame {f}: Keypoint triplet {triple} does not form a valid triangle. Calculated sides: {lens}")

            frame_limb_lens = np.array(frame_limb_lens)  # (L, 3)

            angles = self.compute_body_part_angle(frame_limb_lens)
            all_angles.append(angles)

        return np.array(all_angles)  # (F, L)



class Pipeline: 
    def __init__(self, extractors: List[Any]): # Cambia Any por FeatureExtractor si tienes el tipado
        self.extractors = extractors
        
    def run(self, data: Union[List[Dict], np.ndarray]) -> Union[List[Dict], np.ndarray]:
        
        # ESTRATEGIA 1: Si la entrada es directamente un arreglo de NumPy (keypoints)
        if isinstance(data, np.ndarray):
            features = [] 
            for extractor in self.extractors:
                # Extraemos características directamente del arreglo
                feature = extractor.extract(data)
                features.append(feature)
                
            # Concatenamos y devolvemos solo el arreglo de características
            return np.concatenate(features, axis=1)
            
        # ESTRATEGIA 2: Si la entrada es una lista de diccionarios
        elif isinstance(data, list):
            # Hacemos el deepcopy para no modificar el dataset original
            out_dataset = copy.deepcopy(data)
            
            for sample in out_dataset:
                features = [] 
                for extractor in self.extractors:
                    # Extraemos características accediendo a la llave 'keypoints'
                    feature = extractor.extract(sample['keypoints'])
                    features.append(feature)
                    
                # Guardamos el resultado concatenado en una nueva llave
                sample['handcrafted'] = np.concatenate(features, axis=1)
                
            return out_dataset
            
        # Manejo de errores por si recibe un formato no válido
        else:
            raise TypeError(f"El tipo de dato '{type(data)}' no es soportado. Usa np.ndarray o List[Dict].")