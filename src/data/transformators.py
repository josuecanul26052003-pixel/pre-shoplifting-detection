from abc import ABC
from abc import abstractmethod
import numpy as np
import random
from scipy.interpolate import interp1d
from src.utils.helpers import compute_height_sample
from typing import Any, List, Dict, Tuple, Union
from typing import Optional

class Transformator(ABC):
    @abstractmethod
    def apply(self, data):
        pass


class Sampler(Transformator):
    def __init__(self, n_frames, strategy="uniform"):
        """
        Inicializa el Sampler.
        
        :param n_frames: Número de frames a extraer.
        :param strategy: 'uniform' para tomar muestras equiespaciadas, 
                         'last' para tomar solo los últimos n_frames.
        """
        self.n_frames = n_frames
        
        # Validamos que la estrategia elegida sea correcta
        if strategy not in ["uniform", "last"]:
            raise ValueError("El parámetro 'strategy' debe ser 'uniform' o 'last'.")
        
        self.strategy = strategy
    
    def apply(self, data: np.ndarray):
        F = data.shape[0]
        
        if self.strategy == "uniform":
            # Estrategia original: frames equiespaciados
            frame_indexes = np.linspace(0, F-1, self.n_frames, dtype=int)
            return data[frame_indexes, :, :]
            
        elif self.strategy == "last":
            # Nueva estrategia: últimos n_frames
            # Nota: Si self.n_frames es mayor que F, simplemente devolverá todos los frames disponibles
            return data[-self.n_frames:, :, :]

class MMactionFormater(Transformator):
    
    def apply(self, data):
        return np.expand_dims(data, axis=0)

class TemporalJumpCutter(Transformator):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def apply(self, data: np.ndarray) -> np.ndarray:
        data = data.copy()

        F, K, C = data.shape

        for k in range(K):
            last_valid = None  # último frame válido

            t = 0
            while t < F:
                point = data[t, k, :2]  # solo x,y

                # Si es NaN, skip
                if np.isnan(point).any():
                    t += 1
                    continue

                if last_valid is None:
                    last_valid = t
                    t += 1
                    continue

                prev_point = data[last_valid, k, :2]
                dist = np.linalg.norm(point - prev_point)

                if dist <= self.threshold:
                    last_valid = t
                    t += 1
                else:
                    # 🚨 salto detectado → buscar recuperación
                    start_bad = t
                    t += 1

                    recovered = False

                    while t < F:
                        next_point = data[t, k, :2]

                        if np.isnan(next_point).any():
                            t += 1
                            continue

                        dist_recovery = np.linalg.norm(next_point - prev_point)

                        if dist_recovery <= self.threshold:
                            # recuperación encontrada
                            recovered = True
                            break

                        t += 1

                    # marcar como NaN todo el bloque malo
                    end_bad = t if recovered else F

                    data[start_bad:end_bad, k, :] = np.nan

                    if recovered:
                        last_valid = t
                        t += 1
                    else:
                        break

        return data

class Interpolator(Transformator):
    
    def __init__(self, kind):
        self.kind = kind
    
    def apply(self, data):
        F, K, C = data.shape
        result = data.copy()

        x_axis = np.arange(F)

        for k in range(K):
            for c in range(C):

                series = data[:, k, c]
                valid = ~np.isnan(series)

                # Necesitamos al menos 2 puntos válidos
                if np.sum(valid) < 1:
                    continue

                interpolator = interp1d(
                    x_axis[valid],
                    series[valid],
                    kind=self.kind,
                    bounds_error=False,
                    fill_value=np.nan
                )

                result[:, k, c] = interpolator(x_axis)

        return result
    
class VirtualKeypoint(Transformator):
    def __init__(
        self,
        groups: List[List[int]],
        remove_original: bool = False
    ):
        """
        groups: lista de listas de índices
            Ej: [[11,12], [5,6]] → generará 2 nuevos keypoints

        remove_original: elimina los índices usados si es True
        """
        self.groups = groups
        self.remove_original = remove_original

    def apply(self, data: np.ndarray) -> np.ndarray:
        _, K, _ = data.shape

        new_points = []
        used_idxs = set()

        # Crear puntos virtuales
        for idxs in self.groups:
            pts = data[:, idxs, :]  # (F, len(idxs), C)

            # Promedio ignorando NaNs
            mean_pt = np.nanmean(pts, axis=1)  # (F, C)

            new_points.append(mean_pt[:, np.newaxis, :])  # (F, 1, C)
            used_idxs.update(idxs)

        # Concatenar nuevos puntos al final
        new_points = np.concatenate(new_points, axis=1)  # (F, N_new, C)
        data_out = np.concatenate([data, new_points], axis=1)

        # Eliminar originales si aplica
        if self.remove_original:
            keep_idxs = [i for i in range(K) if i not in used_idxs]

            # índices de los nuevos puntos (ya al final)
            new_idxs = list(range(K, K + len(self.groups)))

            data_out = data_out[:, keep_idxs + new_idxs, :]

        return data_out

class Smoother(Transformator):

    def __init__(self, filter_fn):
        self.filter_fn = filter_fn

    def apply(self, data):
        F, K, C = data.shape
        result = data.copy()
        for k in range(K):
            for c in range(C):
                series = data[:, k, c]
                if np.isnan(series).any():
                    continue  
                result[:, k, c] = self.filter_fn(series)
        return result
    
class KeypointCutter(Transformator):
    def __init__(self, keypoints_idx):
        self.keypoints_idx = keypoints_idx
    def apply(self, data):
        return np.delete(data, self.keypoints_idx, axis=1)

class FrameNaNCutter(Transformator):
    def apply(self, data: np.ndarray) -> np.ndarray:
        # Asegurar float por seguridad
        data = data.astype(float, copy=False)

        # Detectar NaNs en cualquier canal de cualquier keypoint
        # nan_mask shape: (F,)
        nan_mask = np.isnan(data).any(axis=(1, 2))

        # Nos quedamos con frames que NO tienen NaNs
        valid_frames = ~nan_mask

        return data[valid_frames]
    

class ScoresCutter(Transformator):
    def apply(self, data):
        return data[..., :2]
    
class SkeletonCentering(Transformator):
    def __init__(self, midhip_index=9):
        self.midhip_index=midhip_index
    def apply(self, data):
        centered = data.copy()
        midhip = centered[:, self.midhip_index: self.midhip_index+1, :]
        centered = centered - midhip
        return centered
    
class HeightScaler(Transformator):

    def __init__(self, strategy: Optional[str] = None):
        self._set_strategy(strategy)

    def _set_strategy(self, strategy: Optional[str] = None):
        valid = {None, 'mean', 'median'}
        if strategy not in valid:
            raise ValueError(f"strategy debe ser una de {valid}")
        self.strategy = strategy

    def _compute_scale(self, height: np.ndarray) -> float:
        if self.strategy is None:
            # 🔹 por frame (vector)
            return 1 / height  # (T,)
        
        elif self.strategy == 'mean':
            return 1 / np.mean(height)  # escalar
        
        elif self.strategy == 'median':
            return 1 / np.median(height)  # escalar

    def apply(self, data):
        result = data.copy()

        height = compute_height_sample(result)  # (T,)
        scale = self._compute_scale(height)

        if isinstance(scale, np.ndarray):
            # reshape para broadcasting (T, 1, 1)
            scale = scale[:, None, None]

        # aplicar solo a x,y
        result[:, :, :2] *= scale

        return result

class Rotator(Transformator):
      
    def __init__(self, p=0.5, degrees: Tuple[int, int] = (-15, 15)):
        self.p = p
        self.degrees = degrees
            
    def apply(self, data):
        
        if random.random() > self.p:
            return data
        
        low, high = self.degrees
        angle = np.random.uniform(low, high)
        # angle = np.random.randint(low, high+1)
        theta = np.radians(angle)
        cos_val = np.cos(theta)
        sin_val = np.sin(theta)
        rotation_matrix = np.array([
            [cos_val, -sin_val],
            [sin_val,  cos_val]
        ])
        return np.dot(data, rotation_matrix.T)
    

# class Rotator(Transformator):
      
#     def __init__(self, angle: int):
#         self.angle = angle
            
#     def apply(self, data):
        
#         # angle = np.random.randint(low, high+1)
#         theta = np.radians(self.angle)
#         cos_val = np.cos(theta)
#         sin_val = np.sin(theta)
#         rotation_matrix = np.array([
#             [cos_val, -sin_val],
#             [sin_val,  cos_val]
#         ])
#         return np.dot(data, rotation_matrix.T)
    
  
class Flipper(Transformator):
    
    def __init__(self, p=0.5, axis='horizontal'):
        """
        :param p: Probabilidad de que se aplique la transformación.
        :param axis: 'horizontal' (invierte el eje X) o 'vertical' (invierte el eje Y).
        """
        self.p = p
        self.axis = axis
        
    def apply(self, data):
        # Aplicar con probabilidad p
        if random.random() > self.p:
            return data
        
        # Definir la matriz de transformación según el eje
        if self.axis == 'horizontal':
            # Invierte X, mantiene Y
            flip_matrix = np.array([
                [-1,  0],
                [ 0,  1]
            ])
        elif self.axis == 'vertical':
            # Mantiene X, invierte Y
            flip_matrix = np.array([
                [ 1,  0],
                [ 0, -1]
            ])
        else:
            raise ValueError("El parámetro 'axis' debe ser 'horizontal' o 'vertical'")
            
        return np.dot(data, flip_matrix.T)
    
class Pipeline:
    def __init__(self, steps: List[Any]): # Reemplaza Any por Transformator si lo tienes importado
        self.steps = steps
        
    def run(self, samples: Union[List[Dict], np.ndarray]) -> Union[List[Dict], np.ndarray]:
        
        # CASO 1: Si la entrada es directamente un arreglo de NumPy
        if isinstance(samples, np.ndarray):
            output_data = samples.copy()
            
            for step in self.steps:
                output_data = step.apply(output_data)
                
            return output_data
            
        # CASO 2: Si la entrada es una lista de diccionarios
        elif isinstance(samples, list):
            # Copiamos la lista y los diccionarios internos para no mutar los originales
            output_samples = [sample.copy() for sample in samples]
            
            for sample in output_samples:
                # Extraemos los keypoints del diccionario actual
                data = sample['keypoints']
                
                # Pasamos los datos por cada paso del pipeline
                for step in self.steps:
                    data = step.apply(data)
                    
                # Reasignamos el resultado al diccionario
                sample['keypoints'] = data
                
            return output_samples
            
        # Manejo de errores por si se envía un tipo de dato no esperado
        else:
            raise TypeError(f"El tipo de dato '{type(samples)}' no es soportado por el Pipeline.")