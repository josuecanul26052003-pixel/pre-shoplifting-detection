import numpy as np
from abc import abstractmethod
from abc import ABC
from typing import List, Tuple, Dict, Optional,Union, Any
import copy
import cv2
from pathlib import Path
from sklearn.metrics import pairwise_distances_argmin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize


class KeypointFeatureExtractor(ABC):
    
    @abstractmethod
    def extract(self, sample: np.ndarray) -> np.ndarray:
        pass

class Codebook:
    def __init__(self, cluster: BaseEstimator, X_train: np.ndarray):
        self.cluster = cluster
        self.centroids = None
        
        # Al inicializar, armamos el clúster automáticamente
        self.setup(X_train)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Aplica normalización L2 a los vectores crudos."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return normalize(X, norm='l2')

    def setup(self, X_train: np.ndarray):
        """Entrena el modelo de clustering y prepara los centroides."""
        if X_train is None or len(X_train) == 0:
            raise ValueError("Se requiere un 'X_train' válido para armar el Codebook.")
            
        # PASO AÑADIDO: Normalizamos los datos de entrenamiento
        X_train_norm = self._normalize(X_train)
            
        # 1. Entrenamos el algoritmo con los descriptores normalizados
        self.cluster.fit(X_train_norm)
        
        # 2. Una vez entrenado, extraemos o calculamos los centroides
        self.__set_centroids(X_train_norm)

    def __set_centroids(self, X_train: np.ndarray):
        """Retorna los centroides si el clúster los tiene, o los calcula si no."""
        # 1. Si el algoritmo provee los centroides (ej. KMeans), los extraemos.
        if hasattr(self.cluster, 'cluster_centers_'):
            self.centroids = self.cluster.cluster_centers_
            
        # 2. Si no los provee (ej. DBSCAN o AgglomerativeClustering), los calculamos.
        else:
            # Obtenemos las etiquetas asignadas en el entrenamiento
            labels = self.cluster.labels_
            # Extraemos los clústeres válidos (DBSCAN usa -1 para ruido, lo ignoramos)
            cluster_ids = set(labels) - {-1}
            
            if not cluster_ids:
                raise ValueError("El algoritmo de clustering consideró todos los puntos como ruido.")
                
            # Inicializamos una matriz para guardar los centroides
            max_label = max(cluster_ids)
            self.centroids = np.zeros((max_label + 1, X_train.shape[1]))
            
            # Calculamos la media (centroide) para cada clúster
            for cid in cluster_ids:
                puntos_cluster = X_train[labels == cid]
                self.centroids[cid] = np.mean(puntos_cluster, axis=0)

        return self.centroids

    def predict(self, x: np.ndarray):
        """Retorna el clúster al que pertenece x y su centroide correspondiente."""
        # Guardamos si el input original era 1D para saber cómo formatear la salida
        es_un_solo_elemento = (x.ndim == 1)
        
        # PASO AÑADIDO: Normalizamos los nuevos descriptores crudos
        x_norm = self._normalize(x)
            
        # Calculamos los labels usando los datos normalizados
        if hasattr(self.cluster, 'predict'):
            labels = self.cluster.predict(x_norm)
        else:
            labels = pairwise_distances_argmin(x_norm, self.centroids)
            
        # Obtenemos los centroides usando las etiquetas como índices
        centroids = self.centroids[labels]
        
        # Si pasaste un solo elemento (1D), devolvemos el valor entero y el vector 1D
        if es_un_solo_elemento:
            return labels[0], centroids[0]
            
        # Si pasaste una matriz 2D (varios datos a la vez), devolvemos los arreglos completos
        return labels, centroids

class PostVLADNorm(ABC):
    @abstractmethod
    def apply_norm(samples: np.ndarray):
        pass

class VLAD:
    def __init__(self, codebook):
        self.codebook = codebook
        self.centroids = codebook.centroids
    
    def compute(self, descriptors: np.ndarray) -> np.ndarray:
        K, d = self.centroids.shape

        # 1. Manejo de casos donde no hay descriptores (ej. máscara de SAM vacía)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(K * d)
        
        # 2. Asignación a clústeres
        # Nota: Asegúrate de que tu Codebook.predict devuelva (labels, centros_asignados)
        labels, centroids_asignados = self.codebook.predict(descriptors)

        # 3. Cálculo de residuos
        residues = descriptors - centroids_asignados

        # 4. Agregación (suma de residuos por clúster)
        vlad_matrix = np.zeros((K, d))
        np.add.at(vlad_matrix, labels, residues)

        # 5. Normalización y aplanado
        vlad_vector_final = self.normalize_vlad(vlad_matrix)

        return vlad_vector_final

    def normalize_vlad(self, vlad_matrix: np.ndarray) -> np.ndarray:
        """
        Aplica Intra-normalización (por clúster) y luego normalización global.
        """
        # 1. Intra-normalización L2: normaliza cada fila independientemente
        # Evita que un clúster con muchos descriptores domine todo el vector
        vlad_matrix_intra = normalize(vlad_matrix, norm='l2')
        
        # 2. Aplanamos la matriz a un vector 1D (tamaño K * d)
        vlad_vector = vlad_matrix_intra.flatten()
        
        # 3. Normalización L2 global final sobre todo el vector
        vlad_vector_norm = normalize(vlad_vector.reshape(1, -1), norm='l2')[0]
        
        return vlad_vector_norm

class VideoFrameVLADDescriptor:
    def __init__(self, segmentator=None, vlad=None):
        """
        :param segmentator: Objeto o función que recibe una imagen (RGB) y retorna una máscara binaria (uint8). 
                            Si es None, no se realiza segmentación y se extraen características de todo el bbx.
        :param vlad: Instancia de una subclase de VLAD (ej. IntraNormalizedVLAD).
        """
        self.segmentator = segmentator 
        self.vlad = vlad
        # Inicializamos el extractor de descriptores locales (SIFT por defecto)
        self.extractor = cv2.SIFT_create()

    def compute_vlad_frames(self, video_path: str, sample: dict, output_video_path: str = None):
            """
            :param video_path: Ruta al archivo de video.
            :param sample: Diccionario que contiene 'bbox' (F, 4) y 'frame_indices' (F).
            :param output_video_path: (Opcional) Ruta para guardar un video con la visualización de los descriptores.
            :return: Diccionario con {timestamp_segundos: vector_vlad_1D}
            """
            bboxes = sample['bbox']
            indices = sample['frame_indices']
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"No se pudo abrir el video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            vlad_sequence = {}
            null_indices = []
            
            # 1. Configurar el guardado del video si se solicita
            out_video = None
            if output_video_path is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec mp4
                out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # 2. Iterar sobre el array de índices y sus correspondientes bboxes (Tamaño F)
            for idx_counter, (frame_idx, current_bbx) in enumerate(zip(indices, bboxes)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Advertencia: No se pudo leer el frame {frame_idx} en el índice del array {idx_counter}.")
                    break

                timestamp = frame_idx / fps

                # Validación 1: Si no hay bbx válida en este índice
                if current_bbx is None or len(current_bbx) == 0 or np.isnan(current_bbx).any():
                    vlad_sequence[timestamp] = None
                    null_indices.append(frame_idx)
                    
                    if out_video:
                        out_video.write(frame)
                    continue

                # Extracción de coordenadas del bounding box (formato xyxy)
                x1, y1, x2, y2 = map(int, current_bbx)
                
                try:
                    # Asegurarnos de que el recorte no se salga de las dimensiones de la imagen
                    x1_crop, y1_crop = max(0, x1), max(0, y1)
                    x2_crop, y2_crop = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                    
                    if cropped_frame.size == 0:
                        raise ValueError(f"Bounding box inválido (fuera de límites o área 0) en el frame {frame_idx}")

                    gray_cropped = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

                    # ==========================================
                    # EVALUACIÓN DE MÁSCARA O RECORTE COMPLETO
                    # ==========================================
                    if self.segmentator is not None:
                        # Segmentación
                        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                        mask_output = self.segmentator(cropped_rgb)

                        # SOLUCIÓN DE TIPO DE DATO PARA LA MÁSCARA
                        if isinstance(mask_output, list) and hasattr(mask_output[0], 'masks'):
                            if mask_output[0].masks is not None:
                                mask = mask_output[0].masks.data[0].cpu().numpy()
                            else:
                                mask = np.zeros((cropped_frame.shape[0], cropped_frame.shape[1]), dtype=np.uint8)
                        elif hasattr(mask_output, 'cpu'):
                            mask = mask_output.cpu().numpy()
                        else:
                            mask = np.array(mask_output)

                        # Escalar la máscara a 255 si está en formato 0-1
                        if mask.max() <= 1.0:
                            mask = (mask * 255)
                            
                        # Asegurar formato uint8 necesario para SIFT
                        mask = mask.astype(np.uint8)
                        
                        # Redimensionar la máscara al tamaño exacto del crop
                        if mask.shape != (cropped_frame.shape[0], cropped_frame.shape[1]):
                            mask = cv2.resize(mask, (cropped_frame.shape[1], cropped_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                        # Extracción de descriptores SIFT con máscara
                        keypoints, descriptors = self.extractor.detectAndCompute(gray_cropped, mask=mask)
                        if descriptors is None or len(descriptors) == 0:
                            raise ValueError("0 descriptores extraídos tras la segmentación.")
                    else:
                        # Extracción de descriptores SIFT de todo el bounding box sin máscara
                        keypoints, descriptors = self.extractor.detectAndCompute(gray_cropped, mask=None)
                        if descriptors is None or len(descriptors) == 0:
                            raise ValueError("0 descriptores extraídos de la caja delimitadora (sin máscara).")

                    # Unificación con VLAD
                    vlad_vector = self.vlad.compute(descriptors)
                    vlad_sequence[timestamp] = vlad_vector

                    # 3. Dibujar resultados para el video si está activado
                    if out_video:
                        vis_frame = frame.copy()
                        cv2.rectangle(vis_frame, (x1_crop, y1_crop), (x2_crop, y2_crop), (0, 255, 0), 2)
                        
                        adjusted_keypoints = []
                        for kp in keypoints:
                            adj_kp = cv2.KeyPoint(kp.pt[0] + x1_crop, kp.pt[1] + y1_crop, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                            adjusted_keypoints.append(adj_kp)
                            
                        cv2.drawKeypoints(vis_frame, adjusted_keypoints, vis_frame, color=(255, 0, 0), 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        
                        out_video.write(vis_frame)

                except Exception as e:
                    # MANEJO DE ERRORES Y GENERACIÓN DE LOGS
                    import os
                    import datetime
                    
                    os.makedirs("frame_logs", exist_ok=True)
                    
                    # 1. Escribir error detallado en el archivo .log
                    log_file_path = os.path.join("frame_logs", "errores_procesamiento.log")
                    tiempo_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(f"[{tiempo_actual}] Archivo: {video_path} | Frame: {frame_idx} | BBox: [{x1}, {y1}, {x2}, {y2}] | Error: {str(e)}\n")
                    
                    # 2. Crear imagen de depuración (solo con frame y bbox visual)
                    error_img = frame.copy()
                    cv2.rectangle(error_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    info_text1 = f"FRAME ERROR: {frame_idx}"
                    info_text2 = f"BBOX: [{x1}, {y1}, {x2}, {y2}]"
                    
                    cv2.putText(error_img, info_text1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(error_img, info_text2, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    log_img_filename = os.path.join("frame_logs", f"error_frame_{frame_idx}.jpg")
                    cv2.imwrite(log_img_filename, error_img)
                    
                    # Liberar recursos antes de hacer el crash
                    cap.release()
                    if out_video:
                        out_video.release()
                    
                    # Relanzar el error indicando dónde revisar los logs
                    raise RuntimeError(f"Fallo en el frame {frame_idx}. Detalles guardados en '{log_file_path}' y la imagen en '{log_img_filename}'.") from e

            # Liberar recursos (ejecución exitosa)
            cap.release()
            if out_video:
                out_video.release()
                
            return vlad_sequence
            
    def _validate_null_sequence(self, sequence: list):
        """
        Verifica que los None solo aparezcan al principio o al final de la secuencia.
        Lanza ValueError si detecta un gap en medio de detecciones válidas.
        """
        first_valid = -1
        last_valid = -1

        # Encontramos el primer y último índice que no son None
        for i, val in enumerate(sequence):
            if val is not None:
                if first_valid == -1:
                    first_valid = i
                last_valid = i

        # Si toda la secuencia es nula o no hay suficientes frames válidos, no hay gaps en medio
        if first_valid == -1 or first_valid == last_valid:
            return

        # Revisamos si hay algún None entre el primer y el último válido
        for i in range(first_valid, last_valid + 1):
            if sequence[i] is None:
                raise ValueError(f"Secuencia interrumpida: Se encontró un valor nulo en el índice {i}, "
                                 f"el cual está en medio de la secuencia válida ({first_valid} a {last_valid}).")
            
class FlatKeypointsFeatureExtractor(KeypointFeatureExtractor):
    
    def __init__(self, root_index: Optional[int]):
        self.root_index = root_index if root_index else None
    
    def extract(self, sample: np.ndarray):
        if self.root_index:
            sample = np.delete(sample, self.root_index, axis=1)
            
        F, K, C = sample.shape
        return sample.reshape(F, K * C) 
    
    
class TemporalFeatureExtractor(KeypointFeatureExtractor): 
    root_index = 9
    
    def extract(self, sample: np.ndarray) -> np.ndarray:
        filtered_sample = self._delete_keypoint_data(sample)
        return self._extract_temporal_features(filtered_sample)
    
    @abstractmethod
    def _extract_temporal_features(self, filtered_sample: np.ndarray) -> np.ndarray:
        pass    
    
    def _delete_keypoint_data(self, data: np.ndarray) -> np.ndarray:
        return np.delete(data, self.root_index, axis=1)
    
    
    
    
class LimbLengthExtractor(KeypointFeatureExtractor):
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
    
class AngleBetweenLimbsExtractor(KeypointFeatureExtractor):
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