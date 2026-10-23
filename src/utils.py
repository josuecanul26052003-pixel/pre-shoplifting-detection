import h5py
import numpy as np
import pickle
import imageio_ffmpeg
from pathlib import Path
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

def plot_confusion_matrix(directorio_version):
    """
    Lee los verdaderos/falsos positivos y negativos de metrics.csv de cada fold,
    genera una imagen de la matriz de confusión y la guarda en la carpeta del fold.
    """
    ruta_base = Path(directorio_version)
    directorios_folds = [d for d in ruta_base.iterdir() if d.is_dir()]
    
    for dir_fold in directorios_folds:
        archivo_metricas = dir_fold / 'metrics.csv'
        
        if not archivo_metricas.exists():
            raise FileNotFoundError('El archivo de las métricas no existe')
            
        df = pd.read_csv(archivo_metricas)
        # Tomamos la última fila
        valores = df.iloc[-1]
        
        # Extraemos los valores de la matriz
        # Validamos que las columnas existan para evitar errores
        columnas_requeridas = ['test_TN', 'test_FP', 'test_FN', 'test_TP']
        if not all(col in valores for col in columnas_requeridas):
            print(f"Faltan columnas de la matriz en {archivo_metricas}")
            continue
            
        tn = valores['test_TN']  # Verdadero Normal (Real: Normal, Predicho: Normal)
        fp = valores['test_FP']  # Falso Sospechoso (Real: Normal, Predicho: Sospechoso)
        fn = valores['test_FN']  # Falso Normal (Real: Sospechoso, Predicho: Normal)
        tp = valores['test_TP']  # Verdadero Sospechoso (Real: Sospechoso, Predicho: Sospechoso)
        
        matriz = np.array([[tn, fp], 
                           [fn, tp]])
        
        plt.figure(figsize=(6, 5))
        
        ax = sns.heatmap(matriz, annot=True, fmt='g', cmap='Blues',
                         xticklabels=['Normal', 'Sospechoso'],
                         yticklabels=['Normal', 'Sospechoso'])
        
        plt.title(f'Matriz de Confusión - {dir_fold.name}')
        plt.ylabel('Clase Real')
        plt.xlabel('Clase Predicha')
        
        ruta_imagen = dir_fold / 'confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(ruta_imagen, dpi=300)
        plt.close()
        
        print(f"Matriz de confusión guardada en: {ruta_imagen}")
def compute_stadistics(directorio_version):
    ruta_base = Path(directorio_version)
    archivos_metricas = list(ruta_base.glob('*/metrics.csv'))
    
    if not archivos_metricas:
        print(f"No se encontraron archivos metrics.csv en los subdirectorios de {directorio_version}")
        return
    
    lista_df_folds = []
    
    for archivo in archivos_metricas:
        df = pd.read_csv(archivo)
        ultima_fila = df.iloc[-1:] 
        lista_df_folds.append(ultima_fila)
        
    # Concatenamos todos los resultados de los folds
    df_todos_folds = pd.concat(lista_df_folds, ignore_index=True)
    
    # --- Filtrado de columnas ---
    # 1. Definimos las columnas de la matriz de confusión que queremos excluir
    columnas_matriz = ['test_TN', 'test_FP', 'test_FN', 'test_TP']
    
    # 2. Filtramos: conservamos solo si tiene 'test' en el nombre Y no está en la lista de exclusión
    columnas_testing = [col for col in df_todos_folds.columns 
                        if 'test' in col and col not in columnas_matriz]
    
    # Aplicamos el filtro al DataFrame
    df_filtrado = df_todos_folds[columnas_testing]
    # -----------------------------------
    
    # --- NUEVO: Métricas como filas y estadísticos como columnas ---
    # agg() calcula todo de una vez. Al usar .T (transponer), las métricas pasan a ser filas.
    df_estadisticos = df_filtrado.agg(['mean', 'std', 'min', 'max']).T
    
    # Le damos un nombre al índice para que la columna de métricas tenga un encabezado en el CSV
    df_estadisticos.index.name = 'metric'
    
    # Guardamos en la carpeta padre (Version)
    ruta_guardado = ruta_base / 'results.csv'
    
    # IMPORTANTE: Ahora index=True para que se guarden los nombres de las métricas en el CSV
    df_estadisticos.to_csv(ruta_guardado, index=True) 
    print(f"Estadísticos guardados exitosamente en: {ruta_guardado}")

def plot_metric_from_csv(csv_path: str, metric_name: str):
    """
    Lee el CSV generado por el logger de Lightning, genera una gráfica comparativa
    de train vs val para la métrica indicada, y guarda la imagen en la misma carpeta.
    
    Args:
        csv_path (str): Ruta completa al archivo metrics.csv.
        metric_name (str): Nombre base de la métrica (ej: 'loss', 'accuracy', 'f1').
    """
    # 1. Convertimos la ruta a un objeto Path para manejarla fácilmente
    path = Path(csv_path)
    
    if not path.exists():
        print(f"❌ Error: No se encontró el archivo en {csv_path}")
        return

    # 2. Leemos el CSV
    df = pd.read_csv(path)
    
    # Agrupamos por época y promediamos para alinear los logs de train y val
    # Esto elimina los NaNs generados porque Lightning loggea en diferentes momentos
    epoch_stats = df.groupby('epoch').mean()
    
    # 3. Identificamos los nombres exactos de las columnas
    train_col = f'train_{metric_name}'
    val_col = f'val_{metric_name}'
    
    # Verificamos si al menos una de las métricas existe en el CSV
    if train_col not in epoch_stats.columns and val_col not in epoch_stats.columns:
        print(f"⚠️ Advertencia: No se encontraron columnas para '{metric_name}' en el CSV.")
        print(f"Columnas disponibles: {list(epoch_stats.columns)}")
        return

    # 4. Generamos la gráfica
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if train_col in epoch_stats.columns:
        ax.plot(epoch_stats.index, epoch_stats[train_col], label=f'Train {metric_name}', marker='o', linewidth=2)
        
    if val_col in epoch_stats.columns:
        ax.plot(epoch_stats.index, epoch_stats[val_col], label=f'Val {metric_name}', marker='s', linewidth=2)
        
    ax.set_title(f'{metric_name.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    # 5. Definimos la ruta de salida en la misma carpeta que el CSV
    # path.parent obtiene el directorio donde está el csv
    output_filename = path.parent / f"{metric_name}_evolution.png"
    
    # Guardamos la imagen
    fig.savefig(output_filename, bbox_inches='tight', dpi=300)
    
    # IMPORTANTE: Cerramos la figura para liberar memoria RAM
    plt.close(fig)
    
    print(f"✅ Gráfica guardada en: {output_filename}")

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

def load_keypoints_data_by_id(file_path: str, target_person: int) -> np.ndarray:
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
def render_normalizate_skeleton(secuencia, ancho, alto, archivo_salida='esqueletos.mp4', fps=30, conexiones=None):
    """
    Genera un vídeo con ejes -X a X y -Y a Y, con el eje Y invertido.
    
    Parámetros:
    - secuencia: numpy.ndarray (F, K, 2).
    - ancho: El plano irá de -ancho a +ancho en X.
    - alto: El plano irá de -alto a +alto en Y.
    """
    F, K, C = secuencia.shape
    
    # Configuración de la figura
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Definir los límites para incluir los ejes negativos
    ax.set_xlim(-ancho, ancho)
    ax.set_ylim(-alto, alto)
    
    # 2. Invertir el eje Y (estilo Computer Vision / Matriz de imagen)
    # Esto hace que los valores positivos de Y crezcan hacia ABAJO.
    ax.invert_yaxis()
    
    # Estética de plano cartesiano con origen centrado opcional
    ax.axhline(0, color='black', linewidth=1, alpha=0.3) # Eje X
    ax.axvline(0, color='black', linewidth=1, alpha=0.3) # Eje Y
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_title("Plano Cartesiano Invertido (Cuatro Cuadrantes)")

    puntos, = ax.plot([], [], 'go', markersize=5) # Puntos verdes
    lineas = []
    if conexiones:
        for _ in conexiones:
            linea, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7) # Líneas azules
            lineas.append(linea)

    def init():
        puntos.set_data([], [])
        for l in lineas: l.set_data([], [])
        return [puntos] + lineas

    def update(frame_idx):
        x = secuencia[frame_idx, :, 0]
        y = secuencia[frame_idx, :, 1]
        puntos.set_data(x, y)
        if conexiones:
            for i, (p1, p2) in enumerate(conexiones):
                lineas[i].set_data([x[p1], x[p2]], [y[p1], y[p2]])
        return [puntos] + lineas

    ani = animation.FuncAnimation(fig, update, frames=F, init_func=init, blit=True, interval=1000/fps)
    
    print(f"Guardando video en: {archivo_salida}...")
    ani.save(archivo_salida, writer='ffmpeg' if archivo_salida.endswith('.mp4') else 'pillow', fps=fps)
    plt.close()

def load_bbox_by_id(file_path: str, target_person: int) -> np.ndarray:
    with h5py.File(file_path, 'r') as f:
        if 'frames' not in f:
            raise KeyError("El archivo no contiene el grupo 'frames'.")
            
        frames_group = f['frames']
        frame_names = sorted(frames_group.keys())
        T = len(frame_names)
        
        if T == 0:
            return np.zeros((0, 0)) # T, C
            
        C = 0
        for frame_name in frame_names:
            for person_name in frames_group[frame_name].keys():
                if 'bbox' in frames_group[frame_name][person_name]:
                    shape = frames_group[frame_name][person_name]['bbox'].shape
                    # Extraemos únicamente la primera dimensión ya que bbox es 1D (ej. 4)
                    C = shape[0] if len(shape) > 0 else 0
                    break
            if C > 0:
                break
                
        if C == 0:
            return np.zeros((T, 0)) 
            
        # Tensor 2D: (Frames, Coordenadas del bbox)
        tensor = np.full((T, C), np.nan)
        
        target_person_name = f"person_{target_person}" 
        for t_idx, frame_name in enumerate(frame_names):
            frame_data = frames_group[frame_name]
            
            if target_person_name in frame_data and 'bbox' in frame_data[target_person_name]:
                # Asignamos directamente los valores 1D al frame correspondiente
                tensor[t_idx, :] = np.array(frame_data[target_person_name]['bbox'])

    return tensor