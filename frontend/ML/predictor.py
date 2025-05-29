import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import timedelta

# --- Configuración Global (DEBE coincidir con el entrenamiento) ---
N_LAGS = 3
WINDOW_SIZE = 24
HORAS_POR_DIA_PREDICCION = 24

# Variable global para los datos
df_global = None

# Define the directory where ML related files (models, scalers, data CSV) are stored.
# When app.py (in project_root) runs, os.getcwd() is project_root.
# So, 'ML' refers to project_root/ML/
ML_FILES_DIR = 'ML' # This assumes aire_new.csv, models, and scalers are in a dir named 'ML' relative to CWD.
                    # If predictor.py is in ML/, and CWD is ML/, then this should be '.'
                    # However, given Flask structure, CWD is project_root.

# A more robust way to define base directory for ML files IF predictor.py is ALWAYS in the ML folder:
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of the current script (project_root/ML)
# ML_FILES_DIR = SCRIPT_DIR # Models, scalers, and aire_new.csv are in the same dir as predictor.py
# For Flask, it's often simpler to make paths relative to CWD (project_root) or pass explicit paths.
# We'll stick to ML_FILES_DIR = 'ML' assuming CWD=project_root for Flask.


def cargar_datos_historicos():
    """Carga los datos históricos una sola vez"""
    global df_global
    
    if df_global is not None:
        return df_global
    
    try:
        # Path to CSV, assuming it's inside the ML_FILES_DIR
        # If aire_new.csv is directly in the 'ML' folder:
        csv_path = os.path.join(ML_FILES_DIR, 'aire_new.csv')

        if not os.path.exists(csv_path):
            # Fallback: if predictor.py is in ML/, and aire_new.csv is also in ML/
            # this attempts project_root/ML/aire_new.csv if CWD=project_root/ML
            # This part might need adjustment based on final CWD understanding for the script
            alt_csv_path_script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aire_new.csv')
            if os.path.exists(alt_csv_path_script_dir):
                csv_path = alt_csv_path_script_dir
            else:
                # Original logic if the above is too complex or CWD is consistently project_root
                # Attempt 1: ./ML/aire_new.csv (project_root/ML/aire_new.csv if CWD=project_root)
                csv_path_check1 = './ML/aire_new.csv'
                # Attempt 2: ../aire_new.csv (project_root/aire_new.csv if CWD=project_root/some_subdir)
                csv_path_check2 = '../aire_new.csv'
                # Attempt 3: aire_new.csv (project_root/ML/aire_new.csv if CWD=project_root/ML)
                csv_path_check3 = 'aire_new.csv'


                if os.path.exists(csv_path_check1):
                    csv_path = csv_path_check1
                elif os.path.exists(alt_csv_path_script_dir): # If script is in ML and CSV is there too
                    csv_path = alt_csv_path_script_dir
                elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'aire_new.csv')): # if script in ML, csv in root
                     csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'aire_new.csv')
                else: # Defaulting to the most likely correct path for Flask project structure
                    csv_path = os.path.join('ML', 'aire_new.csv') # project_root/ML/aire_new.csv
                    if not os.path.exists(csv_path):
                        print(f"❌ Error: No se encontró 'aire_new.csv' en la ubicación esperada: {csv_path}")
                        return None
        
        print(f"Attempting to load historical data from: {csv_path}")
        df_global = pd.read_csv(csv_path, sep=',', parse_dates=['timestamp_captura'], dayfirst=True)
        df_global = df_global.rename(columns={'timestamp_captura': 'Fecha', 'objectid': 'objectId'})
        df_global['Fecha'] = pd.to_datetime(df_global['Fecha'], errors='coerce', dayfirst=True)
        
        numeric_cols_to_check = ['so2', 'no2', 'o3', 'co', 'pm10', 'pm25']
        for col in numeric_cols_to_check:
            if col in df_global.columns:
                df_global[col] = pd.to_numeric(df_global[col], errors='coerce')
        
        print(f"📊 Datos históricos cargados: {len(df_global)} registros, {df_global['objectId'].nunique()} zonas únicas")
        return df_global
        
    except Exception as e:
        print(f"❌ Error cargando datos históricos desde '{csv_path}': {e}")
        return None

def obtener_object_ids_disponibles():
    """Obtiene la lista de object_ids que tienen modelos entrenados.
       Corrected to look in ML_FILES_DIR.
    """
    object_ids = []
    
    # Ensure ML_FILES_DIR exists and is a directory
    actual_ml_files_dir = ML_FILES_DIR
    # If this script (predictor.py) is inside the 'ML' folder, this path logic simplifies things
    # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # This would be project_root/ML
    # actual_ml_files_dir = SCRIPT_DIR

    if not os.path.isdir(actual_ml_files_dir):
        print(f"⚠️  Advertencia: El directorio de archivos ML '{actual_ml_files_dir}' no se encontró o no es un directorio.")
        return []

    for filename in os.listdir(actual_ml_files_dir):
        if filename.startswith("modelo_") and filename.endswith(".pkl"):
            try:
                object_id = int(filename.split('_')[1])
                escalador_filename = f"escalador_{object_id}.pkl"
                # Check for scaler in the same directory
                if os.path.exists(os.path.join(actual_ml_files_dir, escalador_filename)):
                    object_ids.append(object_id)
            except (IndexError, ValueError):
                continue
    
    return sorted(object_ids)

def generar_predicciones_3_dias(object_id, fecha_inicio_str, historical_df=None):
    """
    Genera predicciones de O3 para los próximos 3 días (72 horas) para un objectId.
    """
    if historical_df is None:
        historical_df = cargar_datos_historicos() # This will use df_global if already loaded
    
    if historical_df is None:
        print("❌ Error: Datos históricos no disponibles.")
        return None

    try:
        fecha_inicio_prediccion = pd.to_datetime(fecha_inicio_str)
    except Exception as e:
        print(f"❌ Error parseando fecha de inicio '{fecha_inicio_str}': {e}")
        return None

    # --- 1. Cargar Modelo y Escalador Específico del objectId ---
    # Paths should be relative to where the files are, using ML_FILES_DIR
    # If this script (predictor.py) is inside the 'ML' folder, this path logic simplifies things:
    # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # This would be project_root/ML
    # files_directory = SCRIPT_DIR
    files_directory = ML_FILES_DIR # Assumes CWD is project_root for Flask app

    model_prefix = f"modelo_{object_id}_"
    model_filenames = []
    try:
        if not os.path.isdir(files_directory):
            print(f"❌ Error: Directorio de modelos '{files_directory}' no encontrado desde CWD '{os.getcwd()}'.")
            return None
        model_filenames = [f for f in os.listdir(files_directory) if f.startswith(model_prefix) and f.endswith(".pkl")]
    except FileNotFoundError:
        print(f"❌ Error: No se pudo acceder al directorio de modelos '{files_directory}'.")
        return None
        
    if not model_filenames:
        print(f"❌ Error: No se encontró modelo para objectId {object_id} en '{files_directory}'. (Buscando prefijo: '{model_prefix}')")
        return None
    
    model_file_name = model_filenames[0] # Take the first one found
    model_file_path = os.path.join(files_directory, model_file_name)

    scaler_file_name = f"escalador_{object_id}.pkl"
    scaler_file_path = os.path.join(files_directory, scaler_file_name)
    
    if not os.path.exists(scaler_file_path):
        print(f"❌ Error: No se encontró escalador '{scaler_file_path}'.")
        return None

    try:
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_file_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Modelo '{model_file_path}' y escalador '{scaler_file_path}' cargados para objectId {object_id}.")
    except Exception as e:
        print(f"❌ Error cargando archivos pickle ({model_file_path}, {scaler_file_path}): {e}")
        return None

    # --- 2. Preparar Datos Históricos Iniciales ---
    # (Rest of the function remains the same)
    df_zona_hist = historical_df[historical_df['objectId'] == object_id].copy()
    
    if df_zona_hist.empty:
        print(f"❌ Error: No hay datos históricos en df_global para objectId {object_id}")
        return None
    
    df_zona_hist.sort_values(by='Fecha', inplace=True)
    
    datos_para_inicio = df_zona_hist[df_zona_hist['Fecha'] < fecha_inicio_prediccion].tail(WINDOW_SIZE + N_LAGS + 5)

    if len(datos_para_inicio) < max(N_LAGS, WINDOW_SIZE):
        print(f"❌ Error: Datos históricos insuficientes ({len(datos_para_inicio)}) para objectId {object_id} antes de {fecha_inicio_str}. Se necesitan al menos {max(N_LAGS, WINDOW_SIZE)}.")
        return None

    predicciones_list = []
    current_o3_series = datos_para_inicio['o3'].dropna().copy()
    columnas_features_base = [col for col in datos_para_inicio.select_dtypes(include=np.number).columns 
                             if col not in ['Fecha', 'o3', 'objectId']]
    latest_exog_features = datos_para_inicio[columnas_features_base].iloc[-1].copy()
    
    for col in columnas_features_base:
        if pd.isna(latest_exog_features[col]):
            mean_val = datos_para_inicio[col].mean()
            latest_exog_features[col] = mean_val if pd.notna(mean_val) else 0

    current_timestamp = fecha_inicio_prediccion
    
    for hora in range(3 * HORAS_POR_DIA_PREDICCION):
        try:
            input_features = {}
            input_features['hora'] = current_timestamp.hour
            input_features['dia_semana'] = current_timestamp.dayofweek
            input_features['mes'] = current_timestamp.month
            input_features['dia_del_anio'] = current_timestamp.dayofyear

            if len(current_o3_series) < N_LAGS:
                mean_o3_hist = current_o3_series.mean() if not current_o3_series.empty else 0
                padded_o3 = np.pad(current_o3_series.values, (N_LAGS - len(current_o3_series), 0), 
                                 'constant', constant_values=mean_o3_hist)
                for i in range(1, N_LAGS + 1):
                    input_features[f'o3_lag_{i}'] = padded_o3[-i]
            else:
                for i in range(1, N_LAGS + 1):
                    input_features[f'o3_lag_{i}'] = current_o3_series.iloc[-i]
            
            if len(current_o3_series) >= 1:
                rolling_data = current_o3_series.iloc[-min(len(current_o3_series), WINDOW_SIZE):]
                input_features['o3_rolling_mean'] = rolling_data.mean()
                input_features['o3_rolling_std'] = rolling_data.std()
            else:
                input_features['o3_rolling_mean'] = 0
                input_features['o3_rolling_std'] = 0

            if pd.isna(input_features['o3_rolling_std']): input_features['o3_rolling_std'] = 0
            if pd.isna(input_features['o3_rolling_mean']): input_features['o3_rolling_mean'] = 0

            for col_base in columnas_features_base:
                input_features[col_base] = latest_exog_features[col_base]

            try:
                model_columns = scaler.feature_names_in_
            except AttributeError:
                print("❌ Error: Escalador sin 'feature_names_in_'. Asegúrate de que el escalador se guardó con esta información o define las columnas explícitamente.")
                # Potentially, you might need to fall back to a predefined list of columns if this happens
                # and the scaler is from an older library version.
                return None 

            input_df_single_row = pd.DataFrame([input_features])
            input_df_single_row = input_df_single_row.reindex(columns=model_columns, fill_value=0)

            input_scaled = scaler.transform(input_df_single_row)
            predicted_o3 = model.predict(input_scaled)[0]
            
            if np.isnan(predicted_o3) or np.isinf(predicted_o3):
                predicted_o3 = current_o3_series.mean() if not current_o3_series.empty else 0
            
            predicciones_list.append({
                "fecha": current_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "o3_predicho": round(float(predicted_o3), 2)
            })
            
            new_o3_value_series = pd.Series([predicted_o3])
            current_o3_series = pd.concat([current_o3_series, new_o3_value_series], ignore_index=True)
            
            if len(current_o3_series) > (WINDOW_SIZE + N_LAGS + 5):
                current_o3_series = current_o3_series.iloc[-(WINDOW_SIZE + N_LAGS + 5):]

            current_timestamp += timedelta(hours=1)
            
        except Exception as e:
            print(f"❌ Error en predicción hora {hora} para {object_id} @ {current_timestamp}: {e}")
            # Log traceback for detailed debugging:
            import traceback
            traceback.print_exc()
            
            mean_pred = np.mean([p['o3_predicho'] for p in predicciones_list]) if predicciones_list else 0
            predicciones_list.append({
                "fecha": current_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "o3_predicho": round(mean_pred, 2) # Fallback prediction
            })
            current_timestamp += timedelta(hours=1) # Ensure timestamp advances
            # Consider whether to continue or halt all predictions if one step fails critically.
            # For now, it continues with a mean.
            continue # Continue to the next hour
            
    return json.dumps(predicciones_list, indent=2)

# Cargar datos al importar el módulo (this will run when app.py imports predictor)
# CWD is project_root at this point.
df_global = cargar_datos_historicos()