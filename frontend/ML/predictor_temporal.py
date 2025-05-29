import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import timedelta

# --- Configuración Global (DEBE coincidir con el entrenamiento) ---
N_LAGS = 3  # Asegúrate que este valor es el mismo que en tu entrenamiento
WINDOW_SIZE = 24 # Asegúrate que este valor es el mismo
HORAS_POR_DIA_PREDICCION = 24 # Asumimos predicciones horarias

# Carga de datos históricos (necesario para iniciar la secuencia de predicción)
try:
    df_global = pd.read_csv('aire_new.csv', sep=',', parse_dates=['timestamp_captura'], dayfirst=True)
    df_global = df_global.rename(columns={'timestamp_captura': 'Fecha', 'objectid': 'objectId'})
    df_global['Fecha'] = pd.to_datetime(df_global['Fecha'], errors='coerce', dayfirst=True)
    numeric_cols_to_check = ['so2', 'no2', 'o3', 'co', 'pm10', 'pm25'] #Añade 'no2' si es una feature
    for col in numeric_cols_to_check:
        if col in df_global.columns:
            df_global[col] = pd.to_numeric(df_global[col], errors='coerce')
    print("📊 Datos históricos cargados para predicciones multi-día.")
except FileNotFoundError:
    print("❌ Error: No se encontró 'aire_new.csv'. Necesitas los datos históricos.")
    df_global = None

def generar_predicciones_3_dias(object_id, fecha_inicio_str, historical_df):
    """
    Genera predicciones de O3 para los próximos 3 días (72 horas) para un objectId.

    Args:
        object_id (int): El ID de la zona.
        fecha_inicio_str (str): La fecha/hora de inicio para la primera predicción
                                (ej. '2025-04-25 00:00:00'). Las predicciones comenzarán
                                DESPUÉS de esta hora.
        historical_df (pd.DataFrame): DataFrame con todos los datos históricos.

    Returns:
        str: Un JSON con las predicciones horarias, o None si hay error.
    """
    if historical_df is None:
        print("❌ Error: Datos históricos no disponibles.")
        return None

    fecha_inicio_prediccion = pd.to_datetime(fecha_inicio_str)

    # --- 1. Cargar Modelo y Escalador Específico del objectId ---
    model_file = None
    scaler_file = f"escalador_{object_id}.pkl"
    # Asumimos que el nombre del modelo incluye el object_id y el nombre del algoritmo
    # Ejemplo: modelo_12_xgboost.pkl (necesitarás buscar el nombre exacto)
    
    # Busca el modelo (puedes tener una lógica más robusta para obtener el "mejor" modelo)
    archivos_modelo_oid = [f for f in os.listdir('.') if f.startswith(f"modelo_{object_id}_") and f.endswith(".pkl")]
    if not archivos_modelo_oid:
        print(f"❌ Error: No se encontró ningún archivo de modelo para objectId {object_id}.")
        return None
    model_file = archivos_modelo_oid[0] # Tomamos el primero que encuentre, idealmente sabrías cuál es el mejor
    
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        print(f"❌ Error: No se encontró el archivo de modelo '{model_file}' o el escalador '{scaler_file}'.")
        return None

    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Modelo '{model_file}' y escalador '{scaler_file}' cargados para objectId {object_id}.")
    except Exception as e:
        print(f"❌ Error cargando archivos para {object_id}: {e}")
        return None

    # --- 2. Preparar Datos Históricos Iniciales ---
    df_zona_hist = historical_df[historical_df['objectId'] == object_id].copy()
    df_zona_hist.sort_values(by='Fecha', inplace=True)
    
    # Tomamos los datos históricos justo ANTES de la fecha de inicio de la predicción
    # Necesitamos suficientes datos para los lags y la ventana móvil inicial
    datos_para_inicio = df_zona_hist[df_zona_hist['Fecha'] < fecha_inicio_prediccion].tail(WINDOW_SIZE + N_LAGS + 5)

    if len(datos_para_inicio) < max(N_LAGS, WINDOW_SIZE):
        print(f"❌ Error: No hay suficientes datos históricos ({len(datos_para_inicio)}) para {object_id} antes de {fecha_inicio_str} para calcular lags/ventanas iniciales.")
        return None

    # Lista para guardar las predicciones [{fecha, o3_predicho}]
    predicciones_list = []
    
    # Copia de los datos recientes para ir actualizando con las predicciones
    # Esta serie `current_o3_series` mantendrá los últimos valores de o3 (reales y luego predichos)
    current_o3_series = datos_para_inicio['o3'].dropna().copy()
    # Datos exógenos recientes para usarlos como base para el futuro (¡GRAN SUPOSICIÓN!)
    columnas_features_base = [col for col in datos_para_inicio.select_dtypes(include=np.number).columns if col not in ['Fecha', 'o3', 'objectId']]
    latest_exog_features = datos_para_inicio[columnas_features_base].iloc[-1].copy() # Tomamos la última fila de exógenas
    
    # Si alguna exógena es NaN en la última fila, rellenar con la media de esa columna en `datos_para_inicio`
    for col in columnas_features_base:
        if pd.isna(latest_exog_features[col]):
            mean_val = datos_para_inicio[col].mean()
            latest_exog_features[col] = mean_val if pd.notna(mean_val) else 0 # Rellena con media o 0


    # --- 3. Bucle de Predicción Secuencial (hora a hora) ---
    current_timestamp = fecha_inicio_prediccion
    for _ in range(3 * HORAS_POR_DIA_PREDICCION): # 3 días * 24 horas/día
        input_features = {}
        
        # Características Temporales (para el current_timestamp)
        input_features['hora'] = current_timestamp.hour
        input_features['dia_semana'] = current_timestamp.dayofweek
        input_features['mes'] = current_timestamp.month
        input_features['dia_del_anio'] = current_timestamp.dayofyear

        # Lags (de current_o3_series, que se actualiza)
        if len(current_o3_series) < N_LAGS:
            # Rellenar si no hay suficientes datos para lags (esto podría pasar al inicio si hay muchos NaNs)
            # Podrías usar la media general de o3 de la zona, o el último valor conocido
            mean_o3_hist = current_o3_series.mean() if not current_o3_series.empty else 0
            padded_o3 = np.pad(current_o3_series.values, (N_LAGS - len(current_o3_series), 0), 'constant', constant_values=mean_o3_hist)
            for i in range(1, N_LAGS + 1):
                input_features[f'o3_lag_{i}'] = padded_o3[-i]
        else:
            for i in range(1, N_LAGS + 1):
                input_features[f'o3_lag_{i}'] = current_o3_series.iloc[-i]
        
        # Ventanas Móviles (de current_o3_series)
        # Para la ventana móvil, tomamos los datos ANTERIORES al paso actual (equivalente al shift(1) del entrenamiento)
        if len(current_o3_series) >= 1: # Necesita al menos 1 valor para el shift
            rolling_window_data = current_o3_series.iloc[-min(len(current_o3_series), WINDOW_SIZE):] # Los últimos WINDOW_SIZE puntos de la serie actual
            input_features['o3_rolling_mean'] = rolling_window_data.mean()
            input_features['o3_rolling_std'] = rolling_window_data.std()
        else: # No hay datos para rolling window
            input_features['o3_rolling_mean'] = current_o3_series.mean() if not current_o3_series.empty else 0 # O algún otro valor por defecto
            input_features['o3_rolling_std'] = 0


        # Rellenar NaN en rolling_std (si la varianza es 0 o pocos puntos)
        if pd.isna(input_features['o3_rolling_std']):
            input_features['o3_rolling_std'] = 0
        if pd.isna(input_features['o3_rolling_mean']): # Si la media también es NaN (serie vacía)
            input_features['o3_rolling_mean'] = 0


        # Variables Exógenas (usamos latest_exog_features)
        # ¡SUPOSICIÓN!: mantenemos constantes las últimas observadas o podrías aplicarles alguna tendencia/ciclo
        for col_base in columnas_features_base:
            input_features[col_base] = latest_exog_features[col_base]

        # Crear DataFrame para la predicción
        try:
            # Usar feature_names_in_ del escalador, que debería tenerlas
            model_columns = scaler.feature_names_in_
        except AttributeError:
            print("❌ Error: El escalador no tiene 'feature_names_in_'. No se puede determinar el orden de las columnas. Guarda las columnas durante el entrenamiento.")
            return None # No se puede continuar si no conocemos las columnas exactas

        input_df_single_row = pd.DataFrame([input_features])
        # Reordenar y asegurar que todas las columnas del modelo están presentes, rellenando con 0 si falta alguna (aunque no debería si usamos las exógenas)
        input_df_single_row = input_df_single_row.reindex(columns=model_columns, fill_value=0) 


        # Escalar y Predecir
        input_scaled = scaler.transform(input_df_single_row)
        predicted_o3 = model.predict(input_scaled)[0]
        
        # Guardar predicción
        predicciones_list.append({
            "fecha": current_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "o3_predicho": round(float(predicted_o3), 2) # Convertir a float nativo para JSON
        })
        
        # Actualizar current_o3_series con la nueva predicción para el siguiente paso
        # Creamos una nueva serie para evitar problemas de índice con pd.concat
        new_o3_value_series = pd.Series([predicted_o3])
        current_o3_series = pd.concat([current_o3_series, new_o3_value_series], ignore_index=True)
        # Mantenemos solo los últimos N necesarios para no crecer indefinidamente
        if len(current_o3_series) > (WINDOW_SIZE + N_LAGS + 5) :
             current_o3_series = current_o3_series.iloc[-(WINDOW_SIZE + N_LAGS + 5):]


        # Avanzar al siguiente timestamp (1 hora)
        current_timestamp += timedelta(hours=1)
        
    return json.dumps(predicciones_list, indent=4)


# --- Ejemplo de Uso ---
if df_global is not None:
    # Elige un objectId para el que tengas un modelo guardado
    # Busca un ID viable que tenga modelo guardado
    id_a_predecir_multidia = None
    for f_name in os.listdir('.'):
        if f_name.startswith("modelo_") and f_name.endswith(".pkl"):
            try:
                # Extrae el ID del nombre del archivo, por ejemplo: "modelo_12_xgboost.pkl" -> 12
                id_test = int(f_name.split('_')[1]) 
                # Comprobar si también existe el escalador
                if os.path.exists(f"escalador_{id_test}.pkl"):
                    id_a_predecir_multidia = id_test
                    print(f"💡 Usando ObjectId {id_a_predecir_multidia} para predicción multi-día (modelo encontrado: {f_name}).")
                    break
            except (IndexError, ValueError):
                # El nombre del archivo no tiene el formato esperado
                continue
    
    if id_a_predecir_multidia is not None:
        # Tomar la última fecha disponible para ese objectId como referencia
        ultima_fecha_conocida_obj = df_global[df_global['objectId'] == id_a_predecir_multidia]['Fecha'].max()
        if pd.isna(ultima_fecha_conocida_obj):
            print(f"⚠️ No se pudo determinar la última fecha conocida para objectId {id_a_predecir_multidia}. Usando fecha por defecto.")
            fecha_de_inicio_predicciones = "2025-04-25 00:00:00" # Cambia esto si es necesario
        else:
            # Iniciamos predicciones justo después de la última fecha conocida
            fecha_de_inicio_predicciones = (ultima_fecha_conocida_obj + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

        print(f"📅 Solicitando predicciones para ObjectId {id_a_predecir_multidia} a partir de {fecha_de_inicio_predicciones}")
        
        json_predicciones = generar_predicciones_3_dias(id_a_predecir_multidia, fecha_de_inicio_predicciones, df_global)
        
        if json_predicciones:
            print("\n--- JSON de Predicciones (próximos 3 días)---")
            print(json_predicciones)
            
            # Para cargarlo como gráfico, normalmente pasarías este JSON a una librería de JS (Chart.js, D3.js, etc.)
            # En Python, para una visualización rápida:
            try:
                import matplotlib.pyplot as plt
                df_plot = pd.read_json(json_predicciones)
                df_plot['fecha'] = pd.to_datetime(df_plot['fecha'])
                plt.figure(figsize=(15, 6))
                plt.plot(df_plot['fecha'], df_plot['o3_predicho'], marker='o', linestyle='-')
                plt.title(f'Predicción O3 (3 días) para ObjectId {id_a_predecir_multidia}')
                plt.xlabel('Fecha')
                plt.ylabel('O3 Predicho (µg/m³)')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("\n(Para graficar directamente en Python, instala matplotlib: pip install matplotlib)")
            except Exception as e_plot:
                print(f"\nError al intentar graficar: {e_plot}")
        else:
            print(f"\n❌ No se pudieron generar las predicciones para ObjectId {id_a_predecir_multidia}.")
    else:
        print("❌ No se encontró ningún objectId con modelo y escalador guardados para realizar la predicción multi-día.")