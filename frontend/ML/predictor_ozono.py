# predictor_ozono.py

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor  # Aunque no se usa directamente aquí, estaba en tu script original
from sklearn.ensemble import RandomForestRegressor # Ídem
from sklearn.preprocessing import StandardScaler # Ídem
import os


# --- Carga y predice ---
def predecir_ozono(entrada_dict, modelo='xgboost'):
    BASE_DIR = os.getcwd() # Considera si esta es la base correcta o si necesitas una ruta más específica.

    # --- CORRECCIÓN DE RUTA (de la pregunta anterior) ---
    # Determina la ruta de la carpeta ML de forma más robusta,
    # por ejemplo, si está en el mismo directorio que este script:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ml_folder_path = os.path.join(script_dir) # Asume que ML es una subcarpeta donde está este script

    # O si sabes que 'ML' está siempre en 'frontend/ML' y tu CWD es la raíz del proyecto:
    # ml_folder_path = os.path.join(BASE_DIR, "frontend", "ML")

    modelo_file = os.path.join(ml_folder_path, f"{modelo.lower()}_modelo.pkl")
    escalador_file = os.path.join(ml_folder_path, "escalador.pkl") # Asume que escalador.pkl está en la misma carpeta ML

    print(f"Ruta modelo: {modelo_file}")
    print(f"Ruta escalador: {escalador_file}")

    try:
        with open(modelo_file, 'rb') as f:
            modelo_cargado = pickle.load(f)
        with open(escalador_file, 'rb') as f:
            escalador = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error al cargar archivos: {e}")
        raise Exception(f"❌ Modelos o escalador no encontrados en '{ml_folder_path}'. Verifica las rutas y ejecuta el entrenamiento primero.")

    # --- MANEJO DE COLUMNAS FALTANTES Y ORDENACIÓN ---

    # 1. Obtener la lista de columnas esperadas por el escalador/modelo
    try:
        expected_columns = list(escalador.feature_names_in_)
        print(f"Columnas esperadas por el escalador: {expected_columns}")
    except AttributeError:
        # --- ¡IMPORTANTE! FALLBACK SI feature_names_in_ NO ESTÁ DISPONIBLE ---
        # Si tu escalador no tiene 'feature_names_in_' (ej. versión antigua de scikit-learn
        # o no se entrenó con un DataFrame de Pandas), DEBES definir 'expected_columns' manualmente.
        # Esta lista DEBE contener TODAS las características que tu modelo usó para entrenar,
        # en el ORDEN EXACTO. Usa tu conocimiento del CSV original.
        print("🚨 ADVERTENCIA: 'escalador.feature_names_in_' no está disponible.")
        print("   Debes definir 'expected_columns' manualmente con la lista COMPLETA y ORDENADA.")
        print("   Revisa las columnas de tu CSV original y las que usaste para entrenar.")
        print(f"   Tu CSV original contenía: objectid;nombre;direccion;tipozona;parametros;mediciones;so2;no2;o3;co;pm10;pm25;tipoemisio;fecha_carg;calidad_am;fiwareid;longitude;latitude;geometry_type;geo_point_lon;geo_point_lat;timestamp_captura;Identificador")
        # EJEMPLO (DEBES AJUSTARLO A TU CASO REAL):
        # expected_columns = ['so2', 'no2', 'o3', 'co', 'pm10', 'pm25', 'longitude', 'latitude', 'geo_point_lon', 'geo_point_lat', 'otra_columna_num', 'tipozona_codificada_1', ...]
        # Si no defines esto correctamente aquí (cuando feature_names_in_ falla), el código fallará o dará predicciones incorrectas.
        # Por seguridad, lanzamos un error si no está disponible y no se ha definido manualmente:
        raise Exception(
            "El escalador no tiene 'feature_names_in_'. Debes definir 'expected_columns' "
            "manualmente en el código con la lista completa y ordenada de columnas usadas para el entrenamiento."
        )

    # 2. Convertir el diccionario de entrada a un DataFrame de Pandas
    df_current_input = pd.DataFrame([entrada_dict])
    print(f"Entrada recibida (DataFrame inicial): \n{df_current_input}")

    # 3. Asegurar que el DataFrame tenga todas las 'expected_columns' en el orden correcto.
    #    - Las columnas en 'expected_columns' que no estén en 'df_current_input' se añadirán con valor np.nan.
    #    - Las columnas en 'df_current_input' que no estén en 'expected_columns' se eliminarán.
    #    - Las columnas resultantes estarán en el orden de 'expected_columns'.
    df_input_for_scaler = df_current_input.reindex(columns=expected_columns)

    # Opcional: Si específicamente necesitas rellenar con 0 en lugar de np.nan (generalmente np.nan es mejor para datos faltantes)
    # df_input_for_scaler = df_input_for_scaler.fillna(0)

    print(f"DataFrame preparado para el escalador (con NaN para faltantes y ordenado): \n{df_input_for_scaler}")
    if df_input_for_scaler.isnull().any().any():
        print("Advertencia: El DataFrame preparado contiene valores NaN. Asegúrate de que tu modelo puede manejarlos o considera la imputación.")

    # 4. Escalar los datos
    try:
        df_input_scaled = escalador.transform(df_input_for_scaler)
    except ValueError as ve:
        print(f"Error durante la transformación del escalador: {ve}")
        print("Esto puede ocurrir si las columnas no coinciden exactamente (nombres, orden, cantidad) con las que se usó para entrenar el escalador, o si hay tipos de datos inesperados.")
        raise ve
        
    # 6. Predecir
    prediction = modelo_cargado.predict(df_input_scaled)
    resultado_prediccion = prediction[0] # Esto es probablemente un np.float32 o np.float64

    # ---- LA CORRECCIÓN ES ESTA LÍNEA ----
    # Convertir el resultado a un float estándar de Python antes de retornarlo
    return float(resultado_prediccion)

# Ejemplo de uso (para probarlo directamente si quieres):
# if __name__ == '__main__':
#     # Simula que los archivos .pkl existen en una carpeta ML en el mismo directorio que este script
#     # Necesitarías crear modelos y escaladores dummy para que esto funcione sin error de FileNotFoundError
#     # os.makedirs("ML", exist_ok=True)
#     # dummy_scaler = StandardScaler()
#     # dummy_model = XGBRegressor()
#     # dummy_data = pd.DataFrame(np.random.rand(10, 5), columns=['feat1', 'no2', 'geo_point_lat', 'feat4', 'feat5'])
#     # dummy_scaler.fit(dummy_data)
#     # dummy_model.fit(dummy_data.iloc[:, :-1], dummy_data.iloc[:, -1]) # model trained on 4 features
#     # with open("ML/escalador.pkl", "wb") as f: pickle.dump(dummy_scaler, f)
#     # with open("ML/xgboost_modelo.pkl", "wb") as f: pickle.dump(dummy_model, f)
#
#     # Supongamos que tu modelo ESPERA estas columnas: ['pm10', 'no2', 'geo_point_lat', 'longitude', 'latitude']
#     # Y tu escalador tiene `feature_names_in_` = ['pm10', 'no2', 'geo_point_lat', 'longitude', 'latitude']
#     entrada_parcial = {
#         'pm10': 50,
#         # 'no2': 25, # Falta no2
#         'geo_point_lat': 40.41,
#         'longitude': -3.70
#         # Falta latitude
#     }
#     try:
#         prediccion = predecir_ozono(entrada_parcial)
#         print(f"Predicción de ozono: {prediccion}")
#     except Exception as e:
#         print(f"Error en la prueba: {e}")