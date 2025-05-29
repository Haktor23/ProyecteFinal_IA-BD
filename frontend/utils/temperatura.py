import csv
import os
from datetime import datetime

# --- Configuración de Rutas ---
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()
    print(f"Advertencia: __file__ no está definido. Usando BASE_DIR como: {BASE_DIR}")

RUTA_CSV_CALIDAD = os.path.join(BASE_DIR, 'data', 'CSV', 'HistoricoTemperatura.csv')

# --- Cache Global para Datos del CSV ---
CACHED_DATA = None
LAST_MODIFIED_TIME = None

# --- Funciones Auxiliares ---
def _parse_timestamp(timestamp_str):
    """
    Intenta parsear un string de timestamp en varios formatos comunes.
    Retorna un objeto datetime si tiene éxito, None en caso contrario.
    """
    if not timestamp_str or not isinstance(timestamp_str, str):
        return None
    formats_to_try = [
        '%d/%m/%Y %H:%M:%S',  # Ej: 23/05/2025 10:30:00
        '%Y-%m-%d %H:%M:%S',  # Ej: 2025-05-23 10:30:00
        '%d/%m/%Y %H:%M',     # Ej: 23/05/2025 10:30
        '%Y-%m-%d %H:%M',     # Ej: 2025-05-23 10:30
        '%Y/%m/%d %H:%M:%S',  # Formato con barras inclinadas en año primero
    ]
    for fmt in formats_to_try:
        try:
            return datetime.strptime(timestamp_str.strip(), fmt)
        except ValueError:
            continue
    return None

def _cargar_datos_desde_csv(ruta_archivo, force_reload=False):
    """
    Carga datos desde un archivo CSV, procesándolos y aplicando un cacheo simple
    basado en la fecha de modificación del archivo.
    """
    global CACHED_DATA, LAST_MODIFIED_TIME

    if not os.path.exists(ruta_archivo):
        print(f"Error: El archivo CSV '{ruta_archivo}' no fue encontrado.")
        CACHED_DATA = []
        LAST_MODIFIED_TIME = None
        return []

    current_mtime = os.path.getmtime(ruta_archivo)

    if CACHED_DATA is not None and not force_reload and LAST_MODIFIED_TIME == current_mtime:
        print("INFO: Usando datos cacheados del CSV (archivo no modificado).")
        return CACHED_DATA

    print(f"INFO: Cargando CSV desde la ruta: {ruta_archivo}")

    datos_csv = []
    try:
        with open(ruta_archivo, mode='r', encoding='utf-8-sig', newline='') as file:
            csv_reader = csv.DictReader(file, delimiter=';')
            if not csv_reader.fieldnames:
                print(f"Error: El archivo CSV '{ruta_archivo}' está vacío o no tiene encabezados.")
                CACHED_DATA = []
                LAST_MODIFIED_TIME = current_mtime
                return []

            for row_idx, row in enumerate(csv_reader):
                processed_row = {'_id': row_idx} 
                
                original_ts = row.get('timestamp_captura', '')
                dt_object = _parse_timestamp(original_ts)
                
                processed_row['timestamp_captura_original'] = original_ts
                if dt_object:
                    processed_row['dt_object'] = dt_object
                    processed_row['date_str'] = dt_object.strftime('%Y-%m-%d')
                    processed_row['time_str'] = dt_object.strftime('%H:%M:%S')
                    processed_row['hour_int'] = dt_object.hour
                else:
                    processed_row['dt_object'] = None
                    processed_row['date_str'] = original_ts.split(' ')[0] if original_ts and ' ' in original_ts else original_ts or 'N/A'
                    processed_row['time_str'] = None
                    processed_row['hour_int'] = None
                
                for key, value in row.items():
                    if key == 'timestamp_captura': 
                        continue

                    # Columnas numéricas de contaminantes
                    if key in ['so2', 'no2', 'o3', 'co', 'pm10', 'pm25']: 
                        try:
                            if isinstance(value, str) and value.strip() != '':
                                processed_row[key] = float(value.strip().replace(',', '.'))
                            elif isinstance(value, (int, float)):
                                processed_row[key] = float(value)
                            else:
                                processed_row[key] = None 
                        except (ValueError, TypeError):
                            processed_row[key] = None 
                    else:
                        processed_row[key] = value.strip() if isinstance(value, str) else value
                
                datos_csv.append(processed_row)
        
        print(f"Datos CSV cargados exitosamente. Filas: {len(datos_csv)}")
        CACHED_DATA = datos_csv 
        LAST_MODIFIED_TIME = current_mtime
        return datos_csv
    except Exception as e:
        print(f"Error al leer el archivo CSV '{ruta_archivo}': {e}")
        CACHED_DATA = []
        LAST_MODIFIED_TIME = current_mtime
        return []

# --- Mapeo de Columnas para DataTables ---
COLUMN_MAP = {
    0: 'nombre', 1: 'direccion', 2: 'tipozona',
    3: 'so2', 4: 'no2', 5: 'o3', 6: 'co',
    7: 'pm10', 8: 'pm25', 9: 'calidad_am',
    10: 'timestamp_captura_original'
}
SEARCHABLE_COLUMNS = ['nombre', 'direccion', 'tipozona', 'calidad_am', 'timestamp_captura_original']

# --- Lógica para DataTables (Server-Side Processing) ---
def procesar_para_datatable(request_args):
    """
    Procesa los datos del CSV para servirlos a DataTables en modo server-side.
    """
    try:
        print(f"DEBUG: Parámetros recibidos: {dict(request_args)}")
        
        all_data = _cargar_datos_desde_csv(RUTA_CSV_CALIDAD)
        
        if not all_data:
            print("DEBUG: No hay datos disponibles")
            return {
                'draw': int(request_args.get('draw', 0)),
                'recordsTotal': 0,
                'recordsFiltered': 0,
                'data': []
            }

        draw = int(request_args.get('draw', 0))
        start = int(request_args.get('start', 0))
        length = int(request_args.get('length', 20)) 
        
        search_value = request_args.get('search[value]', '').strip().lower()
        
        print(f"DEBUG: draw={draw}, start={start}, length={length}, search='{search_value}'")
        
        # 1. Filtrado (Búsqueda)
        filtered_data = all_data
        if search_value:
            temp_data = []
            for row in all_data:
                found = False
                for col_key in SEARCHABLE_COLUMNS:
                    cell_value = row.get(col_key)
                    if cell_value is not None and search_value in str(cell_value).lower():
                        temp_data.append(row)
                        found = True
                        break 
            filtered_data = temp_data
        
        # 2. Ordenamiento
        order_column_index_str = request_args.get('order[0][column]')
        order_dir = request_args.get('order[0][dir]', 'asc').lower()

        if order_column_index_str is not None and order_column_index_str.isdigit():
            order_column_index = int(order_column_index_str)
            if order_column_index in COLUMN_MAP:
                sort_key_name = COLUMN_MAP[order_column_index]
                
                try:
                    if sort_key_name == 'timestamp_captura_original':
                        filtered_data.sort(
                            key=lambda item: item.get('dt_object') or datetime.min,
                            reverse=(order_dir == 'desc')
                        )
                    else:
                        def sort_key_func(item):
                            value = item.get(sort_key_name)
                            if value is None:
                                return float('inf') if order_dir == 'asc' else float('-inf')
                            if isinstance(value, (int, float)):
                                return value
                            return str(value).lower()
                        
                        filtered_data.sort(key=sort_key_func, reverse=(order_dir == 'desc'))
                except Exception as e:
                    print(f"Error durante el ordenamiento: {e}")

        records_total = len(all_data)
        records_filtered = len(filtered_data)
        
        # 3. Paginación
        paginated_data = filtered_data[start : start + length]
        
        # 4. Formatear datos para la respuesta de DataTables
        data_output = []
        for row in paginated_data:
            data_output.append([
                row.get('nombre', "N/D"),
                row.get('direccion', "N/D"),
                row.get('tipozona', "N/D"),
                f"{row.get('so2'):.2f}" if row.get('so2') is not None else "N/A",
                f"{row.get('no2'):.2f}" if row.get('no2') is not None else "N/A",
                f"{row.get('o3'):.2f}" if row.get('o3') is not None else "N/A",
                f"{row.get('co'):.2f}" if row.get('co') is not None else "N/A",
                f"{row.get('pm10'):.2f}" if row.get('pm10') is not None else "N/A",
                f"{row.get('pm25'):.2f}" if row.get('pm25') is not None else "N/A",
                row.get('calidad_am', "N/D"),
                row.get('timestamp_captura_original', "N/D")
            ])
        
        result = {
            'draw': draw,
            'recordsTotal': records_total,
            'recordsFiltered': records_filtered,
            'data': data_output
        }
        
        print(f"DEBUG: Respuesta - Total: {records_total}, Filtrados: {records_filtered}, Datos: {len(data_output)}")
        return result
        
    except Exception as e:
        print(f"ERROR en procesar_para_datatable: {e}")
        import traceback
        traceback.print_exc()
        return {
            'draw': int(request_args.get('draw', 0)),
            'recordsTotal': 0,
            'recordsFiltered': 0,
            'data': [],
            'error': str(e)
        }

# --- Función para el Contexto Inicial ---
def obtener_contexto():
    """
    Prepara datos para la carga inicial de la página de calidad.html
    """
    try:
        all_data = _cargar_datos_desde_csv(RUTA_CSV_CALIDAD, force_reload=False)
        
        if not all_data:
            return {
                'titulo': 'Calidad del Aire - Panel Avanzado',
                'descripcion': 'No se pudieron cargar los datos del CSV.',
                'datos_tarjetas': {'numero_estaciones': 0, 'promedio_pm25': 'N/A'},
                'raw_air_quality_data_for_graphs': [],
                'select_dates': [],
                'select_stations': [],
                'select_pollutants': [
                    {'key': 'so2', 'name': 'SO₂'}, {'key': 'no2', 'name': 'NO₂'},
                    {'key': 'o3', 'name': 'O₃'}, {'key': 'co', 'name': 'CO'},
                    {'key': 'pm10', 'name': 'PM₁₀'}, {'key': 'pm25', 'name': 'PM₂.₅'}
                ],
                'estado_sistema': 'Inactivo (Error al cargar datos)'
            }
        
        # Extraer datos únicos para los selectores
        all_dates = sorted(list(set(row['date_str'] for row in all_data if row.get('date_str') and row['date_str'] != 'N/A')))
        all_stations = sorted(list(set(row['nombre'] for row in all_data if row.get('nombre'))))
        
        pollutants_metadata = [
            {'key': 'so2', 'name': 'SO₂'}, {'key': 'no2', 'name': 'NO₂'},
            {'key': 'o3', 'name': 'O₃'}, {'key': 'co', 'name': 'CO'},
            {'key': 'pm10', 'name': 'PM₁₀'}, {'key': 'pm25', 'name': 'PM₂.₅'}
        ]
        
        # Calcular estadísticas
        num_estaciones = len(all_stations)
        pm25_valid_values = [row['pm25'] for row in all_data if row.get('pm25') is not None]
        promedio_pm25 = sum(pm25_valid_values) / len(pm25_valid_values) if pm25_valid_values else None

        datos_para_tarjetas = { 
            'numero_estaciones': num_estaciones,
            'promedio_pm25': f"{promedio_pm25:.2f}" if promedio_pm25 is not None else 'N/A',
        }
        
        print(f"DEBUG: Contexto preparado - Fechas: {len(all_dates)}, Estaciones: {len(all_stations)}, Datos: {len(all_data)}")
        
        return {
            'titulo': 'Calidad del Aire - Panel Avanzado',
            'descripcion': 'Visualización interactiva de datos históricos y actuales de las estaciones de calidad del aire.',
            'datos_tarjetas': datos_para_tarjetas,
            'raw_air_quality_data_for_graphs': all_data,
            'select_dates': all_dates,
            'select_stations': all_stations,
            'select_pollutants': pollutants_metadata,
            'estado_sistema': 'Activo'
        }
        
    except Exception as e:
        print(f"ERROR en obtener_contexto: {e}")
        import traceback
        traceback.print_exc()
        return {
            'titulo': 'Calidad del Aire - Panel Avanzado',
            'descripcion': f'Error al cargar contexto: {str(e)}',
            'datos_tarjetas': {'numero_estaciones': 0, 'promedio_pm25': 'N/A'},
            'raw_air_quality_data_for_graphs': [],
            'select_dates': [],
            'select_stations': [],
            'select_pollutants': [],
            'estado_sistema': 'Error'
        }

# Función genérica si es necesaria
def procesar_datos(data):
    """Función genérica para procesar otros tipos de datos de calidad"""
    return {"message": "Datos de calidad procesados", "received_data": data}