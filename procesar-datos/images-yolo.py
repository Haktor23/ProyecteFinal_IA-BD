import os
import subprocess
import sys
from azure.storage.blob import BlobServiceClient
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
import uuid
# import matplotlib.pyplot as plt # Comentado ya que no se usa para mostrar imágenes en este script
import csv
import re
from datetime import datetime
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
import traceback # Para trazas de error más detalladas si es necesario

# --- Configuración y Variables Globales ---
# Comprobación de PyTorch y CUDA
print(f"Versión de PyTorch: {torch.__version__}")
print(f"CUDA Disponible: {torch.cuda.is_available()}")
# import torchvision # Descomentar si la ayuda de torchvision.ops.nms es estrictamente necesaria
# help(torchvision.ops.nms) # Descomentar para verificar la firma

# Azure Blob Storage - Capturas Originales
CONNECTION_STRING_ORIG = ""
CONTAINER_NAME_ORIG = "capturas"

# Azure Blob Storage - GIO (para resultados)
CONNECTION_STRING_GIO = ""
CONTAINER_NAME_GIO = "data"
CSV_BLOB_NAME = 'camaras.csv' # CSV centralizado para esta instancia del script

CHECKPOINT_FILE = './checkpoint.txt' # Archivo para guardar el estado de imágenes procesadas
DOWNLOAD_DIR = './imagenes_anotadas/' # Directorio para imágenes anotadas localmente antes de subir
REAL_ESRGAN_SETUP_DONE_FLAG = '.real_esrgan_configurado.flag' # Archivo bandera para la configuración de Real-ESRGAN

# Modelo YOLO - Cargar una vez
YOLO_MODEL_PATH = "yolo11x.pt" # Asegúrate de que este modelo esté disponible
# MODEL = YOLO("yolo11n-seg.pt") # Modelo oficial alternativo

# Carpetas de cámaras a procesar
# 'DR_LLUCH': ["14203", "14204", "14206"],
CARPETAS = [
    'camara_10302/','camara_11502/','camara_11402/','camara_2602/','camara_10510/',
    'camara_10304/','camara_11504/','camara_11403/','camara_2604/','camara_10511/',
    'camara_3103/','camara_3105/','camara_3106/','camara_14203/','camara_14204/',
    'camara_14206/','camara_6004/','camara_6011/','camara_4302/','camara_6012/',
    'camara_11505/','camara_11404/','camara_2605/','camara_10512/','camara_4304/',
    'camara_4305/','camara_10305/'
]
#CARPETAS = ['camara_14203/','camara_14204/','camara_14206/']
CSV_HEADERS = ['timestamp','camera_id','car','truck','motorcycle','bus','image_name','processed_at']
#CSV_HEADERS = ['timestamp', 'id_camara',  'coche', 'camion', 'moto', 'autobus','nombre_imagen_original','confianza_media','fecha_procesado']

# Bloqueos para concurrencia (Threading Locks)
csv_lock = threading.Lock()
checkpoint_lock = threading.Lock()

# Conjunto global para imágenes procesadas, cargado al inicio
processed_images_set = set()

# --- Configuración de Real-ESRGAN ---
def run_shell_command(command_str):
    """Ejecuta un comando de shell. Muestra salida detallada solo si hay errores."""
    nombre_comando = command_str.split(' ')[0]
    print(f"[INFO] Ejecutando comando externo: {nombre_comando} ...")
    
    process = subprocess.Popen(
        command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        universal_newlines=True, encoding='utf-8'
    )
    
    output_lines = []
    # Captura la salida línea por línea
    if process.stdout:
        for line in process.stdout:
            output_lines.append(line)
    
    process.wait() # Espera a que el comando termine
    
    if process.returncode != 0:
        print(f"[ERROR] El comando '{command_str}' falló con código de retorno {process.returncode}")
        print("------- Inicio Salida del Comando -------")
        for line in output_lines:
            print(line, end='')
        print("------- Fin Salida del Comando -------")
    else:
        print(f"[INFO] Comando '{nombre_comando}' ejecutado exitosamente.")
    return process.returncode

def perform_real_esrgan_setup():
    """Configura el entorno para Real-ESRGAN. Se ejecuta solo una vez."""
    current_dir = os.getcwd()
    print("[INFO] Iniciando configuración de Real-ESRGAN...")
    print("[INFO] Paso 1: Desinstalando paquetes potencialmente conflictivos...")
    run_shell_command(f"{sys.executable} -m pip uninstall -y basicsr realesrgan facexlib gfpgan")
    
    print("\n[INFO] Paso 2: Clonando e instalando BasicSR...")
    if not os.path.exists("BasicSR"):
        run_shell_command("git clone https://github.com/xinntao/BasicSR.git")
    
    # Cambia al directorio de BasicSR y ejecuta comandos de setup
    path_basicsr = os.path.join(current_dir, "BasicSR")
    if os.path.exists(path_basicsr):
        os.chdir(path_basicsr)
        print("\n[INFO] Paso 2a: Modificando degradations.py en BasicSR...")
        run_shell_command(
            "find . -type f -name \"degradations.py\" "
            "-exec sed -i "
            "'s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/"
            "from torchvision.transforms.functional import rgb_to_grayscale/' {} \\;"
        )
        print("\n[INFO] Paso 2b: Instalando BasicSR...")
        run_shell_command(f"{sys.executable} -m pip install -e .")
        os.chdir(current_dir) # Regresa al directorio original
    else:
        print("[ERROR] No se encontró el directorio BasicSR después de clonar.")
    
    print("\n[INFO] Paso 3: Clonando e instalando Real-ESRGAN...")
    if not os.path.exists("Real-ESRGAN"):
        run_shell_command("git clone https://github.com/xinntao/Real-ESRGAN.git")

    path_realesrgan = os.path.join(current_dir, "Real-ESRGAN")
    if os.path.exists(path_realesrgan):
        os.chdir(path_realesrgan)
        print("\n[INFO] Paso 4: Instalando dependencias de Real-ESRGAN...")
        run_shell_command(f"{sys.executable} -m pip install -r requirements.txt")
        run_shell_command(f"{sys.executable} setup.py develop")
        print("\n[INFO] Paso 5: Aplicando parche opcional en Real-ESRGAN...")
        run_shell_command(
            "find . -type f -name \"*.py\" "
            "-exec sed -i "
            "'s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/"
            "from torchvision.transforms.functional import rgb_to_grayscale/' {} \\;"
        )
        os.chdir(current_dir) # Regresa al directorio original
    else:
        print("[ERROR] No se encontró el directorio Real-ESRGAN después de clonar.")
        
    print("\n[INFO] Proceso de configuración de Real-ESRGAN completado.")

# --- Funciones de Procesamiento de Imagen ---
def download_image_from_blob(blob_name, container_client_instance):
    """Descarga una imagen desde Azure Blob Storage."""
    blob_client = container_client_instance.get_blob_client(blob_name)
    try:
        download_stream = blob_client.download_blob()
        image_data = download_stream.readall()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image) # Convertir a array NumPy (OpenCV)
        
        width, height = image.size
        # print(f"[{threading.get_ident()}] [DEBUG] Imagen '{blob_name}' - Ancho: {width}px, Alto: {height}px") # Log de depuración
        
        if width > 600 or height > 600: # Límite de tamaño
            print(f"[{threading.get_ident()}] [ADVERTENCIA] Imagen '{blob_name}' descartada, excede el límite de 600px: {width}x{height}")
            return None
        return image_array
    except Exception as e:
        print(f"[{threading.get_ident()}] [ERROR] Fallo al descargar o abrir la imagen {blob_name}: {e}")
        return None

def improve_image_quality(image_cv2):
    """Mejora la calidad de la imagen usando Real-ESRGAN."""
    input_filename = f"temp_input_{uuid.uuid4().hex}.png" # Nombre de archivo temporal único
    cv2.imwrite(input_filename, image_cv2)
    
    os.makedirs("results", exist_ok=True) # Asegura que el directorio de resultados exista
    
    # Nombre esperado del archivo de salida de Real-ESRGAN
    expected_output_filename = os.path.join("results", input_filename.replace(".png", "_out.png"))
#    expected_output_filename = os.path.join(input_filename.replace(".png", "_out.png"))

    cmd = f"python Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x4plus -i {input_filename} --outscale 1.5 --face_enhance -o results"
    # print(f"[{threading.get_ident()}] [DEBUG] Ejecutando mejora de imagen: {cmd}") # Log de depuración
    
    return_code = run_shell_command(cmd) # Ejecuta el comando

    improved_image = None
    if return_code == 0 and os.path.exists(expected_output_filename):
        # print(f"[{threading.get_ident()}] [DEBUG] Mejora de imagen exitosa. Salida: {expected_output_filename}") # Log de depuración
        improved_image = cv2.imread(expected_output_filename)
        os.remove(expected_output_filename) # Limpia archivo de salida
    else:
        print(f"[{threading.get_ident()}] [ERROR] Fallo en la mejora de imagen o archivo de salida no encontrado: {expected_output_filename}")
    
    if os.path.exists(input_filename): # Limpia archivo de entrada
        os.remove(input_filename)
        
    return improved_image

def detect_vehicles_yolo(image_array, yolo_model_instance):
    """Detecta vehículos en la imagen usando el modelo YOLO."""
    results = yolo_model_instance(image_array) # Realiza la predicción
    boxes = results[0].boxes # Accede a las cajas detectadas
    names = yolo_model_instance.names # Nombres de las clases
    
    vehicle_types = ['car', 'truck', 'motorcycle', 'bus'] # Tipos de vehículos de interés
    # Mapeo de clases a español para el CSV (opcional, si se desea)
    # vehicle_types_es = {'car': 'coche', 'truck': 'camion', 'motorcycle': 'moto', 'bus': 'autobus'}
    
    vehicle_count = {v_type: 0 for v_type in vehicle_types}
    annotated_image = image_array.copy() # Copia para dibujar sobre ella
    all_confidences = []

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item()) # ID de la clase detectada
            conf = box.conf[0].item() # Confianza de la detección
            
            if conf < 0.25: # Umbral de confianza (ajustar si es necesario)
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Coordenadas del bounding box
            
            if cls_id < len(names):
                class_name = names[cls_id] # Nombre de la clase
                if class_name in vehicle_types:
                    vehicle_count[class_name] += 1
                    all_confidences.append(conf)
                    # Dibuja el bounding box y la etiqueta en la imagen
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Verde
                    label = f"{class_name} {conf:.2f}" # Etiqueta: "car 0.87"
                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA) # Texto negro
    
    average_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    return annotated_image, vehicle_count, average_confidence

def save_local_annotated_image(image_array, base_filename, download_dir_path):
    """Guarda la imagen anotada localmente."""
    os.makedirs(download_dir_path, exist_ok=True) # Asegura que el directorio exista
    # El nombre de archivo incluye "anotada_" para distinguirlo
    annotated_image_path = os.path.join(download_dir_path, f"anotada_{base_filename}")
    try:
        cv2.imwrite(annotated_image_path, image_array)
        return annotated_image_path
    except Exception as e:
        print(f"[{threading.get_ident()}] [ERROR] Fallo al guardar imagen anotada localmente {annotated_image_path}: {e}")
        return None

def upload_image_to_blob(local_image_path, target_blob_name, container_client_instance):
    """Sube una imagen desde una ruta local a Azure Blob Storage."""
    try:
        with open(local_image_path, 'rb') as data:
            blob_client = container_client_instance.get_blob_client(target_blob_name)
            blob_client.upload_blob(data, overwrite=True) # Sobrescribe si ya existe
        print(f"[{threading.get_ident()}] [INFO] Imagen {local_image_path} subida a Azure como {target_blob_name}")
        os.remove(local_image_path) # Limpia el archivo local después de subirlo
    except Exception as e:
        print(f"[{threading.get_ident()}] [ERROR] Fallo al subir {local_image_path} a Azure: {e}")

def extract_metadata(filename):
    """Extrae el ID de la cámara y el timestamp del nombre del archivo."""
    try:
        # Formato esperado: captura_IDCAMARA_YYYYMMDD_HHMMSS.png
        parts = filename.split('_')
        if len(parts) < 4 or not filename.endswith('.png') or parts[0] != 'captura':
            raise ValueError("Formato de nombre de archivo no válido para extracción de metadatos.")
        
        camera_id = parts[1]
        timestamp_str = parts[2] + '_' + parts[3].split('.')[0] # Une fecha y hora: 'YYYYMMDD_HHMMSS'
        dt_object = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S") # Convierte a objeto datetime
        readable_timestamp = dt_object.strftime("%Y-%m-%d %H:%M:%S") # Formato legible
        return camera_id, readable_timestamp
    except ValueError as e:
        # print(f"[{threading.get_ident()}] [ADVERTENCIA] No se pudieron extraer metadatos de {filename}: {e}") # Log de depuración
        return "desconocido", datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Valores por defecto

# --- Funciones de Manejo de CSV (Seguras para Hilos) ---
def load_csv_data_from_blob(csv_client_instance):
    """Carga los datos del CSV desde Azure Blob Storage."""
    try:
        download_stream = csv_client_instance.download_blob()
        csv_content = download_stream.readall().decode('utf-8').splitlines()
        return list(csv.reader(csv_content))
    except Exception: # Si el blob no existe u ocurre otro error
        return [] # Devuelve una lista vacía

def append_data_to_csv_blob(vehicle_data_row, csv_client_instance, headers_csv):
    """Añade una fila de datos al CSV en Azure Blob Storage de forma segura para hilos."""
    with csv_lock: # Adquiere el bloqueo para acceso exclusivo al CSV
        rows = load_csv_data_from_blob(csv_client_instance)
        # Si el CSV está vacío o las cabeceras no coinciden, empieza uno nuevo con cabeceras
        if not rows or rows[0] != headers_csv: 
            rows = [headers_csv]
        
        rows.append(vehicle_data_row) # Añade la nueva fila
        
        output = io.StringIO() # Buffer en memoria para escribir el CSV
        writer = csv.writer(output)
        writer.writerows(rows)
        csv_text = output.getvalue() # Obtiene el contenido del CSV como texto
        try:
            csv_client_instance.upload_blob(csv_text, overwrite=True) # Sube el CSV actualizado
            # print(f"[{threading.get_ident()}] [DEBUG] Datos CSV añadidos y subidos para imagen {vehicle_data_row[6]}.") # Log de depuración
        except Exception as e:
            print(f"[{threading.get_ident()}] [ERROR] Fallo al subir el CSV: {e}")

# --- Tarea Principal de Procesamiento de Imagen (para cada hilo) ---
def process_single_image(blob_name,
                         source_folder_name, # Carpeta de la cámara, ej: 'camara_10302/'
                         yolo_model, # Instancia del modelo YOLO compartida
                         orig_container_client, # Cliente para leer imágenes originales
                         gio_container_client, # Cliente para escribir resultados (imágenes anotadas, CSV)
                         csv_client_for_results, # Cliente específico para el blob del CSV
                         current_processed_set, # Conjunto de imágenes ya procesadas (compartido)
                         checkpoint_filepath, # Ruta al archivo de checkpoint
                         local_download_dir): # Directorio local para descargas temporales
    
    thread_id = threading.get_ident() # Identificador del hilo actual
    base_image_filename = blob_name.split('/')[-1] # Nombre base del archivo, ej: captura_XXXX_YYYYMMDD_HHMMSS.png

    # Verificación final de elegibilidad para procesamiento (segura para hilos)
    with checkpoint_lock:
        if blob_name in current_processed_set:
            # print(f"[{thread_id}] [OMITIR] Imagen {blob_name} ya procesada (encontrada en conjunto global).") # Log de depuración
            return
        # Solo procesar imágenes que comiencen con 'captura_'
        if not base_image_filename.startswith('captura_'):
             print(f"[{thread_id}] [OMITIR] Imagen {blob_name} no comienza con 'captura_'. Registrando en checkpoint.")
             # Marcar como procesada para evitar revisiones futuras si no es un archivo objetivo
             current_processed_set.add(blob_name)
             with open(checkpoint_filepath, 'a') as f:
                 f.write(blob_name + '\n')
             return

    print(f"[{thread_id}] [PROCESANDO] Imagen: {blob_name}")
    original_image_cv2 = download_image_from_blob(blob_name, orig_container_client)
    
    if original_image_cv2 is None:
        # El error ya fue impreso por download_image_from_blob o se omitió por tamaño
        # Marcar como procesada para evitar reintentar descargas problemáticas o imágenes muy grandes
        with checkpoint_lock:
            if blob_name not in current_processed_set: # Doble verificación por si acaso
                 current_processed_set.add(blob_name)
                 with open(checkpoint_filepath, 'a') as f:
                     f.write(blob_name + '\n')
        return

    improved_image_cv2 = improve_image_quality(original_image_cv2)
    if improved_image_cv2 is None:
        print(f"[{thread_id}] [ADVERTENCIA] Fallo en mejora de imagen para {blob_name}. Usando original para detección.")
        improved_image_cv2 = original_image_cv2 # Usa la imagen original si la mejora falla

    # Detección de vehículos
    annotated_img, vehicle_counts, trust_score = detect_vehicles_yolo(improved_image_cv2, yolo_model)
    
    # Guardar imagen anotada localmente antes de subirla
    # El nombre base ya no incluye la carpeta de la cámara, solo el nombre del archivo
    local_annotated_path = save_local_annotated_image(annotated_img, base_image_filename, local_download_dir)

    if local_annotated_path:
        # Subir a Azure GIO en la carpeta 'revisadas/'
        # target_blob_path_in_gio será algo como "revisadas/captura_XXXX_YYYYMMDD_HHMMSS.png"
        target_blob_path_in_gio = f"revisadas/{base_image_filename}"
        upload_image_to_blob(local_annotated_path, target_blob_path_in_gio, gio_container_client)

    # Extraer metadatos y preparar datos para CSV
    cam_id, img_timestamp = extract_metadata(base_image_filename)
    
    vehicle_data_entry = [
        img_timestamp, # 'timestamp'
        cam_id, # 'id_camara'
        vehicle_counts.get('car', 0), # 'coche'
        vehicle_counts.get('truck', 0), # 'camion'
        vehicle_counts.get('motorcycle', 0), # 'moto'
        vehicle_counts.get('bus', 0), # 'autobus'
        blob_name,  # 'nombre_imagen_original' (ruta completa del blob original)
        datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 'fecha_procesado'
    ]
    append_data_to_csv_blob(vehicle_data_entry, csv_client_for_results, CSV_HEADERS)

    # Marcar como procesada (seguro para hilos)
    with checkpoint_lock:
        current_processed_set.add(blob_name)
        with open(checkpoint_filepath, 'a') as f:
            f.write(blob_name + '\n')
    print(f"[{thread_id}] [COMPLETADO] Finalizado: {blob_name}")


# --- Ejecución Principal ---
if __name__ == "__main__":
    print("[INFO] Iniciando script de procesamiento de imágenes...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True) # Crea el directorio de descarga local si no existe

    # 1. Realizar configuración de Real-ESRGAN si no se ha hecho antes
    if not os.path.exists(REAL_ESRGAN_SETUP_DONE_FLAG):
        print("[INFO] Realizando configuración inicial para Real-ESRGAN (puede tardar)...")
        perform_real_esrgan_setup()
        # Crea el archivo bandera para indicar que la configuración se ha completado
        with open(REAL_ESRGAN_SETUP_DONE_FLAG, 'w') as flag_file:
            flag_file.write(f"Configuración completada el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("[INFO] Configuración de Real-ESRGAN marcada como completa.")
    else:
        print("[INFO] La configuración de Real-ESRGAN ya se ha realizado anteriormente. Omitiendo.")

    # 2. Cargar Modelo YOLO
    print(f"[INFO] Cargando modelo YOLO desde: {YOLO_MODEL_PATH}")
    model_yolo = YOLO(YOLO_MODEL_PATH)
    print("[INFO] Modelo YOLO cargado exitosamente.")

    # 3. Inicializar Clientes de Azure Blob Service
    blob_service_client_orig = BlobServiceClient.from_connection_string(CONNECTION_STRING_ORIG)
    container_client_orig = blob_service_client_orig.get_container_client(CONTAINER_NAME_ORIG)
    
    blob_service_client_gio = BlobServiceClient.from_connection_string(CONNECTION_STRING_GIO)
    container_client_gio = blob_service_client_gio.get_container_client(CONTAINER_NAME_GIO)
    csv_blob_client = container_client_gio.get_blob_client(CSV_BLOB_NAME) # Cliente para el CSV de resultados
    print("[INFO] Clientes de Azure Blob Service inicializados.")

    # 4. Cargar Checkpoint (imágenes ya procesadas)
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            # Carga las líneas del archivo en el conjunto, eliminando espacios en blanco
            processed_images_set.update(line.strip() for line in f if line.strip())
        print(f"[INFO] Cargados {len(processed_images_set)} registros de imágenes procesadas desde '{CHECKPOINT_FILE}'.")
    except FileNotFoundError:
        print(f"[INFO] Archivo de checkpoint '{CHECKPOINT_FILE}' no encontrado. Se creará uno nuevo.")
        with open(CHECKPOINT_FILE, 'w'): # Crea el archivo vacío si no existe
            pass
            
    # 5. Determinar número de hilos trabajadores paralelos
    # Ajustar según los núcleos de CPU del servidor, capacidades de GPU y rendimiento de E/S.
#    num_workers = os.cpu_count() - 4 # Usa el número de CPUs disponibles
    num_workers =5
    if num_workers is None: # Si no se puede determinar, usa un valor por defecto
        num_workers = 4 
    print(f"[INFO] Usando {num_workers} hilos trabajadores en paralelo.")

    # 6. Procesar imágenes usando ThreadPoolExecutor para paralelización
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [] # Lista para almacenar los objetos Future de las tareas enviadas
        for camera_folder_prefix in CARPETAS:
            print(f"[INFO] Obteniendo listado de blobs para la carpeta de cámara: {camera_folder_prefix}...")
            try:
                # Lista los blobs que comienzan con el prefijo de la carpeta de la cámara
                blobs_in_folder = list(container_client_orig.list_blobs(name_starts_with=camera_folder_prefix))
                print(f"[INFO] Se encontraron {len(blobs_in_folder)} blobs en '{camera_folder_prefix}'.")
                for blob_item in blobs_in_folder:
                    blob_item_name = blob_item.name
                    # Comprobación rápida inicial (sin bloqueo) para evitar enviar tareas para imágenes ya procesadas o no deseadas
                    if blob_item_name not in processed_images_set and blob_item_name.split('/')[-1].startswith('captura_'):
                        # Envía la tarea de procesamiento de la imagen al ejecutor
                        future = executor.submit(process_single_image,
                                                 blob_item_name,
                                                 camera_folder_prefix,
                                                 model_yolo, # Modelo YOLO
                                                 container_client_orig, # Cliente para imágenes originales
                                                 container_client_gio, # Cliente para resultados
                                                 csv_blob_client, # Cliente para el CSV
                                                 processed_images_set, # Conjunto de imágenes procesadas
                                                 CHECKPOINT_FILE, # Ruta al archivo de checkpoint
                                                 DOWNLOAD_DIR) # Directorio de descarga local
                        futures.append(future)
                    # else:
                        # print(f"[DEBUG][MAIN] Omitiendo envío de tarea para {blob_item_name} (ya procesada o no es 'captura_')") # Log de depuración
            except Exception as e:
                print(f"[ERROR] Error al listar blobs para la carpeta '{camera_folder_prefix}': {e}")
        
        print(f"[INFO] Se han enviado {len(futures)} tareas de procesamiento al ejecutor. Esperando finalización...")
        # Espera a que todas las tareas enviadas se completen
        for i, future_item in enumerate(futures):
            try:
                future_item.result() # Espera a que la tarea termine y obtiene el resultado (o lanza excepción si la hubo)
                # print(f"[DEBUG] Tarea {i+1}/{len(futures)} completada.") # Log de depuración muy verboso
            except Exception as e:
                print(f"[ERROR] Ocurrió un error durante la ejecución de una tarea en un hilo trabajador: {e}")
                # Descomentar para obtener una traza de error más detallada
                # print(traceback.format_exc())

    print("[INFO] Todas las tareas de procesamiento de imágenes han sido completadas.")
    print("[INFO] Script finalizado.")
