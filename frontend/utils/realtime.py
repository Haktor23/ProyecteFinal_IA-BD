import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import time
from flask import jsonify, Response, request # Asegúrate de importar request
import base64

# Configuración del modelo YOLO
model = YOLO("yolo11x.pt") # Asegúrate que este path es correcto y accesible
vehicle_classes = [2, 3, 5, 7]  # COCO classes: car(2), motorcycle(3), bus(5), truck(7)

cameras_list = [ # Renombrada para evitar conflicto de nombres si 'cameras' se usa como variable
    '10302', '10304', '10305',
    '11502', '11504', '11505',
    '11402', '11403', '11404',
    '2602', '2604', '2605',
    '3103', '3105', '3106',
    '10510', '10511', '10512',
    '14203', '14204', '14206',
    '6004', '6011', '6012',
    '4302', '4304', '4305'
]

FFMPEG_PATH = 'C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe' # Definir como constante

def capture_realtime_frame(camera_id, ffmpeg_path=FFMPEG_PATH):
    """Captura el frame adaptándose a cualquier resolución"""
    rtsp_url = f'rtsp://camaras.valencia.es/stream/{camera_id}/1'
    
    command = [
        ffmpeg_path,
        '-y',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-frames:v', '1',
        '-f', 'image2', # Para salida de imagen cruda
        '-c:v', 'mjpeg', # Para asegurar que es JPEG si es posible, o usa 'png'
        '-qscale:v', '2', # Calidad de la imagen
        '-' # Salida a stdout
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True, # Lanza excepción si ffmpeg falla
            timeout=20 # Aumentado el timeout
        )
        
        # Decodificar la imagen desde el buffer de memoria
        img_array = np.frombuffer(result.stdout, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Error decodificando imagen de cámara {camera_id}: imagen es None. Tamaño de stdout: {len(result.stdout)}")
            return None
            
        return img
    
    except subprocess.TimeoutExpired:
        print(f"Error capturando cámara {camera_id}: Timeout (20s)")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error capturando cámara {camera_id} con FFMPEG: {e}")
        print(f"FFMPEG stderr: {e.stderr.decode('utf-8', errors='ignore')}")
        return None
    except Exception as e:
        print(f"Error inesperado capturando cámara {camera_id}: {str(e)}")
        return None

def process_with_yolo(frame):
    """Procesamiento mejorado con YOLO"""
    if frame is None or frame.size == 0:
        print("Error en process_with_yolo: frame es None o vacío.")
        return None, 0
    
    try:
        # No es necesario redimensionar a 640xN si YOLO puede manejar diferentes tamaños
        # o si prefieres procesar en la resolución original (puede ser más lento)
        # Si quieres redimensionar:
        # height, width = frame.shape[:2]
        # new_height = 640
        # new_width = int(width * (new_height / height))
        # resized_frame = cv2.resize(frame, (new_width, new_height))
        # results = model(resized_frame, verbose=False)
        
        results = model(frame, verbose=False) # Procesar el frame original
        
        # Filtrar detecciones para incluir solo las clases de vehículos
        vehicle_detections = []
        if results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls) in vehicle_classes:
                    vehicle_detections.append(box)
        
        # Si usas una versión de ultralytics que no permite modificar results[0].boxes directamente
        # necesitas crear un nuevo objeto de resultados o dibujar manualmente.
        # Por simplicidad, asumimos que plot() dibujará todas las detecciones y contamos las filtradas.
        annotated_frame = results[0].plot() 
        count = sum(1 for box in results[0].boxes if int(box.cls) in vehicle_classes) if results[0].boxes else 0
        
        # Convertir frame a JPEG para mostrar en HTML
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            print("Error en process_with_yolo: Falló cv2.imencode.")
            return None, 0
        
        return buffer.tobytes(), count
    except Exception as e:
        print(f"Error durante el procesamiento YOLO: {str(e)}")
        return None, 0

def _get_detection_data_internal(camera_id):
    """
    Lógica interna para capturar y procesar una imagen.
    Retorna un diccionario con los datos o un error.
    """
    if camera_id not in cameras_list:
        return {'error': f'Cámara ID "{camera_id}" no es válida.'}

    start_time = time.time()
    
    frame = capture_realtime_frame(camera_id)
    if frame is None:
        return {'error': f'No se pudo capturar la imagen de la cámara {camera_id}. Revisa los logs del servidor.'}
    
    original_h, original_w = frame.shape[:2]
    
    processed_image_bytes, count = process_with_yolo(frame)
    if processed_image_bytes is None:
        return {'error': f'Error procesando la imagen de la cámara {camera_id} con YOLO.'}
    
    processing_time = time.time() - start_time
    fps = 1 / processing_time if processing_time > 0 else float('inf')
    
    image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
    
    return {
        'camera_id': camera_id,
        'vehicle_count': count,
        'resolution': f'{original_w}x{original_h}', # Resolución original del frame capturado
        'processing_fps': round(fps, 2),
        'image': f'data:image/jpeg;base64,{image_base64}'
    }

def obtener_contexto():
    """
    Prepara el contexto para la plantilla realtime.html.
    """
    contexto = {
        'titulo': "Detección de Vehículos en Tiempo Real",
        'descripcion': "Selecciona una cámara para ver el análisis de tráfico y conteo de vehículos.",
        'camaras_disponibles': cameras_list,
        'camara': None, # Cámara actualmente seleccionada
        'datos_deteccion': None, # Resultados de la detección
        'card_stats': { # Datos para las tarjetas de estadísticas
            'active_cameras': len(cameras_list), # Ejemplo, podría ser más dinámico
            'realtime_status': "Activo",
            'ai_models': "YOLOvX" # Ejemplo
        }
    }

    selected_camera_id = request.args.get('camara')

    if selected_camera_id:
        contexto['camara'] = selected_camera_id
        if selected_camera_id in cameras_list:
            print(f"Obteniendo datos para la cámara: {selected_camera_id}")
            contexto['datos_deteccion'] = _get_detection_data_internal(selected_camera_id)
        else:
            contexto['datos_deteccion'] = {'error': f'La cámara "{selected_camera_id}" no es válida.'}
    else:
        contexto['datos_deteccion'] = {'message': 'Por favor, selecciona una cámara para iniciar la detección.'}
        
    return contexto

# --- Endpoints API (si los necesitas para otros propósitos) ---
# Estas funciones devuelven respuestas JSON y no son usadas directamente por render_template

def list_available_cameras_api():
    """Endpoint API para listar cámaras disponibles"""
    return jsonify({
        'cameras': cameras_list,
        'count': len(cameras_list)
    })

def detect_vehicles_api(camera_id):
    """Endpoint API principal para detección de vehículos"""
    if camera_id not in cameras_list:
        return jsonify({
            'error': 'Cámara no encontrada',
            'available_cameras': cameras_list
        }), 404
    
    data = _get_detection_data_internal(camera_id) # Usa la lógica interna
    return jsonify(data)

def welcome_message_api():
    """Endpoint API de bienvenida"""
    # Actualiza los nombres de tus endpoints si los cambias en app.py
    return jsonify({
        'message': 'API de detección de vehículos en tiempo real',
        'endpoints': {
            '/api/rt/cameras': 'Listado de cámaras disponibles (GET)',
            '/api/rt/detect/<camera_id>': 'Detección en cámara específica (GET)'
        }
    })

# La función `procesar_datos(data)` que tu `app.py` espera para `/api/realtime` (POST)
# no está definida aquí. Si la necesitas, tendrías que implementarla.
# def procesar_datos(data_from_post):
#     # Lógica para manejar datos POST, si es necesario para otra funcionalidad
#     pass