import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import time
from flask import jsonify, Response, request
import base64

# Configuración del modelo YOLO
model = YOLO("yolo11x.pt")
vehicle_classes = [2, 3, 5, 7]  # COCO classes: car(2), motorcycle(3), bus(5), truck(7)

# Mapeo de ubicación -> lista de cámaras
camera_fiware_mapping = {
    'AVFRANCIA': ["10302", "10304", "10305"],
    'BULEVARDSUD': ["11502", "11504", "11505"],
    'MOLISOL': ["11402", "11403", "11404"],
    'PISTASILLA': ["2602", "2604", "2605"],
    'VIVERS': ["3103", "3105", "3106"],
    'CENTRE': ["10510", "10511", "10512"],
    'DR_LLUCH': ["14203", "14204", "14206"],
    'OLIVERETA': ["6004", "6011", "6012"],
    'PATRAIX': ["4302", "4304", "4305"],
}

# Lista de cámaras base (solo IDs)
cameras_list = [camera for sublist in camera_fiware_mapping.values() for camera in sublist]

# Generar lista con nombres descriptivos tipo "OLIVERETA 1 (6004)"
camera_labels = []
camera_id_to_label = {}

for location, ids in camera_fiware_mapping.items():
    for i, cam_id in enumerate(ids, start=1):
        label = f"{location} {i} ({cam_id})"
        camera_labels.append({'id': cam_id, 'label': label})
        camera_id_to_label[cam_id] = label

FFMPEG_PATH = 'C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe'

def capture_realtime_frame(camera_id, ffmpeg_path=FFMPEG_PATH):
    rtsp_url = f'rtsp://camaras.valencia.es/stream/{camera_id}/1'
    command = [
        ffmpeg_path, '-y', '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-frames:v', '1',
        '-f', 'image2', '-c:v', 'mjpeg', '-qscale:v', '2', '-'
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            timeout=20
        )
        img_array = np.frombuffer(result.stdout, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error decodificando imagen de cámara {camera_id}.")
            return None
        return img
    except subprocess.TimeoutExpired:
        print(f"Timeout al capturar cámara {camera_id}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error FFMPEG cámara {camera_id}: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado cámara {camera_id}: {str(e)}")
        return None

def process_with_yolo(frame):
    if frame is None or frame.size == 0:
        return None, 0
    try:
        results = model(frame, verbose=False)
        vehicle_detections = []
        if results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls) in vehicle_classes:
                    vehicle_detections.append(box)
        annotated_frame = results[0].plot()
        count = sum(1 for box in results[0].boxes if int(box.cls) in vehicle_classes) if results[0].boxes else 0
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            return None, 0
        return buffer.tobytes(), count
    except Exception as e:
        print(f"Error YOLO: {str(e)}")
        return None, 0

def _get_detection_data_internal(camera_id):
    if camera_id not in cameras_list:
        return {'error': f'Cámara ID "{camera_id}" no es válida.'}
    start_time = time.time()
    frame = capture_realtime_frame(camera_id)
    if frame is None:
        return {'error': f'No se pudo capturar imagen de la cámara {camera_id}'}
    original_h, original_w = frame.shape[:2]
    processed_image_bytes, count = process_with_yolo(frame)
    if processed_image_bytes is None:
        return {'error': f'Error procesando imagen de la cámara {camera_id}'}
    processing_time = time.time() - start_time
    fps = 1 / processing_time if processing_time > 0 else float('inf')
    image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
    return {
        'camera_id': camera_id,
        'camera_label': camera_id_to_label.get(camera_id, camera_id),
        'vehicle_count': count,
        'resolution': f'{original_w}x{original_h}',
        'processing_fps': round(fps, 2),
        'image': f'data:image/jpeg;base64,{image_base64}'
    }

def obtener_contexto():
    contexto = {
        'titulo': "Detección de Vehículos en Tiempo Real",
        'descripcion': "Selecciona una cámara para ver el análisis de tráfico y conteo de vehículos.",
        'camaras_disponibles': camera_labels,
        'camara': None,
        'datos_deteccion': None,
        'card_stats': {
            'active_cameras': len(cameras_list),
            'realtime_status': "Activo",
            'ai_models': "YOLOvX"
        }
    }
    selected_camera_id = request.args.get('camara')
    if selected_camera_id:
        contexto['camara'] = selected_camera_id
        if selected_camera_id in cameras_list:
            contexto['datos_deteccion'] = _get_detection_data_internal(selected_camera_id)
        else:
            contexto['datos_deteccion'] = {'error': f'La cámara "{selected_camera_id}" no es válida.'}
    else:
        contexto['datos_deteccion'] = {'message': 'Por favor, selecciona una cámara para iniciar la detección.'}
    return contexto

# --- Endpoints API ---

def list_available_cameras_api():
    return jsonify({
        'cameras': camera_labels,
        'count': len(camera_labels)
    })

def detect_vehicles_api(camera_id):
    if camera_id not in cameras_list:
        return jsonify({
            'error': 'Cámara no encontrada',
            'available_cameras': camera_labels
        }), 404
    data = _get_detection_data_internal(camera_id)
    return jsonify(data)

def welcome_message_api():
    return jsonify({
        'message': 'API de detección de vehículos en tiempo real',
        'endpoints': {
            '/api/rt/cameras': 'Listado de cámaras disponibles (GET)',
            '/api/rt/detect/<camera_id>': 'Detección en cámara específica (GET)'
        }
    })
