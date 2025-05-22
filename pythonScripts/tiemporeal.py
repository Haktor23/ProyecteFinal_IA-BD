import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import time
from flask import Flask, jsonify, Response
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuración del modelo YOLO
model = YOLO("yolo11x.pt")

vehicle_classes = [2, 3, 5, 7]  # COCO classes: car(2), motorcycle(3), bus(5), truck(7)

cameras = [
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

def capture_realtime_frame(camera_id, ffmpeg_path='C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe'):
    """Captura el frame adaptándose a cualquier resolución"""
    rtsp_url = f'rtsp://camaras.valencia.es/stream/{camera_id}/1'
    
    command = [
        ffmpeg_path,
        '-y',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-frames:v', '1',
        '-f', 'image2',
        '-qscale:v', '2',
        '-'
    ]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            timeout=15
        )
        
        img = cv2.imdecode(
            np.frombuffer(result.stdout, np.uint8), 
            cv2.IMREAD_COLOR
        )
        
        if img is None:
            raise ValueError("Imagen decodificada es None")
            
        return img
    
    except Exception as e:
        print(f"Error capturando cámara {camera_id}: {str(e)}")
        return None

def process_with_yolo(frame):
    """Procesamiento mejorado con YOLO"""
    if frame is None or frame.size == 0:
        return None, 0
    
    height, width = frame.shape[:2]
    new_height = 640
    new_width = int(width * (new_height / height))
    
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    results = model(resized_frame, verbose=False)
    
    if results[0].boxes:
        mask = np.isin(results[0].boxes.cls.cpu().numpy(), vehicle_classes)
        results[0].boxes = results[0].boxes[mask]
    
    annotated_frame = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes else 0
    
    # Convertir frame a JPEG
    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    if not ret:
        return None, 0
    
    return buffer.tobytes(), count

@app.route('/cameras', methods=['GET'])
def get_cameras():
    """Endpoint para listar cámaras disponibles"""
    return jsonify({
        'cameras': cameras,
        'count': len(cameras)
    })

@app.route('/detect/<camera_id>', methods=['GET'])
def detect_vehicles(camera_id):
    """Endpoint principal para detección de vehículos"""
    if camera_id not in cameras:
        return jsonify({
            'error': 'Cámara no encontrada',
            'available_cameras': cameras
        }), 404
    
    start_time = time.time()
    
    # Capturar frame
    frame = capture_realtime_frame(camera_id)
    if frame is None:
        return jsonify({
            'error': 'Error capturando imagen',
            'camera_id': camera_id
        }), 500
    
    # Procesar con YOLO
    processed_image, count = process_with_yolo(frame)
    if processed_image is None:
        return jsonify({
            'error': 'Error procesando imagen',
            'camera_id': camera_id
        }), 500
    
    # Calcular métricas
    h, w = frame.shape[:2]
    fps = 1 / (time.time() - start_time)
    
    # Convertir imagen a base64
    image_base64 = base64.b64encode(processed_image).decode('utf-8')
    
    return jsonify({
        'camera_id': camera_id,
        'vehicle_count': count,
        'resolution': f'{w}x{h}',
        'processing_fps': round(fps, 2),
        'image': f'data:image/jpeg;base64,{image_base64}'
    })

@app.route('/', methods=['GET'])
def home():
    """Endpoint de bienvenida"""
    return jsonify({
        'message': 'API de detección de vehículos',
        'endpoints': {
            '/cameras': 'Listado de cámaras disponibles',
            '/detect/<camera_id>': 'Detección en cámara específica'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)