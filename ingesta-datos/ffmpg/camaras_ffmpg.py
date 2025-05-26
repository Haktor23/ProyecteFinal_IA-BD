import subprocess
import os
import datetime
import time
import concurrent.futures
import numpy as np
import cv2
from PIL import Image
import logging
import shutil
import sys
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('CapturaRTSP')

def is_image_dark(image_path, threshold=30):
    """
    Verifica si una imagen está demasiado oscura (posiblemente una pantalla negra)
    
    Args:
        image_path (str): Ruta a la imagen
        threshold (int): Valor umbral de brillo (0-255)
        
    Returns:
        bool: True si la imagen está demasiado oscura
    """
    try:
        # Abrir imagen usando OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return True  # No se pudo leer la imagen
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calcular brillo promedio
        avg_brightness = np.mean(gray)
        
        # Comprobar si está por debajo del umbral
        return avg_brightness < threshold
    except Exception as e:
        logger.error(f"Error al verificar oscuridad de imagen: {str(e)}")
        return True  # En caso de error, asumir que la imagen no es válida

def capture_frame(camera_id, base_dir="capturas", max_retries=5, ffmpeg_path=None):
    """
    Captura un frame de una cámara RTSP y lo guarda en una carpeta específica.
    Reintenta hasta 5 veces si la imagen es muy oscura.
    
    Args:
        camera_id (str): ID de la cámara
        base_dir (str): Directorio base para guardar las capturas
        max_retries (int): Número máximo de reintentos para imágenes oscuras
        ffmpeg_path (str): Ruta al ejecutable de FFmpeg (opcional)
        
    Returns:
        tuple: (bool, str) - (éxito, mensaje)
    """
    # Crear el directorio para esta cámara si no existe
    camera_dir = os.path.join(base_dir, f"camara_{camera_id}")
    os.makedirs(camera_dir, exist_ok=True)
    
    # Determinar la ruta correcta a FFmpeg
    ffmpeg_executable = ffmpeg_path if ffmpeg_path else "ffmpeg"
    dark_retries = 0
    # Registrar información de diagnóstico
    if dark_retries == 0:
        logger.info(f"Usando ejecutable FFmpeg: {shutil.which(ffmpeg_executable) or 'No encontrado en PATH'}")
    
    
    while dark_retries <= max_retries:
        # Generar nombre de archivo con marca de tiempo
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image = os.path.join(camera_dir, f"camara_{camera_id}_{timestamp}.jpg")
        
        # URL RTSP para esta cámara
        rtsp_url = f'rtsp://camaras.valencia.es/stream/{camera_id}/1'
        
        # Comando FFmpeg con parámetros ajustados para mejores resultados
        command = [
            ffmpeg_executable,
            '-y',                       # Sobrescribir archivo si existe
            '-rtsp_transport', 'tcp',   # Usar TCP para transporte RTSP
            '-i', rtsp_url,             # Stream de entrada
            '-frames:v', '1',           # Capturar solo un frame
            '-qscale:v', '2',           # Escala de calidad (valor más bajo = mayor calidad)
            '-vf', 'unsharp=5:5:1.0:5:5:0.0,eq=contrast=1.1:brightness=0.02',  # Filtros de video
            output_image                # Archivo de salida
        ]
        
        try:
            # Ejecutar el comando con timeout
            result = subprocess.run(
                command, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=30  # 30 segundos de timeout
            )
            
            # Verificar si se creó la imagen
            if not os.path.exists(output_image):
                if dark_retries >= max_retries:
                    return False, f'No se generó la imagen para la cámara {camera_id} después de {max_retries+1} intentos'
                dark_retries += 1
                logger.warning(f"Intento {dark_retries}/{max_retries+1}: No se generó imagen para cámara {camera_id}")
                time.sleep(2)  # Esperar un poco antes de reintentar
                continue
                
            # Verificar si la imagen está oscura
            if is_image_dark(output_image):
                if dark_retries >= max_retries:
                    logger.warning(f"⚠️ Cámara {camera_id}: Imagen oscura después de {max_retries+1} intentos. Programando nuevo intento en 1 minuto.")
                    # Mantener la última imagen para referencia pero agregar _dark al nombre
                    dark_image = output_image.replace(".jpg", "_dark.jpg")
                    os.rename(output_image, dark_image)
                    
                    # Programar un nuevo intento en 1 minuto (retornamos False para la ejecución actual)
                    return False, f"Imagen oscura después de {max_retries+1} intentos. Guardada como {dark_image}. Reintentar en 1 minuto."
                
                # Eliminar imagen oscura y reintentar
                os.remove(output_image)
                dark_retries += 1
                logger.warning(f"Intento {dark_retries}/{max_retries+1}: Imagen oscura para cámara {camera_id}, reintentando...")
                time.sleep(2)  # Esperar un poco antes de reintentar
                continue
            
            # Si llegamos aquí, tenemos una imagen válida
            return True, f'Imagen capturada correctamente: {output_image}'
                
        except subprocess.CalledProcessError as e:
            if dark_retries >= max_retries:
                return False, f'Error al capturar imagen de cámara {camera_id}: {e.stderr}'
            dark_retries += 1
            logger.warning(f"Intento {dark_retries}/{max_retries+1}: Error FFmpeg para cámara {camera_id}: {e.stderr}")
            time.sleep(2)
            
        except subprocess.TimeoutExpired:
            if dark_retries >= max_retries:
                return False, f'Timeout al capturar imagen de cámara {camera_id} después de {max_retries+1} intentos'
            dark_retries += 1
            logger.warning(f"Intento {dark_retries}/{max_retries+1}: Timeout para cámara {camera_id}")
            time.sleep(2)
            
        except Exception as e:
            if dark_retries >= max_retries:
                return False, f'Error inesperado con la cámara {camera_id}: {str(e)}'
            dark_retries += 1
            logger.warning(f"Intento {dark_retries}/{max_retries+1}: Error para cámara {camera_id}: {str(e)}")
            time.sleep(2)
    
    # No deberíamos llegar aquí, pero por si acaso
    return False, f'No se pudo capturar una imagen válida para la cámara {camera_id}'

def capture_all_cameras(camera_ids, max_workers=4):
    """
    Captura frames de múltiples cámaras en paralelo
    
    Args:
        camera_ids (list): Lista de IDs de cámaras
        max_workers (int): Número máximo de procesos paralelos
    """
    logger.info(f"Iniciando captura para {len(camera_ids)} cámaras...")
    
    # Diccionario para almacenar cámaras que necesitan reintento
    retry_cameras = {}
    
    # Usar ThreadPoolExecutor para procesar múltiples cámaras en paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Mapear la función capture_frame a cada ID de cámara
        futures = {executor.submit(capture_frame, camera_id): camera_id for camera_id in camera_ids}
        
        # Procesar resultados a medida que se completan
        for future in concurrent.futures.as_completed(futures):
            camera_id = futures[future]
            try:
                success, message = future.result()
                if success:
                    logger.info(f"✅ Cámara {camera_id}: {message}")
                else:
                    logger.warning(f"❌ Cámara {camera_id}: {message}")
                    # Si el mensaje sugiere reintentar en 1 minuto, agregar a la lista de reintentos
                    if "Reintentar en 1 minuto" in message:
                        retry_cameras[camera_id] = time.time() + 60  # Programar reintento para 1 minuto después
            except Exception as e:
                logger.error(f"❌ Error inesperado con la cámara {camera_id}: {str(e)}")
    
    # Devolver las cámaras que necesitan reintento
    return retry_cameras

def monitor_cameras(camera_ids, interval=300, run_forever=False, max_workers=4):
    """
    Monitorea continuamente un conjunto de cámaras, manejando reintentos automáticos
    
    Args:
        camera_ids (list): Lista de IDs de cámaras
        interval (int): Intervalo entre capturas en segundos (por defecto 5 minutos)
        run_forever (bool): Si es True, ejecuta indefinidamente
        max_workers (int): Número máximo de procesos paralelos
    """
    pending_retries = {}  # Diccionario de cámaras pendientes de reintento {camera_id: timestamp_para_reintento}
    
    end_time = None if run_forever else time.time() + 3600  # Por defecto, ejecutar durante 1 hora si no es forever
    
    next_regular_capture = time.time()  # Inicialmente, hacer captura inmediatamente
    
    while run_forever or time.time() < end_time:
        current_time = time.time()
        
        # Verificar si hay reintentos pendientes que debemos ejecutar ahora
        retries_now = [cam_id for cam_id, retry_time in pending_retries.items() if current_time >= retry_time]
        if retries_now:
            logger.info(f"Ejecutando {len(retries_now)} reintentos programados...")
            # Ejecutar reintentos y actualizar el diccionario con nuevos reintentos si los hay
            new_retries = capture_all_cameras(retries_now, max_workers)
            
            # Eliminar las cámaras procesadas de pending_retries
            for cam_id in retries_now:
                pending_retries.pop(cam_id, None)
                
            # Agregar nuevos reintentos al diccionario
            pending_retries.update(new_retries)
        
        # Verificar si es hora de captura regular
        if current_time >= next_regular_capture:
            logger.info(f"Ejecutando captura regular programada...")
            # Ejecutar captura normal para todas las cámaras excepto las que tienen reintento pendiente
            regular_cameras = [cam_id for cam_id in camera_ids if cam_id not in pending_retries]
            new_retries = capture_all_cameras(regular_cameras, max_workers)
            
            # Agregar nuevos reintentos al diccionario
            pending_retries.update(new_retries)
            
            # Programar la siguiente captura regular
            next_regular_capture = current_time + interval
            logger.info(f"Próxima captura regular programada para: {datetime.datetime.fromtimestamp(next_regular_capture).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calcular tiempo hasta el próximo evento (reintento o captura regular)
        next_events = [next_regular_capture]
        if pending_retries:
            next_events.extend(pending_retries.values())
        
        next_event = min(next_events)
        sleep_time = max(1, min(10, next_event - time.time()))  # Dormir hasta 10 segundos como máximo
        
        # Mostrar estado actual si hay reintentos pendientes
        if pending_retries:
            logger.info(f"Reintentos pendientes: {len(pending_retries)} cámaras")
            
        time.sleep(sleep_time)

def check_ffmpeg_installation():
    """
    Verifica si FFmpeg está instalado y disponible en el sistema.
    Proporciona información detallada sobre cómo solucionar problemas de instalación.
    
    Returns:
        bool: True si FFmpeg está instalado correctamente
    """
    ffmpeg_path = None
    
    # Comprobar si ffmpeg está en PATH
    if shutil.which("ffmpeg"):
        ffmpeg_path = shutil.which("ffmpeg")
        logger.info(f"FFmpeg encontrado en PATH: {ffmpeg_path}")
        
        # Verificar que se puede ejecutar
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            logger.info(f"Versión de FFmpeg: {result.stdout.splitlines()[0]}")
            return True
        except Exception as e:
            logger.error(f"FFmpeg encontrado pero no ejecutable: {str(e)}")
    else:
        logger.error("FFmpeg no encontrado en PATH")
    
    # Sugerencias específicas según el sistema operativo
    if os.name == 'nt':  # Windows
        logger.error("\n" + "="*60)
        logger.error("SOLUCIÓN PARA WINDOWS:")
        logger.error("1. Descargue FFmpeg desde https://ffmpeg.org/download.html o https://github.com/BtbN/FFmpeg-Builds/releases")
        logger.error("2. Extraiga el archivo ZIP en una carpeta (ej: C:\\ffmpeg)")
        logger.error("3. Agregue la ruta C:\\ffmpeg\\bin al PATH del sistema o copie ffmpeg.exe al directorio de trabajo")
        logger.error("4. Reinicie este script")
        logger.error("")
        logger.error("Alternativamente, puede especificar la ruta completa a ffmpeg.exe en el código")
        logger.error("="*60)
    else:  # Linux/Mac
        logger.error("\n" + "="*60)
        logger.error("SOLUCIÓN PARA LINUX/MAC:")
        logger.error("Para instalar FFmpeg en Ubuntu/Debian: sudo apt install ffmpeg")
        logger.error("Para instalar FFmpeg en Mac: brew install ffmpeg")
        logger.error("Para otras distribuciones, consulte la documentación correspondiente")
        logger.error("="*60)
    
    return False

def detailed_error_diagnose(e, camera_id):
    """
    Proporciona diagnóstico detallado para errores comunes
    
    Args:
        e (Exception): La excepción capturada
        camera_id (str): ID de la cámara
        
    Returns:
        str: Mensaje detallado de diagnóstico
    """
    error_str = str(e)
    
    # Extraer detalles completos del error
    exc_type, exc_obj, exc_tb = sys.exc_info()
    tb_details = traceback.format_exc()
    
    # Diagnosticar problemas comunes
    if "WinError 2" in error_str or "No such file or directory" in error_str:
        return (
            f"Error: No se encontró ejecutable de FFmpeg. "
            f"Detalles: {error_str}\n"
            f"Asegúrese de que FFmpeg está instalado y en el PATH del sistema.\n"
            f"Trace completo: {tb_details}"
        )
    elif "Connection refused" in error_str:
        return f"Error: Conexión rechazada al intentar conectar con la cámara {camera_id}. El servidor RTSP puede estar caído o inaccesible."
    elif "Timeout" in error_str:
        return f"Error: Timeout al conectar con la cámara {camera_id}. Verifique la conectividad y que la URL sea correcta."
    else:
        return f"Error: {error_str}\nTrace completo: {tb_details}"

if __name__ == "__main__":
    # Lista de IDs de cámaras
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
    # Añadir más IDs de cámaras según sea necesario
]
 # Verificar instalación de FFmpeg antes de comenzar
    logger.info("Verificando instalación de FFmpeg...")
    if not check_ffmpeg_installation():
        logger.error("No se puede continuar sin FFmpeg instalado correctamente.")
        sys.exit(1)
    
    # Establecer directorio base para capturas
    base_directory = "capturas_camaras"
    
    try:
        # Para una sola ejecución
        # capture_all_cameras(cameras)
        
        # Para monitoreo continuo (cada 1 minutos = 60 segundos)
        monitor_cameras(cameras, interval=60, run_forever=True, max_workers=4)
    except KeyboardInterrupt:
        logger.info("Programa detenido por el usuario (Ctrl+C)")
    except Exception as e:
        logger.critical(f"Error crítico en el programa principal: {str(e)}")
        logger.critical(traceback.format_exc())