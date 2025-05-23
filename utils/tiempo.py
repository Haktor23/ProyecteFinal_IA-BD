# ==================== PLANTILLA PARA ARCHIVOS UTILS ====================
# Copiar este contenido para cada archivo en utils/
# Ejemplo: utils/camaras.py

def obtener_contexto():
    """
    Función que retorna el contexto para el template HTML
    Debe ser implementada por cada módulo
    
    Returns:
        dict: Diccionario con datos para el template
    """
    return {
        'titulo': 'Nombre de la Sección',
        'descripcion': 'Descripción de la funcionalidad',
        'datos': [],
        'estado': 'Activo'
    }

def procesar_datos(data=None):
    """
    Función principal para procesar datos del módulo
    Llamada desde las rutas API POST
    
    Args:
        data (dict): Datos recibidos del frontend
    
    Returns:
        dict: Resultado del procesamiento
    """
    try:
        # Implementar lógica específica del módulo
        resultado = {
            'status': 'success',
            'message': 'Datos procesados correctamente',
            'data': data if data else {}
        }
        return resultado
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error al procesar datos: {str(e)}'
        }

def procesar_archivo(file):
    """
    Función para procesar archivos subidos (opcional)
    Solo implementar si el módulo necesita manejar archivos
    
    Args:
        file: Archivo subido desde el frontend
    
    Returns:
        dict: Resultado del procesamiento del archivo
    """
    import os
    from werkzeug.utils import secure_filename
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join('data/uploads', filename)
        file.save(filepath)
        
        # Implementar lógica específica para el archivo
        resultado = {
            'status': 'success',
            'message': f'Archivo {filename} procesado correctamente',
            'filepath': filepath
        }
        return resultado
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error al procesar archivo: {str(e)}'
        }

# ==================== FUNCIONES ESPECÍFICAS DEL MÓDULO ====================
# Agregar aquí las funciones específicas de cada módulo

def funcion_especifica():
    """Función específica del módulo - implementar según necesidades"""
    pass

# ==================== EJEMPLOS POR MÓDULO ====================

# EJEMPLO PARA utils/camaras.py:
def obtener_camaras_disponibles():
    """Retorna lista de cámaras disponibles"""
    return ['Cámara 1', 'Cámara 2', 'Cámara 3']

def procesar_imagen(imagen_path):
    """Procesa una imagen de cámara"""
    # Implementar procesamiento de imagen
    pass

# EJEMPLO PARA utils/chatbot.py:
def procesar_mensaje(data):
    """Procesa mensaje del chatbot"""
    mensaje = data.get('mensaje', '')
    respuesta = f"Respuesta automática a: {mensaje}"
    return {
        'status': 'success',
        'respuesta': respuesta
    }
    
