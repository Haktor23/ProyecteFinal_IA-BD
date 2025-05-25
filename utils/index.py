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
        'titulo': 'Proyecto calidad del aire en Valencia',
        'descripcion': 'Trabajo de Big Data e IA para conocer los efectos del trafico en la calidad del aire',
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
