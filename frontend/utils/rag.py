from elasticsearch import Elasticsearch
# ==================== PLANTILLA PARA ARCHIVOS UTILS ====================
# Copiar este contenido para cada archivo en utils/
# Ejemplo: utils/camaras.py

def obtener_contexto(query):
    """
    Función que retorna el contexto para el template HTML
    Debe ser implementada por cada módulo
    
    Returns:
        dict: Diccionario con datos para el template
    """
    results = index(query)
    return {
        'titulo': 'Elastic Search',
        'descripcion': 'Información en chUnks para el RAG',
        'datos': [],
        'estado': 'Activo',
        'query': query,
        'results': results
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



def index(query):
    # Conexión a Elasticsearch
    es = Elasticsearch("http://172.205.145.224:9200")

    INDEX_NAME = "calidad_aire"

    results = []

    if query:
        es_query = {
    "_source": ["title", "description", "page_number", "document_name","content"],
    "query": {
        "multi_match": {
            "query": query,
            "fields": ["title", "description", "content"]
        }
    }
}
        if query:
            print(f"🔍 Buscando: {query}")
            print(f"🧠 Query ES: {es_query}")
            response = es.search(index=INDEX_NAME, body=es_query)
            print(f"📥 Respuesta ES: {response}")
            results = response["hits"]["hits"]
            return results
