
# EJEMPLO PARA utils/realtime.py:
def obtener_datos_tiempo_real():
    """Obtiene datos en tiempo real"""
    import datetime
    return {
        'timestamp': datetime.datetime.now().isoformat(),
        'datos': 'Datos simulados'
    }
