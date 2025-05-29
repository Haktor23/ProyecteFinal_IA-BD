import random

def obtener_contexto(id=None, datos=None):
    modelos = [
        {
            'id': 'modelo1',
            'nombre': 'Predicción de PM2.5',
            'campos': [{'nombre': 'temperatura', 'label': 'Temperatura (°C)'}, {'nombre': 'humedad', 'label': 'Humedad (%)'}]
        },
        {
            'id': 'modelo2',
            'nombre': 'Clasificación de Calidad del Aire',
            'campos': [{'nombre': 'pm10', 'label': 'PM10 (μg/m3)'}, {'nombre': 'no2', 'label': 'NO2 (ppb)'}]
        },
        {
            'id': 'modelo3',
            'nombre': 'Nivel de Ozono Estimado',
            'campos': [{'nombre': 'uv', 'label': 'Radiación UV'}, {'nombre': 'hora', 'label': 'Hora del Día'}]
        },
        {
            'id': 'modelo4',
            'nombre': 'Previsión de CO2 Ambiental',
            'campos': [{'nombre': 'trafico', 'label': 'Tráfico Vehicular'}, {'nombre': 'viento', 'label': 'Velocidad del Viento'}]
        },
        {
            'id': 'modelo5',
            'nombre': 'Índice de Riesgo Respiratorio',
            'campos': [{'nombre': 'pm25', 'label': 'PM2.5'}, {'nombre': 'edad', 'label': 'Edad del Individuo'}]
        }
    ]

    resultados = {m['id']: None for m in modelos}
    if id and datos:
        resultados[id] = procesar_simulacion(id, datos)

    return {
        'titulo': 'Modelos de Machine Learning',
        'descripcion': 'Simulaciones de modelos para predecir distintos aspectos de la calidad del aire.',
        'modelos': modelos,
        'resultados': resultados
    }

def procesar_simulacion(id, datos):
    if id == 'modelo1':
        return f"PM2.5 estimado: {round(random.uniform(10, 150), 2)} μg/m³"
    elif id == 'modelo2':
        return f"Calidad del Aire: {random.choice(['Buena', 'Moderada', 'Pobre'])}"
    elif id == 'modelo3':
        return f"Nivel de Ozono estimado: {round(random.uniform(20, 120), 2)} ppb"
    elif id == 'modelo4':
        return f"Nivel de CO2: {round(random.uniform(300, 600), 2)} ppm"
    elif id == 'modelo5':
        return f"Riesgo Respiratorio: {random.choice(['Bajo', 'Medio', 'Alto'])}"
    return "Modelo no encontrado"
