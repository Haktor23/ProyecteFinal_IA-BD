import re
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import generate_container_sas, AccountSasPermissions
from azure.storage.blob import AccountSasPermissions # Import the required permission class
from datetime import datetime, timedelta
import random
from flask import request


def obtener_contexto():
    try:
        df = pd.read_csv('./data/CSV/imagenes_revisadas.csv')

        camaras_disponibles = sorted(df['IdCamara'].unique().tolist())

        camara = request.args.get('camara', type=int)
        fecha = request.args.get('fecha', type=int)

        fechas_disponibles = []
        if camara:
            df_filtrado = df[df['IdCamara'] == camara]
            fechas_disponibles = sorted(df_filtrado['Fecha'].unique().tolist())
            if fecha:
                df_filtrado = df_filtrado[df_filtrado['Fecha'] == fecha]
        else:
            df_filtrado = df

        if df_filtrado.empty:
            urls = df.sample(10)['URL'].tolist()
        else:
            urls = df_filtrado['URL'].tolist()
            if len(urls) > 1:
                urls = urls[:50]

        return {
            'titulo': 'Galería de Cámaras',
            'descripcion': 'Filtra por cámara y fecha, o explora imágenes aleatorias.',
            'imagenes': urls,
            'estado': 'Activo',
            'camaras': camaras_disponibles,
            'fechas': fechas_disponibles,
            'camara_seleccionada': camara,
            'fecha_seleccionada': fecha
        }

    except Exception as e:
        return {
            'titulo': 'Error al cargar imágenes',
            'descripcion': str(e),
            'imagenes': [],
            'estado': 'Error',
            'camaras': [],
            'fechas': []
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
        
    # Reemplaza con tu cadena de conexión real
    connection_string = "DefaultEndpointsProtocol=https;AccountName=storageia2;AccountKey=szSngv9jjWf3dcKZcQYFAAwRdf/Lojl50Q9vR2DRl31Bj6EVaSd0Jip7HyFEfA+K7CMARv2oJQNL+ASta10n/g==;EndpointSuffix=core.windows.net"

    # Nombre del contenedor
    container_name = "data"

    # Crear cliente
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # Extraer datos de la cuenta
    account_name = blob_service_client.account_name
    account_key = blob_service_client.credential.account_key

    # Generar SAS token válido por 7 días
    # Use the imported generate_blob_sas function
    sas_token = generate_container_sas(
        account_name=account_name,
        account_key=account_key,
        container_name=container_name,
        permission=AccountSasPermissions(read=True, list=True),
        expiry=datetime.utcnow() + timedelta(days=7)
    )

    print("✅ SAS token generado correctamente.")

    # ==== PROCESAR BLOBS ====

    # Base URL para formar los enlaces
    base_url = f"https://{account_name}.blob.core.windows.net/{container_name}"

    # Regex para extraer info del nombre del archivo
    pattern = r"revisadas/captura_(\d+)_(\d{8})_(\d{6})\.png"

    # Lista para almacenar datos
    data = []

    # Recorrer blobs en la carpeta 'revisadas/'
    for blob in container_client.list_blobs(name_starts_with="revisadas/"):
        match = re.match(pattern, blob.name)
        if match:
            id_camara, fecha, hora = match.groups()
            url = f"{base_url}/{blob.name}?{sas_token}"
            data.append({
                "IdCamara": id_camara,
                "Fecha": fecha,
                "Hora": hora,
                "URL": url
            })

    # Crear DataFrame y exportar CSV
    df = pd.DataFrame(data)
    df.to_csv("../data/CSV/imagenes_revisadas.csv", index=False)
