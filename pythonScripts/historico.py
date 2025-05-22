from flask import Blueprint, render_template, request, send_from_directory
from azure.storage.blob import ContainerClient
import os
import requests
from collections import defaultdict
from urllib.parse import urljoin

historico_bp = Blueprint('historico', __name__)

AZURE_CONTAINER_URL = "https://storageia2.blob.core.windows.net/data?sp=r&st=2025-05-22T14:27:39Z&se=2025-05-22T22:27:39Z&spr=https&sv=2024-11-04&sr=c&sig=psy%2BLBw3095%2FyCxCbIuMoypsvCZHh%2BKbD8W5W70A12s%3D"
LOCAL_CAPTURAS_PATH = "revisadas"
CAMARAS = ["camara_265", "camara_300"]

def script_historico():
    def imagenes():
        camara = request.args.get('camara', 'camara_265')
        fecha = request.args.get('fecha', '20250521')
        hora_filtro = request.args.get('hora')

        camara_path = os.path.join(LOCAL_CAPTURAS_PATH, camara, fecha)
        os.makedirs(camara_path, exist_ok=True)

        container = ContainerClient.from_container_url(AZURE_CONTAINER_URL)
        prefix = f"{camara}/anotada_{camara.split('_')[1]}_{fecha}"
        blobs = container.list_blobs(name_starts_with=prefix)

        imagenes_por_hora = defaultdict(list)
        base_url, sas_token = AZURE_CONTAINER_URL.split("?", 1)

        for blob in blobs:
            blob_name = blob.name.split("/")[-1]
            parts = blob_name.replace(".png", "").split("_")
            if len(parts) < 4:
                continue

            hora = parts[3][:2]
            local_path = os.path.join(camara_path, blob_name)

            if not os.path.exists(local_path):
                blob_url = urljoin(base_url + "/", blob.name) + "?" + sas_token
                r = requests.get(blob_url)
                if r.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                else:
                    print(f"[ERROR] No se pudo descargar {blob_name}")
                    continue

            if hora_filtro is None or hora == hora_filtro:
                imagenes_por_hora[hora].append({
                    "nombre": blob_name,
                    "hora": hora,
                    "ruta_local": f"/capturas/{camara}/{fecha}/{blob_name}"
                })

        horas_disponibles = sorted(imagenes_por_hora.keys(), key=lambda h: int(h))
        imagenes = imagenes_por_hora.get(hora_filtro, []) if hora_filtro else []

        return render_template(
            "imagenes.html", 
            camara=camara, 
            fecha=fecha, 
            horas=horas_disponibles, 
            hora_filtro=hora_filtro, 
            imagenes=imagenes, 
            camaras=CAMARAS
        )

    def servir_imagen_local(camara, fecha, filename):
        path = os.path.join(LOCAL_CAPTURAS_PATH, camara, fecha)
        return send_from_directory(path, filename)
