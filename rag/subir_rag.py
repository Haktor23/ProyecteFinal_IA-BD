import os
import json
import requests
from elasticsearch import Elasticsearch
from datetime import datetime

# Configuraciones
CHUNKS_DIR = "chunks_json"
INDEX_NAME = "calidad_aire"
LOG_FILE = "log_resultados.txt"
CHECKPOINT_FILE = "procesados.txt"
API_URL = "http://127.0.0.1:1234/v1/chat/completions"

# Conexión a Elasticsearch
es = Elasticsearch("http://172.205.145.224:9200")

# Crear índice si no existe
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME)

# Cargar archivos ya procesados
procesados = set()
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        procesados = set(line.strip() for line in f.readlines())

# Clasificación con Mistral
def es_util_con_mistral(texto, nombre_archivo):
    prompt = f"Texto:\n{texto}\n\n¿Es útil para entrenar un chatbot sobre la calidad del aire?"
    try:
        data = {
            "model": "mistral-7b-instruct-v0.3",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }

        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            salida = response.json()
            respuesta_modelo = salida.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
            primera_linea = respuesta_modelo.splitlines()[0]

            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"{nombre_archivo}: {primera_linea}\n")

            return "útil" in primera_linea
        else:
            print(f"⚠️ Error HTTP {response.status_code} al clasificar {nombre_archivo}")
            print(f"Detalles: {response.text}")
            return False

    except Exception as e:
        print(f"⚠️ Excepción al procesar {nombre_archivo}: {e}")
        return False

# Procesar chunks
for nombre_archivo in os.listdir(CHUNKS_DIR):
    if not nombre_archivo.endswith(".json") or nombre_archivo in procesados:
        continue

    ruta_archivo = os.path.join(CHUNKS_DIR, nombre_archivo)
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            chunk = json.load(f)
    except Exception as e:
        print(f"⚠️ Error leyendo {nombre_archivo}: {e}")
        continue

    texto = (
        chunk.get("content") or
        chunk.get("texto") or
        chunk.get("contenido") or
        chunk.get("text") or
        ""
    ).strip()

    if not texto:
        print(f"⚠️ Sin texto útil en {nombre_archivo}. Claves: {list(chunk.keys())}")
        continue

    print(f"📂 Analizando: {nombre_archivo}")

    if es_util_con_mistral(texto, nombre_archivo):
        try:
            doc_id = chunk.get("id", nombre_archivo.replace(".json", ""))
            es.index(index=INDEX_NAME, id=doc_id, body=chunk)
            print(f"✅ Subido: {nombre_archivo}")

            with open(CHECKPOINT_FILE, "a", encoding="utf-8") as cp:
                cp.write(nombre_archivo + "\n")

        except Exception as e:
            print(f"❌ Error subiendo {nombre_archivo} a Elasticsearch: {e}")
    else:
        print(f"❌ No útil: {nombre_archivo}")
        with open(CHECKPOINT_FILE, "a", encoding="utf-8") as cp:
            cp.write(nombre_archivo + "\n")

print("🏁 Finalizado.")
