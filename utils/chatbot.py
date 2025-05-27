# ==================== PLANTILLA PARA ARCHIVOS UTILS ====================
# Copiar este contenido para cada archivo en utils/
# Ejemplo: utils/camaras.py

import os
import json
import traceback
import re
import unicodedata
import requests
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory, render_template, Response, stream_with_context
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from datetime import datetime, timedelta, date

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

# ==================== PLANTILLA PARA ARCHIVOS UTILS ====================



# --- Configuración ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://10.90.6.251:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:0.6b")
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv("all-minilm:l6-v2")

ES_HOST = os.getenv("ES_HOST", "http://172.205.145.224:9200")
ES_INDEX = os.getenv("ES_INDEX", "contaminacion")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
PDF_DIR_CONTAINER = os.getenv("PDF_DIR_CONTAINER", "/app/pdfs_data")
ES_VECTOR_FIELD = os.getenv("embedding", "embedding")

# --- Conexión a Elasticsearch ---
es_client = None
try:
    es_auth = (ES_USER, ES_PASSWORD) if ES_USER and ES_PASSWORD else None
    es_client = Elasticsearch(ES_HOST, basic_auth=es_auth, request_timeout=10)
    if not es_client.ping():
        raise ConnectionError("Fallo el ping a Elasticsearch")
    print(f"Conectado a ES en {ES_HOST} (Índice: {ES_INDEX}).")
except Exception as e:
    print(f"ADVERTENCIA: Error ES: {e}. La funcionalidad RAG no estará disponible.")
    es_client = None

# --- Utilidades ---
def obtener_contexto():
    return {
        'titulo': 'Nombre de la Sección',
        'descripcion': 'Descripción de la funcionalidad',
        'datos': [],
        'estado': 'Activo'
    }

def get_current_date_str(format_str='%Y-%m-%d'):
    return datetime.today().strftime(format_str)

def get_yesterdays_date_str(format_str='%Y-%m-%d'):
    return (datetime.today() - timedelta(days=1)).strftime(format_str)

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

# --- Tools Registration (para LLM) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_temperature_data",
            "description": "Obtiene la temperatura y descripción del clima para una fecha específica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_query": {"type": "string", "description": "Fecha ('ayer', 'hoy' o 'YYYY-MM-DD')"},
                    "station_name": {"type": "string", "description": "Nombre de la estación"},
                },
                "required": ["date_query", "station_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "Busca información en Elasticsearch sobre temas técnicos.",
            "parameters": {
                "type": "object",
                "properties": {"search_query": {"type": "string", "description": "Términos o pregunta."}},
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality_tool_impl",
            "description": "Obtiene la calidad del aire para una estación.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nombre_estacion": {"type": "string", "description": "Nombre de la estación (ej. 'Patraix')"},
                },
                "required": ["nombre_estacion"]
            },
        },
    }
]

# --- Función: obtener embedding con Ollama ---
def get_embedding(text_to_embed: str, embedding_model_name: str):
    if not OLLAMA_HOST or not embedding_model_name:
        print("OLLAMA_HOST o modelo de embeddings no configurados.")
        return None
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": embedding_model_name, "prompt": text_to_embed},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        embedding_data = response.json()
        return embedding_data.get("embedding")
    except Exception as e:
        print(f"Error al obtener embedding: {e}")
        return None

# --- Tool: temperatura desde CSV o API ---
def get_air_quality_tool_impl(nombre_estacion: str, date_query: str) -> dict:
    csv_path = "data/CSV/HistoricoCalidadAire.csv"
    today = datetime.utcnow().date()

    # Determinar fuente de datos y fecha
    if date_query.lower() == "ayer":
        fecha_obj = today - timedelta(days=1)
        source = "csv"
    elif date_query.lower() == "hoy":
        fecha_obj = today
        source = "api"
    else:
        try:
            fecha_obj = datetime.strptime(date_query, "%Y-%m-%d").date()
            if fecha_obj >= today:
                return {"error": "La fecha debe ser anterior a hoy. Use 'hoy' para datos actuales."}
            source = "csv"
        except ValueError:
            return {"error": "Formato de fecha inválido. Use 'YYYY-MM-DD', 'ayer' o 'hoy'."}

    nombre_normalizado = normalize_text(nombre_estacion)

    if source == "api":
        try:
            resp = requests.get("https://valencia.opendatasoft.com/api/explore/v2.1/catalog/datasets/estacions-contaminacio-atmosferiques-estaciones-contaminacion-atmosfericas/records?limit=100")
            estaciones = resp.json().get("results", [])
            for est in estaciones:
                if nombre_normalizado in normalize_text(est.get("nombre", "")):
                    return {
                        "fecha": str(fecha_obj),
                        "estacion": est["nombre"],
                        "calidad": est.get("calidad_am", "Desconocida"),
                        "detalles": {
                            "SO2": est.get("so2"),
                            "NO2": est.get("no2"),
                            "O3": est.get("o3"),
                            "CO": est.get("co"),
                            "PM10": est.get("pm10"),
                            "PM2.5": est.get("pm25"),
                        },
                        "descripcion": f"Datos actuales de calidad del aire para '{est['nombre']}'"
                    }
            return {"error": f"No se encontró información para '{nombre_estacion}' hoy."}
        except Exception as e:
            return {"error": f"Error en API: {e}"}

    # Si es histórico, buscar en CSV
    try:
        df = pd.read_csv(csv_path, sep=';')
        df['fecha_carg_date'] = pd.to_datetime(df['fecha_carg']).dt.date
        datos = df[(df['fecha_carg_date'] == fecha_obj) & df['nombre'].str.contains(nombre_estacion, case=False, na=False)]
        if datos.empty:
            return {"error": f"No hay datos para '{nombre_estacion}' en {fecha_obj}"}
        registro = datos.iloc[-1]
        return {
            "fecha": str(fecha_obj),
            "estacion": registro["nombre"],
            "calidad": registro.get("calidad_am", "Desconocida"),
            "detalles": {
                "SO2": registro.get("so2"),
                "NO2": registro.get("no2"),
                "O3": registro.get("o3"),
                "CO": registro.get("co"),
                "PM10": registro.get("pm10"),
                "PM2.5": registro.get("pm25"),
            },
            "descripcion": f"Datos históricos de calidad del aire para '{registro['nombre']}' en {fecha_obj}"
        }
    except Exception as e:
        return {"error": f"Error CSV: {e}"}


# --- Tool: calidad del aire ---
def get_air_quality_tool_impl(nombre_estacion: str) -> dict:
    nombre_normalizado = normalize_text(nombre_estacion)
    try:
        resp = requests.get("https://valencia.opendatasoft.com/api/explore/v2.1/catalog/datasets/estacions-contaminacio-atmosferiques-estaciones-contaminacion-atmosfericas/records?limit=100")
        estaciones = resp.json().get("results", [])
        for est in estaciones:
            if nombre_normalizado in normalize_text(est.get("nombre", "")):
                return {
                    "estacion": est["nombre"],
                    "calidad": est.get("calidad_am", "Desconocida"),
                    "fecha": est.get("fecha_carg"),
                    "detalles": {
                        "SO2": est.get("so2"),
                        "NO2": est.get("no2"),
                        "O3": est.get("o3"),
                        "CO": est.get("co"),
                        "PM10": est.get("pm10"),
                        "PM2.5": est.get("pm25"),
                    }
                }
        return {"error": f"No se encontró información para '{nombre_estacion}'"}
    except Exception as e:
        return {"error": f"Error al consultar calidad del aire: {e}"}

# --- Tool: búsqueda en Elasticsearch (RAG) ---
def query_elasticsearch_tool_impl(search_query: str):
    if not es_client:
        return json.dumps({"error": "Cliente Elasticsearch no disponible."})

    resultados = []
    seen = set()

    try:
        keyword_hits = es_client.search(
            index=ES_INDEX,
            query={"match": {"content": {"query": search_query}}},
            size=7
        )['hits']['hits']
        for hit in keyword_hits:
            if hit['_id'] not in seen:
                resultados.append({
                    "text": hit['_source'].get("content", "")[:1000],
                    "title": hit['_source'].get("title", "N/A"),
                    "filename": hit['_source'].get("document_name", "N/A"),
                    "score": hit.get("_score", 0),
                    "match_type": "keyword"
                })
                seen.add(hit['_id'])
    except Exception as e:
        print(f"Error búsqueda BM25: {e}")

    # Vector search
    if OLLAMA_EMBEDDING_MODEL_NAME:
        embedding = get_embedding(search_query, OLLAMA_EMBEDDING_MODEL_NAME)
        if embedding:
            try:
                vector_hits = es_client.search(
                    index=ES_INDEX,
                    knn={"field": ES_VECTOR_FIELD, "query_vector": embedding, "k": 5, "num_candidates": 20},
                    size=5
                )['hits']['hits']
                for hit in vector_hits:
                    if hit['_id'] not in seen:
                        resultados.append({
                            "text": hit['_source'].get("content", "")[:1000],
                            "title": hit['_source'].get("title", "N/A"),
                            "filename": hit['_source'].get("document_name", "N/A"),
                            "score": hit.get("_score", 0),
                            "match_type": "vector"
                        })
                        seen.add(hit['_id'])
            except Exception as e:
                print(f"Error búsqueda vectorial: {e}")

    if not resultados:
        return json.dumps({"message": "No se encontraron documentos relevantes."})
    
    return json.dumps({"retrieved_chunks_metadata": resultados[:5]})
def procesar_mensaje(data):
    prompt_raw = data.get("prompt")
    if not prompt_raw or not isinstance(prompt_raw, str):
        return jsonify({"type": "error", "data": "Prompt no proporcionado"}), 400

    user_prompt = prompt_raw.lower()

    def generate_chat_responses():
        messages_for_ollama = [{"role": "user", "content": user_prompt}]
        yield json.dumps({"type": "reasoning_step", "data": messages_for_ollama[0]}) + "\n"

        max_tool_calls = 5
        tool_calls_count = 0
        rag_sources_for_client = None

        try:
            while tool_calls_count < max_tool_calls:
                print(f"\n--- Enviando a Ollama (Turno {tool_calls_count + 1}) ---")
                tool_calls_detected_this_turn = []
                assistant_response_content_buffer = ""
                current_think_buffer = ""
                in_think_block = False
                final_assistant_message_object_this_turn = None

                ollama_response_stream = requests.post(
                    f"{OLLAMA_HOST}/api/chat",
                    json={"model": MODEL_NAME, "messages": messages_for_ollama, "tools": tools, "stream": True},
                    headers={"Content-Type": "application/json"},
                    stream=True,
                    timeout=300
                )
                ollama_response_stream.raise_for_status()

                for line in ollama_response_stream.iter_lines():
                    if line:
                        try:
                            chunk_str = line.decode('utf-8')
                            ollama_chunk = json.loads(chunk_str)
                            content_piece = ollama_chunk.get("message", {}).get("content", "")
                            temp_content_piece = content_piece

                            while temp_content_piece:
                                if not in_think_block:
                                    think_start = temp_content_piece.find("<think>")
                                    if think_start != -1:
                                        if think_start > 0:
                                            token_to_yield = temp_content_piece[:think_start]
                                            assistant_response_content_buffer += token_to_yield
                                            yield json.dumps({"type": "token", "data": token_to_yield}) + "\n"
                                        in_think_block = True
                                        temp_content_piece = temp_content_piece[think_start + len("<think>"):]
                                    else:
                                        assistant_response_content_buffer += temp_content_piece
                                        yield json.dumps({"type": "token", "data": temp_content_piece}) + "\n"
                                        temp_content_piece = ""
                                else:
                                    think_end = temp_content_piece.find("</think>")
                                    if think_end != -1:
                                        current_think_buffer += temp_content_piece[:think_end]
                                        current_think_buffer = ""
                                        in_think_block = False
                                        temp_content_piece = temp_content_piece[think_end + len("</think>"):]
                                    else:
                                        current_think_buffer += temp_content_piece
                                        temp_content_piece = ""

                            if ollama_chunk.get("message", {}).get("tool_calls"):
                                tool_calls_detected_this_turn.extend(ollama_chunk["message"]["tool_calls"])

                            if ollama_chunk.get("done"):
                                final_assistant_message_object_this_turn = ollama_chunk.get("message")
                                if tool_calls_detected_this_turn and final_assistant_message_object_this_turn:
                                    final_assistant_message_object_this_turn["tool_calls"] = tool_calls_detected_this_turn
                                elif tool_calls_detected_this_turn and not final_assistant_message_object_this_turn:
                                    final_assistant_message_object_this_turn = {"role": "assistant", "tool_calls": tool_calls_detected_this_turn}
                                break
                        except json.JSONDecodeError:
                            print(f"Error JSON Ollama: {line.decode('utf-8', errors='ignore')}")
                        except Exception as e_chunk:
                            print(f"Error chunk Ollama: {type(e_chunk).__name__} {e_chunk}")
                            traceback.print_exc()

                if final_assistant_message_object_this_turn:
                    messages_for_ollama.append(final_assistant_message_object_this_turn)
                    yield json.dumps({"type": "reasoning_step", "data": final_assistant_message_object_this_turn}) + "\n"

                if tool_calls_detected_this_turn:
                    yield json.dumps({"type": "tool_interaction", "stage": "request", "data": tool_calls_detected_this_turn}) + "\n"
                    tool_response_messages_for_ollama = []

                    for tool_call in tool_calls_detected_this_turn:
                        tool_name = tool_call["function"]["name"]
                        tool_arguments_from_ollama = tool_call["function"]["arguments"]
                        tool_args = {}
                        tool_execution_error = False

                        print(f"DEBUG: Args para '{tool_name}': {tool_arguments_from_ollama} (Tipo: {type(tool_arguments_from_ollama)})")

                        if isinstance(tool_arguments_from_ollama, str):
                            try:
                                tool_args = json.loads(tool_arguments_from_ollama)
                            except json.JSONDecodeError as e_json:
                                print(f"Error JSON args str para {tool_name}: {e_json}")
                                tool_executed_result_str = json.dumps({
                                    "error": f"Args JSON malformados para '{tool_name}'.",
                                    "details": str(e_json)
                                })
                                tool_execution_error = True
                        elif isinstance(tool_arguments_from_ollama, dict):
                            tool_args = tool_arguments_from_ollama
                        else:
                            print(f"Error tipo args para {tool_name}: {type(tool_arguments_from_ollama)}")
                            tool_executed_result_str = json.dumps({
                                "error": f"Tipo arg inesperado para '{tool_name}'."
                            })
                            tool_execution_error = True

                        if not tool_execution_error:
                            if tool_name == "get_temperature_data":
                                tool_result_object = get_temperature_data_tool_impl(**tool_args)
                                tool_executed_result_str = json.dumps(tool_result_object)
                            elif tool_name == "query_knowledge_base":
                                search_query = tool_args.get("search_query", "")
                                tool_executed_result_str = query_elasticsearch_tool_impl(search_query)
                                try:
                                    rag_tool_data = json.loads(tool_executed_result_str)
                                    if "retrieved_chunks_metadata" in rag_tool_data:
                                        rag_sources_for_client = rag_tool_data["retrieved_chunks_metadata"]
                                except Exception:
                                    pass
                            elif tool_name == "get_air_quality_tool_impl":
                                tool_result_object = get_air_quality_tool_impl(**tool_args)
                                tool_executed_result_str = json.dumps(tool_result_object)
                            else:
                                tool_executed_result_str = json.dumps({
                                    "error": f"Herramienta '{tool_name}' no reconocida."
                                })

                        tool_call_id_from_model = tool_call.get("id")
                        if not tool_call_id_from_model:
                            tool_call_id_from_model = f"gen_tool_id_{tool_calls_count}_{len(tool_response_messages_for_ollama)}"

                        tool_response_msg_part = {
                            "role": "tool",
                            "content": tool_executed_result_str,
                            "tool_call_id": tool_call_id_from_model
                        }

                        tool_response_messages_for_ollama.append(tool_response_msg_part)

                        yield json.dumps({"type": "reasoning_step", "data": tool_response_msg_part}) + "\n"
                        yield json.dumps({
                            "type": "tool_interaction",
                            "stage": "response",
                            "data": {
                                "name": tool_name,
                                "id": tool_call_id_from_model,
                                "result_preview": tool_executed_result_str[:200] + ("..." if len(tool_executed_result_str) > 200 else "")
                            }
                        }) + "\n"

                    messages_for_ollama.extend(tool_response_messages_for_ollama)
                    tool_calls_count += 1
                else:
                    if assistant_response_content_buffer.strip() or (
                        final_assistant_message_object_this_turn and final_assistant_message_object_this_turn.get("content", "").strip()
                    ):
                        if rag_sources_for_client:
                            yield json.dumps({"type": "rag_sources", "data": rag_sources_for_client}) + "\n"
                    else:
                        print("INFO: Modelo no generó contenido ni llamó herramientas.")

                    yield json.dumps({"type": "final_history", "data": messages_for_ollama}) + "\n"
                    yield json.dumps({"type": "stream_end"}) + "\n"
                    return

            yield json.dumps({"type": "error", "data": "Límite de llamadas a herramientas."}) + "\n"
            yield json.dumps({"type": "final_history", "data": messages_for_ollama}) + "\n"
            yield json.dumps({"type": "stream_end"}) + "\n"

        except requests.exceptions.HTTPError as http_err:
            error_text = http_err.response.text
            print(f"Error HTTP Ollama: {http_err.response.status_code} - {error_text}")
            yield json.dumps({
                "type": "error",
                "data": f"Error Ollama: {http_err.response.status_code}. Detalles: {error_text[:200]}"
            }) + "\n"

        except requests.exceptions.RequestException as req_err:
            print(f"Error conexión Ollama: {req_err}")
            yield json.dumps({
                "type": "error",
                "data": f"Error red contactando Ollama: {str(req_err)}"
            }) + "\n"

        except Exception as e_gen:
            print(f"Error inesperado en generador: {type(e_gen).__name__} - {e_gen}")
            traceback.print_exc()
            yield json.dumps({
                "type": "error",
                "data": f"Error inesperado servidor: {str(e_gen)}"
            }) + "\n"

        finally:
            print("generate_chat_responses finalizado.")
            # Nota: ya se envió `stream_end` en el flujo normal

    return Response(stream_with_context(generate_chat_responses()), mimetype="application/x-ndjson")
