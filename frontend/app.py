from flask import Flask, render_template, request, jsonify, redirect, url_for
import threading
import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
# Agregar la carpeta ML al path para importar
ml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ML'))
sys.path.append(ml_path)
from predictor import generar_predicciones_3_dias, df_global
from predictor_ozono import predecir_ozono



# Importar módulos de utils
from utils import index
from utils import camaras
from utils import realtime
from utils import calidad
from utils import chatbot
from utils import rag
from utils import machinelearning
from utils import temperatura

app = Flask(__name__)

# Configuración básica
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Crear carpetas necesarias si no existen
os.makedirs('data/uploads', exist_ok=True)
os.makedirs('data/models', exist_ok=True)
os.makedirs('data/temp', exist_ok=True)

# ==================== RUTAS PRINCIPALES ====================

@app.route('/')
def home():
    """Página de inicio - redirige a index"""
    return redirect(url_for('index_page'))

@app.route('/index')
def index_page():
    """Página principal del sistema"""
    try:
        # Suponiendo que index.obtener_contexto() existe
        contexto = index.obtener_contexto()
        return render_template('index.html', **contexto)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/camaras')
def camaras_page():
    """Página de gestión de cámaras"""
    try:
        contexto = camaras.obtener_contexto()
        return render_template('camaras.html', **contexto)
    except Exception as e:
        return render_template('camaras.html', error=str(e))

@app.route('/procesar_datos', methods=['POST'])
def procesar_datos_route():
    def proceso_largo():
        camaras.procesar_datos()
    
    # Ejecutamos en hilo aparte para no bloquear el servidor
    thread = threading.Thread(target=proceso_largo)
    thread.start()
    
    return jsonify({'mensaje': 'Proceso iniciado'})

@app.route('/realtime')
def realtime_page():
    """Página de análisis en tiempo real"""
    try:
        contexto = realtime.obtener_contexto()
        return render_template('realtime.html', **contexto)
    except Exception as e:
        return render_template('realtime.html', error=str(e))


@app.route('/calidad')
def calidad_page():
    """Página de métricas de calidad"""
    try:
        # Esta función debe existir en utils/calidad.py y preparar el contexto inicial
        # para la plantilla calidad.html (ej. datos para selectores de gráficos).
        contexto = calidad.obtener_contexto()
        return render_template('calidad.html', **contexto)
    except Exception as e:
        # Es buena idea tener una plantilla de error más genérica o manejar esto
        # de forma más robusta, pero para el ejemplo está bien.
        return render_template('calidad.html', error=str(e), titulo="Error en Calidad", descripcion="No se pudo cargar la página de calidad.")

@app.route('/chatbot')
def chatbot_page():
    """Página del chatbot"""
    try:
        contexto = chatbot.obtener_contexto()
        return render_template('chatbot.html', **contexto)
    except Exception as e:
        return render_template('chatbot.html', error=str(e))

@app.route('/rag', methods=["GET", "POST"])
def rag_page():
    """Página del sistema RAG"""
    try:
        query="*"
        if request.method == "POST":
            query = request.form.get("query", "")
        contexto = rag.obtener_contexto(query)
        return render_template('rag.html', **contexto)
    except Exception as e:
        return render_template('rag.html', error=str(e))

@app.route('/machinelearning')
def machinelearning_page():
    contexto = machinelearning.obtener_contexto()
    return render_template('machinelearning.html', **contexto)

@app.route('/machinelearning/<id>', methods=['POST'])
def machinelearning_modelo(id):
    datos = request.form.to_dict()
    contexto = machinelearning.obtener_contexto(id=id, datos=datos)
    return render_template('machinelearning.html', **contexto)

@app.route('/temperaturas')
def temperaturas_page():
    """Página de datos meteorológicos y temperaturas"""
    try:
        # Esta función debe existir en utils/temperaturas.py y preparar el contexto inicial
        contexto = temperatura.obtener_contexto()
        return render_template('temperaturas.html', **contexto) # Asegúrate que 'temperaturas.html' existe
    except Exception as e:
        app.logger.error(f"Error en /temperaturas: {e}")
        # Considera una plantilla de error genérica o un manejo más robusto
        return render_template('temperaturas.html', error=str(e), titulo="Error en Meteorología", descripcion="No se pudo cargar la página de datos meteorológicos.")



# ==================== RUTAS API (POST) ====================

@app.route('/api/camaras', methods=['POST'])
def api_camaras():
    """API para procesamiento de cámaras"""
    try:
        data = request.get_json()
        resultado = camaras.procesar_datos(data)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime', methods=['POST'])
def api_realtime():
    """API para datos en tiempo real"""
    try:
        data = request.get_json()
        resultado = realtime.procesar_datos(data)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calidad', methods=['POST'])
def api_calidad():
    """API para métricas de calidad (genérica, si es necesaria)"""
    try:
        data = request.get_json()
        # Esta función podría ser para otras operaciones de calidad, no para DataTables.
        # Si solo tienes la de DataTables, esta ruta podría no ser necesaria o
        # tener un propósito diferente.
        resultado = calidad.procesar_datos(data) # Asumiendo que existe una función procesar_datos
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datos_calidad_aire', methods=['POST']) # Endpoint para DataTables
def api_datos_calidad_aire():
    """
    Endpoint específico para que DataTables obtenga los datos de calidad del aire.
    Utiliza request.form para acceder a los parámetros enviados por DataTables
    cuando se usa type: "POST" y no se envían datos JSON explícitamente.
    """
    try:
        # DataTables envía los parámetros como form data en una solicitud POST.
        response_data = calidad.procesar_para_datatable(request.form)
        return jsonify(response_data)
    except Exception as e:
        # Loggear el error es importante en producción
        print(f"Error en /api/datos_calidad_aire: {e}")
        return jsonify({'error': str(e), 'data': [], 'recordsFiltered': 0, 'recordsTotal': 0, 'draw': request.form.get('draw', 0)}), 500


@app.route('/api/chatbot', methods=['POST'])
def api_chatbot():
    """API para el chatbot"""
    try:
        data = request.get_json()
        return chatbot.procesar_mensaje(data)  # <-- Aquí directo
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rag', methods=['POST'])
def api_rag():
    """API para sistema RAG"""
    try:
        data = request.get_json()
        resultado = rag.procesar_consulta(data)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/machinelearning', methods=['POST'])
def api_machinelearning():
    """API para machine learning"""
    try:
        data = request.get_json()
        resultado = machinelearning.procesar_prediccion(data)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/datos_temperatura', methods=['POST'])
def api_datos_temperatura():
    """
    Endpoint específico para que DataTables obtenga los datos de temperatura.
    """
    try:
        # La plantilla temperaturas.html que generamos SÍ envía JSON:
        # contentType: "application/json" y data: JSON.stringify(d)
        # Por lo tanto, aquí usamos request.get_json()
        if request.is_json:
            request_data = request.get_json()
        else:
            # Fallback o error si no es JSON, aunque el frontend está configurado para enviar JSON.
            # DataTables puede enviar como form data si contentType no es 'application/json'.
            # Para ser robusto, podrías manejar ambos o forzar JSON.
            # Por ahora, asumimos que siempre será JSON según la plantilla.
            app.logger.warning("/api/datos_temperatura recibió datos no JSON, usando request.form como fallback.")
            request_data = request.form.to_dict() # Convertir ImmutableMultiDict a dict normal

        response_data = temperatura.procesar_para_datatable_temperatura(request_data)
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error en /api/datos_temperatura: {e}")
        # Intentar obtener 'draw' de la manera más segura posible
        draw_val = 0
        if request.is_json:
            try:
                draw_val = request.get_json().get('draw',0)
            except:
                pass # si get_json falla
        else:
            draw_val = request.form.get('draw', request.args.get('draw',0))

        return jsonify({'error': str(e), 'data': [], 'recordsFiltered': 0, 'recordsTotal': 0, 'draw': draw_val}), 500


@app.route('/api/ozono', methods=['POST'])
def ml_ozono():
    """
    Endpoint para generar predicciones de O3 
    
    Esperado en el body JSON:
    {
    'co': 0.15,
    'so2': 4.1,
    'pm10': 35,
    'pm25': 20,
   
    }
    """
    try:
        # Verificar que el módulo de ML esté disponible
        if predecir_ozono is None:
            return jsonify({
                'error': 'Módulo de predicción de ozono no disponible',
                'message': 'No se pudo cargar el sistema de predicciones'
            }), 500

        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Datos requeridos',
                'message': 'Debe enviar un JSON con co,so2,pm10,pm25'
            }), 400

        co = data.get('co')
        so2 = data.get('so2')
        pm10 = data.get('pm10')
        pm25 = data.get('pm25')

        if co is None or so2 is None or pm10 is None or pm25 is None:
        # Identificar qué parámetro falta (opcional, para un mensaje más específico)
            missing_params = []
            if co is None:
                missing_params.append('co')
            if so2 is None:
                missing_params.append('so2')
            if pm10 is None:
                missing_params.append('pm10')
            if pm25 is None:
                missing_params.append('pm25')

            error_message = f"Los siguientes parámetros son requeridos y no fueron proporcionados: {', '.join(missing_params)}."
            
            return jsonify({
                'error': 'datos_incompletos',
                'message': error_message
            }), 400
        
       
        valores_entrada = {
    'co': co,
    'so2': so2,
    'pm10': pm10,
    'pm25': pm25,
   
}
        # Llamada a tu función, que ahora devuelve un float de Python
        valor_predicho = predecir_ozono(valores_entrada)

        # jsonify ahora recibe un float estándar y no debería dar error
        return jsonify({"prediccion_ozono": valor_predicho})

    except FileNotFoundError as e: # Ejemplo de manejo de error específico
        app.logger.error(f"Error de modelo no encontrado: {e}")
        return jsonify({"error": str(e)}), 500
    except ValueError as e: # Ejemplo de manejo de error específico
        app.logger.error(f"Error de valor durante la predicción: {e}")
        return jsonify({"error": f"Error en los datos de entrada o configuración del modelo: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Error inesperado en /api/ozono: {e}", exc_info=True) # exc_info=True para loggear el traceback
        return jsonify({"error": "Ocurrió un error interno en el servidor."}), 500



@app.route('/api/prevision_temporal', methods=['POST'])
def prevision_temporal():
    """
    Endpoint para generar predicciones de O3 para los próximos 3 días
    
    Esperado en el body JSON:
    {
        "object_id": 12,
        "fecha_inicio": "2025-04-25 00:00:00" (opcional)
    }
    """
    try:
        # Verificar que el módulo de ML esté disponible
        if generar_predicciones_3_dias is None or df_global is None:
            return jsonify({
                'error': 'Módulo de predicción no disponible',
                'message': 'No se pudo cargar el sistema de predicciones'
            }), 500

        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Datos requeridos',
                'message': 'Debe enviar un JSON con object_id'
            }), 400

        object_id = data.get('object_id')
        fecha_inicio = data.get('fecha_inicio')

        if object_id is None:
            return jsonify({
                'error': 'object_id requerido',
                'message': 'Debe especificar el object_id para la predicción'
            }), 400

        # Si no se proporciona fecha_inicio, usar la última fecha disponible + 1 hora
        if not fecha_inicio:
            try:
                ultima_fecha = df_global[df_global['objectId'] == object_id]['Fecha'].max()
                if pd.isna(ultima_fecha):
                    # Si no hay datos para ese object_id, usar fecha actual
                    fecha_inicio = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                else:
                    fecha_inicio = (ultima_fecha + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                # Fallback a fecha actual
                fecha_inicio = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Generar predicciones
        print(f"Generando predicciones para ObjectId: {object_id}, Fecha inicio: {fecha_inicio}")
        
        json_predicciones = generar_predicciones_3_dias(
            object_id=object_id,
            fecha_inicio_str=fecha_inicio,
            historical_df=df_global
        )
        
        if json_predicciones is None:
            return jsonify({
                'error': 'Error en predicción',
                'message': f'No se pudieron generar predicciones para ObjectId {object_id}'
            }), 400

        # Convertir el JSON string a dict para enviarlo como respuesta
        predicciones_data = json.loads(json_predicciones)
        
        # Preparar respuesta con metadatos adicionales
        response_data = {
            'success': True,
            'object_id': object_id,
            'fecha_inicio': fecha_inicio,
            'total_predicciones': len(predicciones_data),
            'predicciones': predicciones_data,
            'metadata': {
                'periodo': '3 días',
                'frecuencia': 'horaria',
                'variable': 'O3 (µg/m³)',
                'generado_en': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Error en prevision_temporal: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

@app.route('/api/object_ids_disponibles', methods=['GET'])
def get_object_ids_disponibles():
    """
    Endpoint para obtener los object_ids que tienen modelos disponibles
    """
    try:
        object_ids_disponibles = []
        
        # Buscar archivos de modelo en la carpeta ML
        ml_path = os.path.join(os.path.dirname(__file__), 'ML')
        
        for filename in os.listdir(ml_path):
            if filename.startswith("modelo_") and filename.endswith(".pkl"):
                try:
                    # Extraer object_id del nombre del archivo
                    object_id = int(filename.split('_')[1])
                    
                    # Verificar que también existe el escalador
                    escalador_file = f"escalador_{object_id}.pkl"
                    if os.path.exists(os.path.join(ml_path, escalador_file)):
                        object_ids_disponibles.append(object_id)
                        
                except (IndexError, ValueError):
                    continue
        
        return jsonify({
            'success': True,
            'object_ids': sorted(object_ids_disponibles),
            'total': len(object_ids_disponibles)
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Error obteniendo object_ids',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servicio"""
    status = {
        'status': 'OK',
        'ml_module_loaded': generar_predicciones_3_dias is not None,
        'data_loaded': df_global is not None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if df_global is not None:
        status['total_records'] = len(df_global)
        status['unique_object_ids'] = df_global['objectId'].nunique()
        status['date_range'] = {
            'min': df_global['Fecha'].min().strftime('%Y-%m-%d %H:%M:%S'),
            'max': df_global['Fecha'].max().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    return jsonify(status)


# ==================== RUTAS DE ARCHIVOS ====================

@app.route('/upload', methods=['POST'])
def upload_file():
    """Subida de archivos general"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        modulo = request.form.get('modulo', 'general')
        
        if modulo == 'camaras':
            resultado = camaras.procesar_archivo(file)
        elif modulo == 'machinelearning':
            resultado = machinelearning.procesar_archivo(file)
        else:
            # Guardar el archivo de forma genérica o procesarlo
            filename = file.filename # Asegúrate de sanitizar el nombre del archivo
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            resultado = {'message': f'Archivo {filename} subido exitosamente a {modulo}'}
        
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== MANEJO DE ERRORES ====================

@app.errorhandler(404)
def not_found(error):
    """Página no encontrada"""
    return render_template('index.html', error='Página no encontrada'), 404 # O una plantilla 404.html dedicada

@app.errorhandler(500)
def internal_error(error):
    """Error interno del servidor"""
    return render_template('index.html', error='Error interno del servidor'), 500 # O una plantilla 500.html dedicada

# ==================== FUNCIÓN PRINCIPAL ====================

if __name__ == '__main__':
    # Verificar que existan los archivos de utils
    # (Esta verificación es útil para desarrollo)
    utils_path = 'utils'
    required_utils_modules = [
        '__init__.py', 'index.py', 'camaras.py', 'realtime.py', 
        'tiempo.py', 'calidad.py', 'chatbot.py', 'rag.py', 'machinelearning.py'
    ]
    
    # Crear carpeta utils si no existe
    if not os.path.exists(utils_path):
        os.makedirs(utils_path)
        print(f"INFO: Carpeta '{utils_path}' creada.")

    missing_files = []
    for file_name in required_utils_modules:
        file_path = os.path.join(utils_path, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            # Crear archivos básicos si faltan (opcional, para facilitar inicio)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if file_name == '__init__.py':
                        f.write(f"# Package marker for {utils_path}\n")
                    else:
                        f.write(f"# Placeholder for {file_name}\n")
                        f.write("def obtener_contexto(): return {}\n")
                        f.write("def procesar_datos(data): return {'message': 'Not implemented'}\n")
                print(f"INFO: Archivo '{file_path}' creado como placeholder.")
            except IOError as e:
                print(f"ERROR: No se pudo crear el archivo '{file_path}': {e}")

    if missing_files:
        print("\n⚠️  ADVERTENCIA: Algunos archivos de 'utils' podrían necesitar implementación completa.")
    
    print("🚀 Iniciando aplicación Flask...")
    print("📁 Estructura del proyecto lista (asegúrate que utils/calidad.py esté implementado).")
    print(f"📂 Ruta del CSV de calidad esperado: {os.path.join(os.getcwd(), 'data', 'CSV', 'HistoricoAire.csv')}")
    print("🌐 La aplicación estará disponible en: http://localhost:5000 (o la IP de tu máquina)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
