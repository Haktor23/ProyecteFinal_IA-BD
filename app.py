from flask import Flask, render_template, request, jsonify, redirect, url_for
import threading
import os

# Importar módulos de utils
from utils import index
from utils import camaras
from utils import realtime
from utils import calidad
from utils import chatbot
from utils import rag
from utils import machinelearning

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
        resultado = chatbot.procesar_mensaje(data)
        return jsonify(resultado)
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
