from flask import Flask, render_template, request, jsonify, redirect, url_for
import os

# Importar módulos de utils
from utils import index
from utils import camaras
from utils import realtime
from utils import tiempo
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

@app.route('/realtime')
def realtime_page():
    """Página de análisis en tiempo real"""
    try:
        contexto = realtime.obtener_contexto()
        return render_template('realtime.html', **contexto)
    except Exception as e:
        return render_template('realtime.html', error=str(e))

@app.route('/tiempo')
def tiempo_page():
    """Página de análisis temporal"""
    try:
        contexto = tiempo.obtener_contexto()
        return render_template('tiempo.html', **contexto)
    except Exception as e:
        return render_template('tiempo.html', error=str(e))

@app.route('/calidad')
def calidad_page():
    """Página de métricas de calidad"""
    try:
        contexto = calidad.obtener_contexto()
        return render_template('calidad.html', **contexto)
    except Exception as e:
        return render_template('calidad.html', error=str(e))

@app.route('/chatbot')
def chatbot_page():
    """Página del chatbot"""
    try:
        contexto = chatbot.obtener_contexto()
        return render_template('chatbot.html', **contexto)
    except Exception as e:
        return render_template('chatbot.html', error=str(e))

@app.route('/rag')
def rag_page():
    """Página del sistema RAG"""
    try:
        contexto = rag.obtener_contexto()
        return render_template('rag.html', **contexto)
    except Exception as e:
        return render_template('rag.html', error=str(e))

@app.route('/machinelearning')
def machinelearning_page():
    """Página de machine learning"""
    try:
        contexto = machinelearning.obtener_contexto()
        return render_template('machinelearning.html', **contexto)
    except Exception as e:
        return render_template('machinelearning.html', error=str(e))

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

@app.route('/api/tiempo', methods=['POST'])
def api_tiempo():
    """API para análisis temporal"""
    try:
        data = request.get_json()
        resultado = tiempo.procesar_datos(data)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calidad', methods=['POST'])
def api_calidad():
    """API para métricas de calidad"""
    try:
        data = request.get_json()
        resultado = calidad.procesar_datos(data)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        # Determinar qué módulo debe procesar el archivo
        modulo = request.form.get('modulo', 'general')
        
        if modulo == 'camaras':
            resultado = camaras.procesar_archivo(file)
        elif modulo == 'machinelearning':
            resultado = machinelearning.procesar_archivo(file)
        else:
            resultado = {'message': 'Archivo subido exitosamente'}
        
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== MANEJO DE ERRORES ====================

@app.errorhandler(404)
def not_found(error):
    """Página no encontrada"""
    return render_template('index.html', error='Página no encontrada'), 404

@app.errorhandler(500)
def internal_error(error):
    """Error interno del servidor"""
    return render_template('index.html', error='Error interno del servidor'), 500

# ==================== FUNCIÓN PRINCIPAL ====================

if __name__ == '__main__':
    # Verificar que existan los archivos de utils
    required_files = [
        'utils/__init__.py',
        'utils/index.py',
        'utils/camaras.py',
        'utils/realtime.py',
        'utils/tiempo.py',
        'utils/calidad.py',
        'utils/chatbot.py',
        'utils/rag.py',
        'utils/machinelearning.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  ADVERTENCIA: Faltan los siguientes archivos:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nCreando archivos básicos...")
        
        # Crear archivo __init__.py si no existe
        if not os.path.exists('utils/__init__.py'):
            os.makedirs('utils', exist_ok=True)
            with open('utils/__init__.py', 'w') as f:
                f.write('# Utils package\n')
    
    print("🚀 Iniciando aplicación Flask...")
    print("📁 Estructura del proyecto lista")
    print("🌐 La aplicación estará disponible en: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
