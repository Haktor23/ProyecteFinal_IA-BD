from flask import Flask, render_template, request
import requests
from model import tiemporeal
from pythonScripts import historico

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/imagenes')
def imagenes():
    return render_template('historico.html')

@app.route('/datos')
def datos():
    return render_template('datos.html')

@app.route('/ml')
def ml():
    return render_template('ml.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/tiempo-real', methods=['GET'])
def tiempo_real():
    camaras = tiemporeal.cameras
    camara = request.args.get('camara', camaras[0])  # Por defecto, primera cámara

    datos = None
    if camara:
        try:
            response = requests.get(f'http://localhost:5001/detect/{camara}', timeout=10)
            if response.status_code == 200:
                datos = response.json()
            else:
                datos = {'error': f"No se pudo obtener datos de la cámara {camara}"}
        except Exception as e:
            datos = {'error': str(e)}

    return render_template('tiempo_real.html', camaras=camaras, camara=camara, datos=datos)

@app.route('/repositorio')
def repositorio():
    return render_template('repositorio.html')

if __name__ == '__main__':
    app.run(debug=True)
