from flask import Flask, render_template
from pythonScripts import historico


from pythonScripts import tiemporeal

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

@app.route('/tiempo-real')
def tiempo_real():
    return render_template('tiempo_real.html')

@app.route('/repositorio')
def repositorio():
    return render_template('repositorio.html')

if __name__ == '__main__':
    app.run(debug=True)
