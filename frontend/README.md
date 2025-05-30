# ProyecteFinal_IA-BD: Sistema Inteligente de Monitorización Urbana

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3.x-black?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-lightgreen?style=for-the-badge&logo=pandas&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

## 📖 Prólogo del Proyecto

¡Bienvenido al repositorio de **ProyecteFinal_IA-BD**! Este proyecto es un sistema integral de monitorización urbana que combina la inteligencia artificial, el análisis de datos y la visualización interactiva para ofrecer una visión profunda de diversos aspectos de la ciudad. Desde la calidad del aire hasta la monitorización de cámaras de tráfico, pasando por análisis de temperatura y capacidades de chatbot, nuestro objetivo es proporcionar herramientas robustas para la gestión y comprensión del entorno urbano.

Hemos desarrollado una aplicación web dinámica que permite a los usuarios explorar datos históricos y en tiempo real, interactuar con modelos de Machine Learning y obtener respuestas a sus preguntas a través de un sistema RAG (Retrieval-Augmented Generation) potenciado por IA. Este proyecto es la culminación de un esfuerzo por integrar tecnologías avanzadas para crear soluciones prácticas y accesibles para la toma de decisiones informada en un contexto urbano.

## ✨ Características Principales

* **Monitorización de Calidad del Aire:** Visualización y análisis de datos históricos de calidad del aire.
* **Monitorización de Cámaras de Tráfico:** Gestión y visualización de datos de cámaras de tráfico (posiblemente imágenes y metadatos).
* **Análisis de Temperatura:** Seguimiento y análisis de datos de temperatura a lo largo del tiempo.
* **Capacidades de Machine Learning:** Implementación de modelos para predicción o análisis de patrones urbanos.
* **Chatbot Inteligente:** Un asistente conversacional para responder preguntas relacionadas con los datos urbanos utilizando RAG.
* **Visualización de Datos Interactiva:** Interfaz web intuitiva para explorar y comprender los datos.
* **Gestión de Datos:** Procesamiento y almacenamiento de datos históricos y en tiempo real.

## 🛠️ Tecnologías Utilizadas

Este proyecto ha sido construido utilizando las siguientes tecnologías:

* **Backend:**
    * **Python 3.9+**: Lenguaje de programación principal.
    * **Flask**: Micro-framework web para la creación de la aplicación.
    * **Pandas**: Para la manipulación y análisis de datos.
    * **NumPy**: Para operaciones numéricas eficientes.
    * **Scikit-learn**: Para la implementación de modelos de Machine Learning.
    * **Otros paquetes de Python**: Ver `requirements.txt`.
* **Frontend:**
    * **HTML5**: Estructura de las páginas web.
    * **CSS3**: Estilos y diseño visual.
    * **JavaScript**: Interactividad en la interfaz de usuario.
* **Bases de Datos (asumido/potencial):**
    * **CSV**: Almacenamiento de datos históricos en archivos CSV.
    * *(Si usas alguna otra base de datos como SQLite, PostgreSQL, etc., añádelo aquí)*

## 🚀 Instalación y Uso

Sigue estos pasos para poner en marcha el proyecto en tu máquina local:

### Prerrequisitos

Asegúrate de tener instalado lo siguiente:

* [Python 3.9+](https://www.python.org/downloads/)
* [pip](https://pip.pypa.io/en/stable/installation/) (gestor de paquetes de Python)

### Pasos de Instalación

1.  **Clonar el repositorio:**

    ```bash
    git clone [https://github.com/haktor23/proyectefinal_ia-bd.git](https://github.com/haktor23/proyectefinal_ia-bd.git)
    cd proyectefinal_ia-bd/ProyecteFinal_IA-BD-jonatan/ # Ajusta la ruta si tu estructura es diferente
    ```

2.  **Crear y activar un entorno virtual (recomendado):**

    ```bash
    python -m venv venv
    # En Windows
    .\venv\Scripts\activate
    # En macOS/Linux
    source venv/bin/activate
    ```

3.  **Instalar las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación Flask:**

    ```bash
    python app.py
    ```

    La aplicación debería estar disponible en `http://127.0.0.1:5000/` en tu navegador web.

## 📁 Estructura del Proyecto

.
├── app.py                      # Punto de entrada principal de la aplicación Flask
├── data/                       # Contiene los archivos de datos
│   └── CSV/
│       ├── DistritoCP.csv
│       ├── HistoricoAire.csv
│       ├── HistoricoCamaras.csv
│       ├── HistoricoTemperatura.csv
│       └── imagenes_revisadas.csv
├── requirements.txt            # Dependencias del proyecto
├── static/                     # Archivos estáticos (CSS, JS, imágenes)
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
└── templates/                  # Plantillas HTML
├── base.html               # Plantilla base para todas las páginas
├── calidad.html            # Página de calidad del aire
├── camaras.html            # Página de cámaras de tráfico
├── chatbot.html            # Página del chatbot
├── index.html              # Página de inicio
├── machinelearning.html    # Página de Machine Learning
├── rag.html                # Página de RAG (si es diferente al chatbot)
└── realtime.html           # Página de datos en tiempo real
└── utils/                      # Módulos de utilidad y lógica de negocio
├── init.py
├── calidad.py              # Funciones para datos de calidad del aire
├── camaras.py              # Funciones para datos de cámaras
├── chatbot.py              # Lógica del chatbot
├── index.py                # Lógica para la página de inicio
├── machinelearning.py      # Implementación de modelos ML
├── rag.py                  # Funciones específicas para RAG
├── realtime.py             # Funciones para datos en tiempo real
└── tiempo.py               # Funciones relacionadas con el tiempo/temperatura


## 🤝 Contribución

¡Las contribuciones son bienvenidas! Si deseas contribuir a este proyecto, por favor sigue estos pasos:

1.  Haz un "fork" del repositorio.
2.  Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3.  Realiza tus cambios y haz "commit" de ellos (`git commit -m 'feat: Añadir nueva funcionalidad'`).
4.  Haz "push" a tu rama (`git push origin feature/nueva-funcionalidad`).
5.  Abre un "Pull Request".

Por favor, asegúrate de que tu código siga las convenciones de estilo existentes y que las pruebas (si las hay) pasen.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` (si existe) para más detalles.

## 📧 Contacto

Para cualquier pregunta o comentario, puedes contactar a:

* **Jonatan** (haktor23 en GitHub) - [Tu Correo Electrónico Aquí]

---
