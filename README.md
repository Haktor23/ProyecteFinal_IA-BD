# ProyecteFinal_IA-BD: Sistema Inteligente de Monitorización Urbana

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3.x-black?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-lightgreen?style=for-the-badge&logo=pandas&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

## 📖 Prólogo del Proyecto

Proyecto integrador especialización en Inteligencia Artificial y Big Data, centrado en el análisis y control de la contaminación ambiental para mejorar la gestión del tráfico y la movilidad urbana en Valencia. El objetivo es desarrollar una solución de software que utilice la IA y el Big Data para recopilar y analizar datos de calidad del aire, predecir niveles de contaminación, y generar alertas y simulaciones que apoyen la toma de decisiones operativas y estratégicas por parte de los organismos públicos. El proyecto incluye la creación de un sistema de ingesta de datos, modelos predictivos y clasificadores de imágenes, un panel de control interactivo y un chatbot para consultar información relevante, con el fin de implementar restricciones dinámicas de tráfico y otras medidas de mitigación basadas en los niveles de contaminación en tiempo real y previstos.

Hemos desarrollado una aplicación web dinámica que permite a los usuarios explorar datos históricos y en tiempo real, interactuar con modelos de Machine Learning y obtener respuestas a sus preguntas a través de un sistema RAG (Retrieval-Augmented Generation) potenciado por IA. Este proyecto es la culminación de un esfuerzo por integrar tecnologías avanzadas para crear soluciones prácticas y accesibles para la toma de decisiones informada en un contexto urbano.

## ✨ [Características Principales](./PresentacionBigDataIA.pdf)

- **Monitorización de Calidad del Aire y Temperatura:** Visualización y análisis de datos históricos de calidad del aire. Realizamos request de forma periodica a la [api](https://nominatim.openstreetmap.org) y guardamos los datos extraidos.
  - [Recogida de datos](./ingesta-datos/estacionesmeteo-colab.ipynb)
- **Monitorización de Cámaras de Tráfico:** Gestión y visualización de datos de cámaras de tráfico (posiblemente imágenes y metadatos). Accedemos a las [camaras de valencia](camaras.valencia.es) para poder quedarnos con una captura en el momento del request.

  - [Scraping](./ingesta-datos/scraping/)
  - [FFmpg](./ingesta-datos/ffmpg/)

  Para el procesamiento de las imagen utilizamos [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN) para mejorar la calidad de las imagenes y [**ultralytics**](https://github.com/ultralytics/ultralytics) para la deteccion de vehiculos dentro de las imagenes. Para realzar esta tarea es necesario capacidad de hardware y el [codigo](./procesar-datos/images-yolo.py) se ha realizado pensando en su optimizacion, por ello se ejecutan tareas en paralelo.

- **Capacidades de Machine Learning:** Implementación de modelos para predicción o análisis de patrones urbanos.

  - [EDA](https://colab.research.google.com/drive/1AwcKQTgA36tbLWY3Qg3byaj2qXbrNWqJ)

- **Chatbot Inteligente:** Un asistente conversacional para responder preguntas relacionadas con los datos urbanos utilizando **RAG**.

  - [PDFs](https://drive.google.com/drive/folders/1JmVWpUJKNMuxcLBH8Ry__bGrHo1tOTyY?usp=drive_link)
  - [Division de textos](./rag/semantic-pdf-processing.py)
  - [Sincronizacion datos con elasticsearch](./rag/subir_rag.py)
  - [Automatizacion de Procesos](./rag/process-pdf/)

- [**Visualización de Datos Interactiva:**](./frontend/readme.md) Interfaz web intuitiva para explorar y comprender los datos.

## 🗃️ Almacenamiento datos

Los flujos de datos en Azure DataFactory permiten integrar y procesar múltiples fuentes heterogéneas (aire, temperatura y cámaras), generando tablas raw organizadas y limpias que son almacenadas en [Databricks](https://www.databricks.com/) para su análisis posterior.

## 📊 Visualizacion/Analisis datos

Para el análisis exploratorio y la creación de dashboards avanzados, se utilizó [Power BI](https://www.microsoft.com/es-es/power-platform/products/power-bi). Esta herramienta permitió conectar con los datasets procesados en Databricks, generando informes visuales, paneles interactivos y métricas clave sobre calidad del aire, tráfico y condiciones meteorológicas, facilitando la toma de decisiones por parte de usuarios técnicos y no técnicos.
