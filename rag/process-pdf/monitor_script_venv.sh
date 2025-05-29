#!/bin/bash

# Configuración
VENV_PATH="/home/alumnos_ia_2025/RAG_new/venv"
SCRIPT_PATH="/home/alumnos_ia_2025/RAG_new/main_pipeline.py"
LOG_FILE="/home/alumnos_ia_2025/RAG_ne-cron/script.log"
PID_FILE="/tmp/script_python_$(basename "/home/alumnos_ia_2025/RAG_new/main_pipeline.py" .py).pid"

# Aseguramos que el directorio de logs existe
mkdir -p $(dirname "$LOG_FILE")

# Función para iniciar el script
start_script() {
  echo "$(date): Iniciando script Python en entorno virtual $VENV_PATH..." >> "$LOG_FILE"
  
  # Verificar que el entorno virtual existe
  if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "$(date): ERROR - No se encontró el entorno virtual en $VENV_PATH" >> "$LOG_FILE"
    return 1
  fi
  
  # Activar el entorno virtual y ejecutar el script
  (
    source "$VENV_PATH/bin/activate"
    nohup python "$SCRIPT_PATH" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "$(date): Script iniciado con PID: $(cat $PID_FILE) usando entorno virtual $VENV_PATH" >> "$LOG_FILE"
  )
}

# Función para verificar si el script está en ejecución
check_script() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null; then
      echo "$(date): Script en ejecución con PID: $PID" >> "$LOG_FILE"
      return 0
    else
      echo "$(date): El script no está en ejecución. Reiniciando..." >> "$LOG_FILE"
      rm -f "$PID_FILE"
      return 1
    fi
  else
    echo "$(date): No se encontró archivo PID. Iniciando el script..." >> "$LOG_FILE"
    return 1
  fi
}

# Función para detener el script
stop_script() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
      kill $PID
      echo "$(date): Script detenido (PID: $PID)" >> "$LOG_FILE"
      rm -f "$PID_FILE"
    fi
  fi
}

# Verificar argumentos de línea de comandos
case "$1" in
  start)
    start_script
    ;;
  stop)
    stop_script
    ;;
  restart)
    stop_script
    sleep 2
    start_script
    ;;
  *)
    # Comportamiento por defecto (para el cron)
    if ! check_script; then
      start_script
    fi
    ;;
esac

exit 0
