#!/bin/bash

# Script de gestión para el monitor Python con venv
MONITOR_SCRIPT="/home/alumnos_ia_2025/RAG_new/monitor_script_venv.sh"
LOG_FILE="/home/alumnos_ia_2025/RAG_ne-cron/script.log"
SCRIPT_NAME=$(basename "/home/alumnos_ia_2025/RAG_new/main_pipeline.py" .py)
PID_FILE="/tmp/script_python_$SCRIPT_NAME.pid"

case "$1" in
  start)
    echo "Iniciando el script..."
    $MONITOR_SCRIPT start
    ;;
  stop)
    echo "Deteniendo el script..."
    $MONITOR_SCRIPT stop
    ;;
  restart)
    echo "Reiniciando el script..."
    $MONITOR_SCRIPT restart
    ;;
  status)
    if [ -f "$PID_FILE" ]; then
      PID=$(cat "$PID_FILE")
      if ps -p $PID > /dev/null; then
        echo "El script está en ejecución con PID: $PID"
        echo "Tiempo de ejecución: $(ps -o etime= -p $PID)"
        echo "Uso de memoria: $(ps -o rss= -p $PID) KB"
      else
        echo "El script no está en ejecución (archivo PID obsoleto)"
        rm -f "$PID_FILE"
      fi
    else
      echo "El script no está en ejecución"
    fi
    ;;
  logs)
    echo "Mostrando los últimos logs del script:"
    tail -20 "$LOG_FILE"
    ;;
  follow)
    echo "Siguiendo los logs en tiempo real (Ctrl+C para salir):"
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "Uso: $0 {start|stop|restart|status|logs|follow}"
    exit 1
    ;;
esac
