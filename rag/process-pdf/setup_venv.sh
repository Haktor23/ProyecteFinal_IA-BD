#!/bin/bash

# Script de configuración y prueba para el monitor con entorno virtual (venv)
# Este script te ayudará a configurar correctamente todo el sistema

echo "========================================================"
echo "Configuración del monitor para script Python con venv"
echo "========================================================"

# Directorio actual
CURRENT_DIR=$(pwd)

# 1. Recopilar información necesaria
echo -e "\n[1] Recopilando información necesaria...\n"

# Ruta al entorno virtual
read -p "Introduce la ruta completa a tu entorno virtual (ej: /home/usuario/mi_proyecto/venv): " VENV_PATH
while [ ! -f "$VENV_PATH/bin/activate" ]; do
    echo "El entorno virtual no existe o no es válido. Debe contener bin/activate."
    read -p "¿Deseas crear un nuevo entorno virtual? (s/n): " CREATE_VENV
    if [[ "$CREATE_VENV" == "s" || "$CREATE_VENV" == "S" ]]; then
        read -p "Introduce la ruta donde crear el entorno virtual: " NEW_VENV_PATH
        read -p "¿Qué versión de Python usar? (ej: python3.11): " PYTHON_VERSION
        
        # Verificar que la versión de Python existe
        if command -v $PYTHON_VERSION &> /dev/null; then
            echo "Creando entorno virtual con $PYTHON_VERSION..."
            $PYTHON_VERSION -m venv "$NEW_VENV_PATH"
            if [ $? -eq 0 ]; then
                VENV_PATH="$NEW_VENV_PATH"
                echo "Entorno virtual creado exitosamente en: $VENV_PATH"
                break
            else
                echo "Error al crear el entorno virtual."
            fi
        else
            echo "La versión $PYTHON_VERSION no está disponible en el sistema."
        fi
    else
        read -p "Introduce la ruta completa a tu entorno virtual: " VENV_PATH
    fi
done

# Ruta al script Python
read -p "Introduce la ruta completa a tu script Python: " SCRIPT_PATH
while [ ! -f "$SCRIPT_PATH" ]; do
    echo "El archivo no existe. Por favor, introduce una ruta válida."
    read -p "Introduce la ruta completa a tu script Python: " SCRIPT_PATH
done

# Directorio de logs
read -p "Introduce la ruta para los archivos de log (ej: /home/usuario/logs): " LOG_DIR
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/script.log"
CRON_LOG="$LOG_DIR/cron.log"

# Usuario actual
CURRENT_USER=$(whoami)
HOME_DIR=$(eval echo ~$CURRENT_USER)

# 2. Verificar dependencias en el entorno virtual
echo -e "\n[2] Verificando entorno virtual...\n"

# Activar el entorno y verificar Python
source "$VENV_PATH/bin/activate"
PYTHON_VERSION_VENV=$(python --version)
echo "Versión de Python en el entorno virtual: $PYTHON_VERSION_VENV"

# Verificar si el script puede importar sus dependencias
echo "Verificando dependencias del script..."
if python -c "import ast; ast.parse(open('$SCRIPT_PATH').read())" 2>/dev/null; then
    echo "El script tiene una sintaxis válida."
else
    echo "ADVERTENCIA: El script puede tener errores de sintaxis."
fi

deactivate

# 3. Crear el script de monitoreo
echo -e "\n[3] Creando script de monitoreo...\n"

MONITOR_SCRIPT="$CURRENT_DIR/monitor_script_venv.sh"

cat > "$MONITOR_SCRIPT" << EOL
#!/bin/bash

# Configuración
VENV_PATH="$VENV_PATH"
SCRIPT_PATH="$SCRIPT_PATH"
LOG_FILE="$LOG_FILE"
PID_FILE="/tmp/script_python_\$(basename "$SCRIPT_PATH" .py).pid"

# Aseguramos que el directorio de logs existe
mkdir -p \$(dirname "\$LOG_FILE")

# Función para iniciar el script
start_script() {
  echo "\$(date): Iniciando script Python en entorno virtual \$VENV_PATH..." >> "\$LOG_FILE"
  
  # Verificar que el entorno virtual existe
  if [ ! -f "\$VENV_PATH/bin/activate" ]; then
    echo "\$(date): ERROR - No se encontró el entorno virtual en \$VENV_PATH" >> "\$LOG_FILE"
    return 1
  fi
  
  # Activar el entorno virtual y ejecutar el script
  (
    source "\$VENV_PATH/bin/activate"
    nohup python "\$SCRIPT_PATH" >> "\$LOG_FILE" 2>&1 &
    echo \$! > "\$PID_FILE"
    echo "\$(date): Script iniciado con PID: \$(cat \$PID_FILE) usando entorno virtual \$VENV_PATH" >> "\$LOG_FILE"
  )
}

# Función para verificar si el script está en ejecución
check_script() {
  if [ -f "\$PID_FILE" ]; then
    PID=\$(cat "\$PID_FILE")
    
    if ps -p \$PID > /dev/null; then
      echo "\$(date): Script en ejecución con PID: \$PID" >> "\$LOG_FILE"
      return 0
    else
      echo "\$(date): El script no está en ejecución. Reiniciando..." >> "\$LOG_FILE"
      rm -f "\$PID_FILE"
      return 1
    fi
  else
    echo "\$(date): No se encontró archivo PID. Iniciando el script..." >> "\$LOG_FILE"
    return 1
  fi
}

# Función para detener el script
stop_script() {
  if [ -f "\$PID_FILE" ]; then
    PID=\$(cat "\$PID_FILE")
    if ps -p \$PID > /dev/null; then
      kill \$PID
      echo "\$(date): Script detenido (PID: \$PID)" >> "\$LOG_FILE"
      rm -f "\$PID_FILE"
    fi
  fi
}

# Verificar argumentos de línea de comandos
case "\$1" in
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
EOL

# Hacer el script ejecutable
chmod +x "$MONITOR_SCRIPT"
echo "Script de monitoreo creado en: $MONITOR_SCRIPT"

# 4. Crear script de gestión adicional
echo -e "\n[4] Creando script de gestión...\n"

MANAGE_SCRIPT="$CURRENT_DIR/manage_script.sh"

cat > "$MANAGE_SCRIPT" << EOL
#!/bin/bash

# Script de gestión para el monitor Python con venv
MONITOR_SCRIPT="$MONITOR_SCRIPT"
LOG_FILE="$LOG_FILE"
SCRIPT_NAME=\$(basename "$SCRIPT_PATH" .py)
PID_FILE="/tmp/script_python_\$SCRIPT_NAME.pid"

case "\$1" in
  start)
    echo "Iniciando el script..."
    \$MONITOR_SCRIPT start
    ;;
  stop)
    echo "Deteniendo el script..."
    \$MONITOR_SCRIPT stop
    ;;
  restart)
    echo "Reiniciando el script..."
    \$MONITOR_SCRIPT restart
    ;;
  status)
    if [ -f "\$PID_FILE" ]; then
      PID=\$(cat "\$PID_FILE")
      if ps -p \$PID > /dev/null; then
        echo "El script está en ejecución con PID: \$PID"
        echo "Tiempo de ejecución: \$(ps -o etime= -p \$PID)"
        echo "Uso de memoria: \$(ps -o rss= -p \$PID) KB"
      else
        echo "El script no está en ejecución (archivo PID obsoleto)"
        rm -f "\$PID_FILE"
      fi
    else
      echo "El script no está en ejecución"
    fi
    ;;
  logs)
    echo "Mostrando los últimos logs del script:"
    tail -20 "\$LOG_FILE"
    ;;
  follow)
    echo "Siguiendo los logs en tiempo real (Ctrl+C para salir):"
    tail -f "\$LOG_FILE"
    ;;
  *)
    echo "Uso: \$0 {start|stop|restart|status|logs|follow}"
    exit 1
    ;;
esac
EOL

chmod +x "$MANAGE_SCRIPT"
echo "Script de gestión creado en: $MANAGE_SCRIPT"

# 5. Configurar crontab
echo -e "\n[5] Configurando crontab...\n"

TEMP_CRONTAB="$CURRENT_DIR/temp_crontab.txt"
crontab -l > "$TEMP_CRONTAB" 2>/dev/null || echo "" > "$TEMP_CRONTAB"

cat >> "$TEMP_CRONTAB" << EOL

# Monitor para script Python con entorno virtual (Configurado el $(date))
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOME=$HOME_DIR

# Ejecutar el script de monitoreo cada 5 minutos
*/5 * * * * $MONITOR_SCRIPT >> $CRON_LOG 2>&1

# Asegurar que el script se ejecute al iniciar el sistema
@reboot sleep 30 && $MONITOR_SCRIPT >> $CRON_LOG 2>&1
EOL

echo "Se añadirá la siguiente configuración al crontab:"
echo "------------------------------------------------------"
cat "$TEMP_CRONTAB" | tail -7
echo "------------------------------------------------------"

read -p "¿Confirmas que deseas añadir esta configuración al crontab? (s/n): " CONFIRM
if [[ "$CONFIRM" == "s" || "$CONFIRM" == "S" ]]; then
    crontab "$TEMP_CRONTAB"
    echo "Crontab actualizado correctamente."
else
    echo "No se ha modificado el crontab."
fi

rm "$TEMP_CRONTAB"

# 6. Probar el script
echo -e "\n[6] Probando el script...\n"
read -p "¿Deseas probar el script de monitoreo ahora? (s/n): " TEST
if [[ "$TEST" == "s" || "$TEST" == "S" ]]; then
    echo "Ejecutando el script de monitoreo..."
    "$MONITOR_SCRIPT"
    echo "Verificando si el proceso se inició correctamente..."
    sleep 3
    
    # Verificar estado usando el script de gestión
    "$MANAGE_SCRIPT" status
fi

echo -e "\n========================================================"
echo "Configuración completada."
echo ""
echo "Archivos creados:"
echo "- Script de monitoreo: $MONITOR_SCRIPT"
echo "- Script de gestión: $MANAGE_SCRIPT"
echo ""
echo "Logs:"
echo "- Logs del script: $LOG_FILE"
echo "- Logs del cron: $CRON_LOG"
echo ""
echo "Comandos útiles:"
echo "- Iniciar: $MANAGE_SCRIPT start"
echo "- Parar: $MANAGE_SCRIPT stop"
echo "- Estado: $MANAGE_SCRIPT status"
echo "- Ver logs: $MANAGE_SCRIPT logs"
echo "- Seguir logs: $MANAGE_SCRIPT follow"
echo "========================================================"
