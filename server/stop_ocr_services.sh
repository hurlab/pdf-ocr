#!/bin/bash
# Stop vLLM OCR model servers

PID_DIR="$SCRIPT_DIR/pids"

stop_service() {
    local name="$1"
    local pid_file="$PID_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (PID $pid)..."
            kill "$pid"
            # Wait up to 15 seconds for graceful shutdown
            for i in $(seq 1 15); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing $name..."
                kill -9 "$pid"
            fi
            echo "  $name stopped."
        else
            echo "$name not running (stale PID file)."
        fi
        rm -f "$pid_file"
    else
        echo "$name: no PID file found."
    fi
}

echo "Stopping OCR vLLM services..."
stop_service "paddleocr"
stop_service "hunyuan"
stop_service "deepseek"
echo "Done."
