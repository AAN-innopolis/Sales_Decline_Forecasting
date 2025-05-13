#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

AIRFLOW_PORT=8080
TENSORBOARD_PORT=6006
FEAST_PORT=8888

print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

start_airflow() {
    print_message "Starting Airflow locally..."
    
    AIRFLOW_HOME="$(pwd)/src/airflow"
    mkdir -p ${AIRFLOW_HOME}/logs/api_server
    mkdir -p ${AIRFLOW_HOME}/logs/scheduler
    mkdir -p ${AIRFLOW_HOME}/logs/dag_processor
    mkdir -p ${AIRFLOW_HOME}/logs/triggerer
    mkdir -p ${AIRFLOW_HOME}/db
    export AIRFLOW_HOME
    
    if [ ! -f "${AIRFLOW_HOME}/db/airflow.db" ]; then
        print_message "Initializing Airflow database..."
        uv run airflow db migrate
    else
        print_message "Airflow database already exists."
    fi
    
    print_message "Starting Airflow API server..."
    uv run nohup airflow api-server -p ${AIRFLOW_PORT} \
        >| ${AIRFLOW_HOME}/logs/api_server/api_server.log \
        2>| ${AIRFLOW_HOME}/logs/api_server/api_server.err < /dev/null &
    
    print_message "Starting Airflow scheduler..."
    uv run nohup airflow scheduler \
        >| ${AIRFLOW_HOME}/logs/scheduler/scheduler.log \
        2>| ${AIRFLOW_HOME}/logs/scheduler/scheduler.err < /dev/null &
    
    print_message "Starting Airflow DAG processor..."
    uv run nohup airflow dag-processor \
        >| ${AIRFLOW_HOME}/logs/dag_processor/dag_processor.log \
        2>| ${AIRFLOW_HOME}/logs/dag_processor/dag_processor.err < /dev/null &
    
    print_message "Starting Airflow triggerer..."
    uv run nohup airflow triggerer \
        >| ${AIRFLOW_HOME}/logs/triggerer/triggerer.log \
        2>| ${AIRFLOW_HOME}/logs/triggerer/triggerer.err < /dev/null &
}

start_tensorboard() {
    print_message "Starting Tensorboard tracking server..."
    TENSORBOARD_HOME="./src/tensorboard"
    mkdir -p ${TENSORBOARD_HOME}/
    mkdir -p ${TENSORBOARD_HOME}/logs
    
    uv run tensorboard \
        --logdir ${TENSORBOARD_HOME} 
        --host 0.0.0.0 \
        --port ${TENSORBOARD_PORT} &
}

start_tunnels() {
    print_message "Starting LocalTunnel..."
    print_message "Public IP (to be used as password in LocalTunnel):"
    curl https://loca.lt/mytunnelpassword
    print_message "Airflow:"
    lt --port ${AIRFLOW_PORT} &
    sleep 10
    print_message "Tensorboard:"
    lt --port ${TENSORBOARD_PORT} &
    sleep 10
    print_message "Tunnels started."
}

start_all() {
    print_message "Starting all services locally..."
    start_airflow
    start_tensorboard
    # start_tunnels
    print_message "All services started."
}

stop_airflow() {
    print_message "Stopping Airflow services..."
    pkill -f "airflow"
}

stop_tensorboard() {
    print_message "Stopping Tensorboard service..."
    pkill -f "tensorboard"
}

stop_all() {
    print_message "Stopping all services..."
    stop_airflow
    stop_tensorboard
}

check_status() {
    print_message "Checking services status..."
    echo "Redis processes:"
    ps aux | grep redis-server | grep -v grep
    echo "Airflow processes:"
    ps aux | grep airflow | grep -v grep
    echo "Tensorboard processes:"
    ps aux | grep tensorboard | grep -v grep
}

case "$1" in
    "start-airflow")
        start_airflow
        ;;
    "start-tensorboard")
        start_tensorboard
        ;;
    "start-all")
        start_all
        ;;
    "stop-airflow")
        stop_airflow
        ;;
    "stop-tensorboard")
        stop_tensorboard
        ;;
    "stop-all")
        stop_all
        ;;
    "status")
        check_status
        ;;
    *)
        echo "Usage: $0 {start-airflow|start-tensorboard|start-all|stop-airflow|stop-redis|stop-all|status}"
        exit 1
        ;;
esac