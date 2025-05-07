#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'


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
        airflow db migrate
    else
        print_message "Airflow database already exists."
    fi
    
    print_message "Starting Airflow API server..."
    nohup airflow api-server -p 8080 >| ${AIRFLOW_HOME}/logs/api_server/api_server.log 2>| ${AIRFLOW_HOME}/logs/api_server/api_server.err < /dev/null &
    
    print_message "Starting Airflow scheduler..."
    nohup airflow scheduler >| ${AIRFLOW_HOME}/logs/scheduler/scheduler.log 2>| ${AIRFLOW_HOME}/logs/scheduler/scheduler.err < /dev/null &
    
    print_message "Starting Airflow DAG processor..."
    nohup airflow dag-processor >| ${AIRFLOW_HOME}/logs/dag_processor/dag_processor.log 2>| ${AIRFLOW_HOME}/logs/dag_processor/dag_processor.err < /dev/null &
    
    print_message "Starting Airflow triggerer..."
    nohup airflow triggerer >| ${AIRFLOW_HOME}/logs/triggerer/triggerer.log 2>| ${AIRFLOW_HOME}/logs/triggerer/triggerer.err < /dev/null &
}

start_mlflow() {
    print_message "Starting MLflow tracking server..."
    MLFLOW_HOME="./src/mlflow"
    mkdir -p ${MLFLOW_HOME}/artifacts
    mkdir -p ${MLFLOW_HOME}/db
    mkdir -p ${MLFLOW_HOME}/logs
    
    nohup mlflow server \
        --backend-store-uri sqlite:///${MLFLOW_HOME}/db/mlflow.db \
        --default-artifact-root ${MLFLOW_HOME}/artifacts \
        --host 0.0.0.0 \
        --port 5000 \
        --workers 4 >| ${MLFLOW_HOME}/logs/mlflow.log 2>| ${MLFLOW_HOME}/logs/mlflow.err < /dev/null &
}

start_feast() {
    print_message "Starting Feast feature server..."
    FEAST_HOME="$(pwd)/src/feast"
    FEAST_CONFIG="${FEAST_HOME}/feature_store.yaml"

    mkdir -p ${FEAST_HOME}/logs/feast
    mkdir -p ${FEAST_HOME}/logs/feast_ui
    mkdir -p ${FEAST_HOME}/db
    # export FEAST_REPO_PATH=FEAST_HOME

    cd ${FEAST_HOME}
    nohup feast -f ${FEAST_CONFIG} serve >| ${FEAST_HOME}/logs/feast/feast.log 2>| ${FEAST_HOME}/logs/feast/feast.err < /dev/null &
    nohup feast -f ${FEAST_CONFIG} ui >| ${FEAST_HOME}/logs/feast_ui/feast_ui.log 2>| ${FEAST_HOME}/logs/feast_ui/feast_ui.err < /dev/null &
    cd - > /dev/null
}

start_redis() {
    print_message "Starting Redis server..."
    REDIS_HOME="$(pwd)/src/redis"
    mkdir -p ${REDIS_HOME}/data
    mkdir -p ${REDIS_HOME}/logs
    
    redis-server --dir ${REDIS_HOME}/data --logfile ${REDIS_HOME}/logs/redis.log --daemonize yes < /dev/null &
    print_message "Redis server started."
}

start_all() {
    print_message "Starting all services locally..."
    start_redis
    start_airflow
    start_mlflow
    start_feast
    print_message "All services started. Access them at:"
    print_message "- Airflow API: http://localhost:8080"
    print_message "- MLflow: http://localhost:5000"
    print_message "- Feast: http://localhost:8888"
}

stop_airflow() {
    print_message "Stopping Airflow services..."
    pkill -f "airflow"
}

stop_mlflow() {
    print_message "Stopping MLflow service..."
    pkill -f "mlflow"
    pkill -f "gunicorn"
}

stop_feast() {
    print_message "Stopping Feast service..."
    pkill -f "feast"
}

stop_redis() {
    print_message "Stopping Redis service..."
    redis-cli shutdown
}

stop_all() {
    print_message "Stopping all services..."
    stop_airflow
    stop_mlflow
    stop_feast
    stop_redis
}

check_status() {
    print_message "Checking services status..."
    echo "Redis processes:"
    ps aux | grep redis-server | grep -v grep
    echo "Airflow processes:"
    ps aux | grep airflow | grep -v grep
    echo "MLflow processes:"
    ps aux | grep mlflow | grep -v grep
    echo "Feast processes:"
    ps aux | grep feast | grep -v grep
}


case "$1" in
    "start-airflow")
        start_airflow
        ;;
    "start-mlflow")
        start_mlflow
        ;;
    "start-feast")
        start_feast
        ;;
    "start-redis")
        start_redis
        ;;
    "start-all")
        start_all
        ;;
    "stop-airflow")
        stop_airflow
        ;;
    "stop-mlflow")
        stop_mlflow
        ;;
    "stop-feast")
        stop_feast
        ;;
    "stop-redis")
        stop_redis
        ;;
    "stop-all")
        stop_all
        ;;
    "status")
        check_status
        ;;
    *)
        echo "Usage: $0 {start-airflow|start-mlflow|start-feast|start-redis|start-all|stop-airflow|stop-mlflow|stop-feast|stop-redis|stop-all|status}"
        exit 1
        ;;
esac 