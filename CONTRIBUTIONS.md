# Contributing to Sales Decline Forecasting Project

## Environment Setup

The project uses UV as both a package manager and environment creator. To set up the complete environment and install all dependencies, run:

```bash
uv sync
```

## Service Management

The project uses a management script `manage.sh` to control various services. Here's how to use it:

```bash
./manage.sh <command>
```

Available commands:
- `start-all`: Starts all services (Redis, Airflow, MLflow, and Feast)
- `stop-all`: Stops all running services
- `status`: Shows the status of all service processes

## Service Access

The following services are available with their respective UI endpoints:

| Service | URL | Description |
|---------|-----|-------------|
| Airflow | http://localhost:8080 | Workflow management and scheduling |
| MLflow | http://localhost:5000 | Experiment tracking and model management |
| Feast | http://localhost:8888 | Feature store management |

## Airflow Configuration

### Airflow Services
When Airflow starts, it launches the following services:
- `api-server`: Handles REST API requests
- `scheduler`: Manages DAG scheduling and task execution
- `dag-processor`: Processes DAG files and updates the database
- `triggerer`: Handles trigger-based task execution

### Authentication
Airflow uses a simple authentication system. Login credentials are automatically generated and stored at:
```
PROJECT_ROOT/src/airflow/db/simple_auth_manager_passwords.json.generated
```

### File System Connection
Before using Airflow, ensure the file system connection is properly configured:

1. Navigate to Admin => Connections in the Airflow UI
2. Verify the existence of a connection with:
   - Connection Id: `fs_default`
   - Connection Type: `fs`
3. If the connection doesn't exist, create it using the "Add Connection" button

### Development and Debugging
- Tasks can be executed manually through the Airflow UI
- Individual task instances can be run and monitored separately
- Logs are available through the Airflow interface

### Data Processing
- The main data processing DAG is located at: `PROJECT_ROOT/src/airflow/dags/data_processing_dag.py`
- Processed data is stored in: `PROJECT_ROOT/data/prepared`

### Configuration Parameters

The main Airflow configuration file is located at:
```
PROJECT_ROOT/src/airflow/airflow.cfg
```

You'll need to edit this file to update the configuration parameters listed below.

```cfg
dag_processor_child_process_log_directory = PROJECT_ROOT/src/airflow/logs/child_process
dags_folder = PROJECT_ROOT/src/airflow/dags
simple_auth_manager_passwords_file = PROJECT_ROOT/src/airflow/db/simple_auth_manager_passwords.json.generated
sql_alchemy_conn = sqlite:////PROJECT_ROOT/src/airflow/db/airflow.db
base_log_folder = PROJECT_ROOT/src/airflow/logs
load_examples = False
```


⚠️ **Important**: Replace `PROJECT_ROOT` with the absolute path to your project directory in all configuration files.

#### Notes
- View distribution information
> cat /etc/*-release
- View public IP (used as LocalTunnel password)
> curl https://loca.lt/mytunnelpassword
