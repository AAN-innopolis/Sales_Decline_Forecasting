"""
DAG для обработки данных проекта прогнозирования снижения продаж.
Выполняет подготовку базовых признаков и параллельную обработку признаков для разных моделей.
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.configuration import conf
from airflow.sdk import chain

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DAG_TASKS_DIR = PROJECT_ROOT / "src" / "airflow" / "dag_tasks"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=0),
}


dag = DAG(
    'sales_decline_data_processing',
    default_args=default_args,
    description='The pipeline for data processing for sales decline forecasting',
    # schedule_interval=timedelta(days=1),  # Ежедневный запуск
    # start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['sales', 'forecasting', 'data_processing'],
)


with TaskGroup(group_id='data_preparation', dag=dag) as data_preparation:
    check_data_exists = FileSensor(
        task_id='check_data_exists',
        filepath='data/raw/sazerac_df.csv',
        poke_interval=60,  # проверять каждую минуту
        timeout=60 * 1,    # таймаут 5 минут
        mode='poke',
        dag=dag
    )
    
    clean_data = BashOperator(
        task_id='clean_data',
        bash_command=f'python {DAG_TASKS_DIR}/clean_data.py --log-level INFO',
        dag=dag
    )
    
    check_cleaned_data = FileSensor(
        task_id='check_cleaned_data',
        filepath='data/prepared/cleaned_data.parquet',
        poke_interval=60,
        timeout=60 * 5,
        mode='poke',
        dag=dag
    )
    
    check_data_exists >> clean_data >> check_cleaned_data


with TaskGroup(group_id='model_features_preparation', dag=dag) as model_features_preparation:
    
    with TaskGroup(group_id='lstm_tft_common_features', dag=dag) as lstm_tft_common_features:
        prepare_lstm_tft_features = BashOperator(
            task_id='prepare_features',
            bash_command=f'python {DAG_TASKS_DIR}/prepare_lstm_tft_features.py --log-level INFO',
            dag=dag
        )
        
        check_lstm_tft_features = FileSensor(
            task_id='check_features',
            filepath='data/prepared/lstm_tft_features.parquet',
            poke_interval=60,
            timeout=60 * 5,
            mode='poke',
            dag=dag
        )
        
        prepare_lstm_tft_features >> check_lstm_tft_features
    
    with TaskGroup(group_id='lstm_tft_specific_features', dag=dag) as lstm_tft_specific_features:
        with TaskGroup(group_id='lstm_specific_features', dag=dag) as lstm_specific_features:
            prepare_lstm_features = BashOperator(
                task_id='prepare_lstm_features',
                bash_command=f'python {DAG_TASKS_DIR}/prepare_lstm_features.py --log-level INFO',
                dag=dag
            )
            encode_lstm_features = BashOperator(
                task_id='encode_lstm_features',
                bash_command=f'python {DAG_TASKS_DIR}/encode_lstm_features.py --log-level INFO  --embedding-dim 16',
                dag=dag
            )
            construct_lstm_dataset = BashOperator(
                task_id='construct_lstm_dataset',
                bash_command=f'python {DAG_TASKS_DIR}/construct_lstm_dataset.py --log-level INFO',
                dag=dag
            )
            prepare_lstm_features >> encode_lstm_features >> construct_lstm_dataset


        with TaskGroup(group_id='tft_specific_features', dag=dag) as tft_specific_features:
            prepare_tft_features = BashOperator(
                task_id='prepare_tft_features',
                bash_command=f'python {DAG_TASKS_DIR}/prepare_tft_features.py --log-level INFO',
                dag=dag
            )
            encode_tft_features = BashOperator(
                task_id='encode_tft_features',
                bash_command=f'python {DAG_TASKS_DIR}/encode_tft_features.py --log-level INFO',
                dag=dag
            )
            construct_tft_dataset = BashOperator(
                task_id='construct_tft_dataset',
                bash_command=f'python {DAG_TASKS_DIR}/construct_tft_dataset.py --log-level INFO',
                dag=dag
            )
            prepare_tft_features >> encode_tft_features >> construct_tft_dataset
        
    
    with TaskGroup(group_id='llm_features', dag=dag) as llm_features:
        prepare_llm_features = BashOperator(
            task_id='prepare_llm_features',
            bash_command=f'python {DAG_TASKS_DIR}/prepare_llm_features.py --log-level INFO',
            dag=dag
        )
        construct_llm_prompts = BashOperator(
            task_id='construct_llm_prompts',
            bash_command=f'python {DAG_TASKS_DIR}/construct_llm_prompts.py --log-level INFO',
            dag=dag
        )
        prepare_llm_features >> construct_llm_prompts
        
    
    lstm_tft_common_features >> lstm_tft_specific_features
    

with TaskGroup(group_id='check_features_exist', dag=dag) as check_features_exist:
    check_lstm_features = FileSensor(
        task_id='check_lstm_features',
        filepath='data/prepared/lstm_features.parquet',
        poke_interval=60,
        timeout=60 * 5,
        mode='poke',
        dag=dag
    )
    
    check_tft_features = FileSensor(
        task_id='check_tft_features',
        filepath='data/prepared/tft_features.parquet',
        poke_interval=60,
        timeout=60 * 5,
        mode='poke',
        dag=dag
    )
    
    check_llm_features = FileSensor(
        task_id='check_llm_features',
        filepath='data/prepared/llm_features.parquet',
        poke_interval=60,
        timeout=60 * 5,
        mode='poke',
        dag=dag
    )
    

data_preparation >> model_features_preparation >> check_features_exist

# # Применение Feast
# apply_feast = BashOperator(
#     task_id='apply_feast',
#     bash_command=f'python {DAG_TASKS_DIR}/init_feast.py init --log-level INFO',
#     dag=dag,
# )

# # Материализация признаков в Feast
# materialize_feast = BashOperator(
#     task_id='materialize_feast',
#     bash_command=f'python {DAG_TASKS_DIR}/materialize_feast.py --log-level INFO',
#     dag=dag,
# )
