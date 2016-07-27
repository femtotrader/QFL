from airflow.models import DAG
from airflow.operators import DummyOperator, BashOperator, PythonOperator
from datetime import datetime, timedelta
import os
import sys

# home = os.path.expanduser("~")
# local_repo = os.path.join("Documents", "Code", "qfl")
# os.chdir(os.path.join(home, local_repo))  # Activate .env
#
# home = os.path.expanduser("~")
# local_repo = os.path.join("Documents", "Code", "qfl")
# sys.path.append(os.path.join(home, local_repo))  # Activate .env
#
# modules = ["qfl"]
# sub_modules =['qfl', 'airflow']
# for sm in sub_modules:
#     modules.append(os.path.join(modules[0], sm))
# modules.append(os.path.join("qfl", "etl"))
# modules.append(os.path.join("qfl", "core"))
# modules.append(os.path.join("airflow", "dags"))

from qfl.etl.data_ingest import test_airflow, daily_equity_price_ingest

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 8, 1),
    'email': ['beifert@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG('etl_daily',
          start_date=datetime(2016, 05, 01),
          schedule_interval="0 0 14 * MON-FRI",
          default_args=default_args)

t1 = PythonOperator(task_id='test_airflow',
                    python_callable=test_airflow,
                    dag=dag)

t2 = PythonOperator(task_id='daily_equity_price_ingest',
                    python_callable=daily_equity_price_ingest,
                    dag=dag)

run_this_last = DummyOperator(task_id='run_this_last', dag=dag)

t2.set_upstream(t1)

run_this_last.set_upstream(t2)
