from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
import sys
import os

# Adicionar src ao path
sys.path.append('/opt/airflow/src')

default_args = {
    'owner': 'risk_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_data_collection():
    """Executa a coleta de dados do BCB"""
    import sys
    sys.path.append('/opt/airflow/src')
    from etl.data_collector_bcb import main
    main()

def run_risk_calculation():
    """Executa o cálculo de métricas de risco"""
    import sys
    sys.path.append('/opt/airflow/src')
    from metrics.risk_calculator import main
    main()

def run_visualizations():
    """Gera visualizações"""
    import sys
    sys.path.append('/opt/airflow/src')
    from visualization.risk_plots import main
    main()

def run_monte_carlo():
    """Executa simulações de Monte Carlo"""
    import sys
    sys.path.append('/opt/airflow/src')
    from simulation.monte_carlo import main
    main()

def generate_daily_report():
    """Gera relatório diário"""
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Carregar dados mais recentes
    returns = pd.read_parquet('/opt/airflow/data/processed/macro_portfolio_returns.parquet')
    
    # Métricas do dia
    latest_returns = returns.iloc[-1]
    portfolio_return = latest_returns.mean()
    
    # Criar relatório simples
    fig, ax = plt.subplots(figsize=(10, 6))
    latest_returns.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(f'Retornos do Dia - {datetime.now().strftime("%d/%m/%Y")}')
    ax.set_ylabel('Retorno')
    plt.tight_layout()
    
    # Salvar relatório
    report_path = f'/opt/airflow/reports/daily_report_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(report_path)
    plt.close()
    
    print(f"Relatório diário salvo em: {report_path}")

with DAG(
    'risk_analytics_pipeline',
    default_args=default_args,
    description='Pipeline completo de análise de risco',
    schedule_interval='0 9 * * 1-5',  # Dias úteis às 9h
    catchup=False,
    tags=['risk', 'analytics', 'b3'],
) as dag:

    # Task 1: Coleta de dados
    collect_data = PythonOperator(
        task_id='collect_market_data',
        python_callable=run_data_collection,
    )

    # Task 2: Cálculo de métricas
    calculate_metrics = PythonOperator(
        task_id='calculate_risk_metrics',
        python_callable=run_risk_calculation,
    )

    # Task 3: Visualizações
    generate_visualizations = PythonOperator(
        task_id='generate_risk_visualizations',
        python_callable=run_visualizations,
    )

    # Task 4: Simulações
    run_simulations = PythonOperator(
        task_id='run_monte_carlo_simulations',
        python_callable=run_monte_carlo,
    )

    # Task 5: Relatório diário
    daily_report = PythonOperator(
        task_id='generate_daily_report',
        python_callable=generate_daily_report,
    )

    # Task 6: Notificação (opcional)
    notify_completion = BashOperator(
        task_id='notify_pipeline_completion',
        bash_command='echo "Pipeline de risco executado com sucesso em $(date)"',
    )

    # Definir dependências
    collect_data >> calculate_metrics >> [generate_visualizations, run_simulations] >> daily_report >> notify_completion
