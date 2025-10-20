import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import tempfile

# Adicionar dependências ao path
sys.path.append('/tmp')

def lambda_handler(event, context):
    """
    Lambda function para executar pipeline de risco diário
    """
    print("🚀 Iniciando pipeline de risco na AWS Lambda...")
    
    try:
        # Criar diretório temporário para dados
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(f"{temp_dir}/data/processed", exist_ok=True)
            os.makedirs(f"{temp_dir}/reports", exist_ok=True)
            
            # Executar pipeline
            result = run_risk_pipeline(temp_dir)
            
            # Salvar resultados no S3 (opcional)
            if os.getenv('S3_BUCKET'):
                upload_to_s3(temp_dir, os.getenv('S3_BUCKET'))
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Pipeline executado com sucesso',
                    'timestamp': datetime.now().isoformat(),
                    'results': result
                })
            }
            
    except Exception as e:
        print(f"❌ Erro no pipeline: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

def run_risk_pipeline(temp_dir):
    """Executa o pipeline completo de risco"""
    results = {}
    
    try:
        # 1. Coleta de dados
        print("📥 Coletando dados...")
        from src.etl.data_collector_bcb import BCBDataCollector
        collector = BCBDataCollector()
        raw_data = collector.download_reliable_data()
        
        if not raw_data.empty:
            raw_data.to_parquet(f"{temp_dir}/data/raw_data.parquet")
            results['data_collection'] = 'success'
            results['data_shape'] = raw_data.shape
        else:
            results['data_collection'] = 'failed'
            return results
        
        # 2. Processamento
        print("🧮 Calculando métricas...")
        portfolio_prices, returns_simple, returns_log = collector.create_optimized_portfolio(raw_data)
        
        # Salvar dados processados
        portfolio_prices.to_parquet(f"{temp_dir}/data/processed/portfolio_prices.parquet")
        returns_simple.to_parquet(f"{temp_dir}/data/processed/portfolio_returns.parquet")
        
        # 3. Métricas básicas
        annual_returns = returns_simple.mean() * 252
        annual_volatility = returns_simple.std() * np.sqrt(252)
        
        results['metrics'] = {
            'annual_returns': annual_returns.to_dict(),
            'annual_volatility': annual_volatility.to_dict(),
            'portfolio_return': annual_returns.mean(),
            'portfolio_volatility': annual_volatility.mean()
        }
        
        # 4. Simulação simples
        print("🎲 Executando simulação...")
        try:
            from src.simulation.monte_carlo import MonteCarloSimulator
            simulator = MonteCarloSimulator(returns_simple)
            simulator.simulate_portfolio_paths()
            mc_metrics, _ = simulator.calculate_risk_metrics()
            results['monte_carlo'] = mc_metrics
        except Exception as e:
            print(f"⚠️ Simulação falhou: {e}")
            results['monte_carlo'] = 'simulation_failed'
        
        print("✅ Pipeline concluído!")
        return results
        
    except Exception as e:
        print(f"❌ Erro no pipeline: {e}")
        raise e

def upload_to_s3(local_dir, bucket_name):
    """Faz upload dos resultados para S3"""
    s3 = boto3.client('s3')
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.relpath(local_path, local_dir)
            
            try:
                s3.upload_file(local_path, bucket_name, f"risk-reports/{s3_path}")
                print(f"📤 Uploaded: {s3_path}")
            except Exception as e:
                print(f"❌ Upload failed: {s3_path} - {e}")
