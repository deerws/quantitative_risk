import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import warnings
import time

warnings.filterwarnings('ignore')

class BCBDataCollector:
    def __init__(self):
        self.base_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados"
        
        # ✅ APENAS CÓDIGOS CONFIRMADOS QUE FUNCIONAM
        self.series_bcb = {
            # 💰 Taxas de Juros (FUNCIONAM)
            'SELIC': 11,                      # Taxa SELIC - CONFIRMADO
            'CDI': 12,                        # Taxa CDI - CONFIRMADO
            
            # 📈 Inflação (FUNCIONAM)  
            'IPCA': 433,                      # IPCA Mensal - CONFIRMADO
            'IGPM': 189,                      # IGP-M - CONFIRMADO
            
            # 💵 Câmbio (FUNCIONAM)
            'USD_BRL': 1,                     # USD/BRL - CONFIRMADO
            'EUR_BRL': 21619,                 # EUR/BRL - CONFIRMADO
            
            # 🛢️ Commodities (FUNCIONAM)
            'PETROLEO_BRENT': 20742,          # Petróleo Brent - CONFIRMADO
        }
    
    def get_bcb_data_simple(self, codigo, nome):
        """Busca dados do BCB - método SIMPLES e CONFIÁVEL"""
        url = self.base_url.format(codigo)
        
        try:
            # Período fixo de 2 anos para consistência
            end_date = datetime.now().strftime('%d/%m/%Y')
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%d/%m/%Y')
            
            params = {
                'formato': 'json',
                'dataInicial': start_date,
                'dataFinal': end_date
            }
            
            print(f"📥 Baixando {nome}...", end=" ")
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
                    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                    df = df.set_index('data').sort_index()
                    
                    # Remover valores extremos (outliers)
                    Q1 = df['valor'].quantile(0.01)
                    Q3 = df['valor'].quantile(0.99)
                    df = df[(df['valor'] >= Q1) & (df['valor'] <= Q3)]
                    
                    result_df = df[['valor']].rename(columns={'valor': nome})
                    print(f"✅ {len(result_df)} períodos")
                    return result_df
                else:
                    print("❌ Dados vazios")
                    return None
            else:
                print(f"❌ HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Erro: {str(e)[:50]}...")
            return None
    
    def download_reliable_data(self):
        """Baixa APENAS os dados que FUNCIONAM"""
        print("🏦 INICIANDO COLETA DE DADOS CONFIÁVEIS DO BCB")
        print("=" * 50)
        
        all_data = []
        successful_downloads = 0
        
        for nome, codigo in self.series_bcb.items():
            data = self.get_bcb_data_simple(codigo, nome)
            
            if data is not None and not data.empty:
                all_data.append(data)
                successful_downloads += 1
        
        print("=" * 50)
        print(f"📊 RESUMO: {successful_downloads}/{len(self.series_bcb)} séries obtidas")
        
        if all_data:
            # Combinar dados
            combined_df = pd.concat(all_data, axis=1)
            
            # Preencher valores missing de forma conservadora
            combined_df = combined_df.ffill().bfill().dropna()
            
            if not combined_df.empty:
                print(f"🎯 DATASET FINAL: {combined_df.shape}")
                return combined_df
        
        print("❌ Nenhum dado válido obtido")
        return pd.DataFrame()
    
    def create_optimized_portfolio(self, prices_df):
        """Cria portfólio otimizado com os dados disponíveis"""
        print("\n📊 CRIANDO PORTFÓLIO PARA ANÁLISE DE RISCO...")
        
        available_assets = list(prices_df.columns)
        print(f"💼 Ativos disponíveis: {available_assets}")
        
        # Estratégia: Portfolio diversificado com os dados que temos
        portfolio_prices = prices_df.copy()
        
        # Normalizar para base 100 (padronização)
        portfolio_normalized = portfolio_prices / portfolio_prices.iloc[0] * 100
        
        # Calcular retornos
        returns_simple = portfolio_normalized.pct_change().dropna()
        returns_log = np.log(portfolio_normalized / portfolio_normalized.shift(1)).dropna()
        
        print(f"✅ Portfólio criado: {portfolio_normalized.shape}")
        print(f"📈 Retornos calculados: {returns_simple.shape}")
        
        return portfolio_normalized, returns_simple, returns_log
    
    def save_optimized_data(self, prices_df, returns_simple, returns_log):
        """Salva dados de forma organizada"""
        os.makedirs('data/processed', exist_ok=True)
        
        # Salvar dados principais
        prices_df.to_parquet('data/processed/portfolio_prices.parquet')
        returns_simple.to_parquet('data/processed/portfolio_returns.parquet')
        returns_log.to_parquet('data/processed/portfolio_returns_log.parquet')
        
        # Salvar CSVs para verificação
        prices_df.to_csv('data/processed/portfolio_prices.csv')
        returns_simple.to_csv('data/processed/portfolio_returns.csv')
        
        print("💾 Dados salvos em data/processed/")
        print("📁 Arquivos criados:")
        print("   • portfolio_prices.parquet/csv")
        print("   • portfolio_returns.parquet/csv") 
        print("   • portfolio_returns_log.parquet")

def main():
    print("🚀 COLETOR BCB - VERSÃO OTIMIZADA")
    print("⭐ Usando apenas fontes CONFIRMADAS e CONFIÁVEIS\n")
    
    collector = BCBDataCollector()
    
    # Baixar dados confiáveis
    raw_data = collector.download_reliable_data()
    
    if not raw_data.empty:
        print(f"\n✅ SUCESSO! Dados obtidos do Banco Central")
        print(f"📊 Dataset: {raw_data.shape}")
        print(f"📅 Período: {raw_data.index[0].strftime('%d/%m/%Y')} até {raw_data.index[-1].strftime('%d/%m/%Y')}")
        print(f"📈 Séries: {list(raw_data.columns)}")
        
        # Criar portfólio
        portfolio_prices, returns_simple, returns_log = collector.create_optimized_portfolio(raw_data)
        
        if not portfolio_prices.empty:
            # Salvar dados
            collector.save_optimized_data(portfolio_prices, returns_simple, returns_log)
            
            print(f"\n🎯 PORTFÓLIO PRONTO PARA ANÁLISE!")
            print(f"💼 Composição: {list(portfolio_prices.columns)}")
            print(f"📈 Período: {(portfolio_prices.index[-1] - portfolio_prices.index[0]).days} dias")
            
            # Estatísticas rápidas
            print(f"\n📊 ESTATÍSTICAS RÁPIDAS:")
            for asset in returns_simple.columns:
                ret_anual = returns_simple[asset].mean() * 252
                vol_anual = returns_simple[asset].std() * np.sqrt(252)
                print(f"   {asset}: Retorno {ret_anual:7.2%} | Vol {vol_anual:7.2%}")
            
            print(f"\n{'='*50}")
            print("🚀 PRÓXIMA ETAPA: Execute python src/metrics/risk_calculator.py")
            print(f"{'='*50}")
            
        else:
            print("❌ Problema ao criar portfólio")
    else:
        print("❌ Falha na coleta de dados")

if __name__ == "__main__":
    main()
