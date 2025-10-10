import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    def __init__(self):
        self.tickers = {
            'Ações': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'WEGE3.SA'],
            'ETFs': ['BOVA11.SA', 'IVVB11.SA', 'SMAL11.SA'],
            'Índices': ['^BVSP', '^GSPC']  # Ibovespa e S&P500
        }
    
    def download_data(self, period="2y", auto_adjust=True):
        """Baixa dados históricos para todos os tickers"""
        print("📥 Iniciando download de dados...")
        
        all_data = {}
        
        for category, ticker_list in self.tickers.items():
            print(f"\n📊 Baixando {category}: {ticker_list}")
            
            for ticker in ticker_list:
                try:
                    # Download dos dados
                    stock = yf.download(ticker, period=period, auto_adjust=auto_adjust)
                    
                    if not stock.empty:
                        # Padronizar colunas
                        if 'Close' in stock.columns:
                            close_col = 'Close'
                        elif 'Adj Close' in stock.columns:
                            close_col = 'Adj Close'
                        else:
                            close_col = stock.columns[0]
                        
                        all_data[ticker] = stock[close_col].rename(ticker)
                        print(f"✅ {ticker}: {len(stock)} períodos")
                    else:
                        print(f"❌ {ticker}: Dados vazios")
                        
                except Exception as e:
                    print(f"❌ Erro em {ticker}: {e}")
        
        # Combinar todos os dados em um DataFrame
        df = pd.DataFrame(all_data)
        df = df.dropna()  # Remover linhas com NaN
        
        print(f"\n🎯 Download concluído! Shape final: {df.shape}")
        return df
    
    def calculate_returns(self, prices_df):
        """Calcula retornos simples e logarítmicos"""
        print("🧮 Calculando retornos...")
        
        # Retornos simples
        returns_simple = prices_df.pct_change().dropna()
        
        # Retornos logarítmicos
        returns_log = np.log(prices_df / prices_df.shift(1)).dropna()
        
        return returns_simple, returns_log
    
    def save_data(self, prices_df, returns_simple, returns_log, prefix=""):
        """Salva dados em formato parquet"""
        import os
        
        # Criar diretório se não existir
        os.makedirs('data/processed', exist_ok=True)
        
        # Salvar dados
        prices_df.to_parquet(f'data/processed/{prefix}prices.parquet')
        returns_simple.to_parquet(f'data/processed/{prefix}returns_simple.parquet')
        returns_log.to_parquet(f'data/processed/{prefix}returns_log.parquet')
        
        print("💾 Dados salvos em data/processed/")

def main():
    collector = DataCollector()
    
    # Baixar dados
    prices = collector.download_data(period="2y")
    
    if prices.empty:
        print("❌ Nenhum dado foi baixado!")
        return
    
    # Calcular retornos
    returns_simple, returns_log = collector.calculate_returns(prices)
    
    # Salvar dados
    collector.save_data(prices, returns_simple, returns_log)
    
    print("\n✅ FASE 1 CONCLUÍDA!")
    print(f"📊 Preços: {prices.shape}")
    print(f"📈 Retornos simples: {returns_simple.shape}")
    print(f"📈 Retornos log: {returns_log.shape}")

if __name__ == "__main__":
    main()