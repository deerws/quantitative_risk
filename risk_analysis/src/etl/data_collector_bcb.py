import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import time
warnings.filterwarnings('ignore')

class DataCollector:
    def __init__(self):
        self.tickers = [
            'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'WEGE3.SA',
            'BOVA11.SA', 'IVVB11.SA', 'SMAL11.SA',
            '^BVSP', '^GSPC'
        ]
    
    def download_data_robust(self, start_date=None, end_date=None, period="2y"):
        """Método ROBUSTO com datas explícitas e retry"""
        print("📥 Iniciando download ROBUSTO de dados...")
        
        # Se não forneceu datas, calcula automaticamente
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=730)  # 2 anos
        
        print(f"📅 Período: {start_date.date()} até {end_date.date()}")
        
        prices_dict = {}
        
        for ticker in self.tickers:
            print(f"\n📥 Tentando {ticker}...")
            
            # Tenta 3 vezes com delay
            for attempt in range(3):
                try:
                    # Usa start e end explícitos
                    stock = yf.Ticker(ticker)
                    hist = stock.history(
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,
                        actions=False  # Não baixa dividendos/splits
                    )
                    
                    if not hist.empty and len(hist) > 10:
                        prices_dict[ticker] = hist['Close']
                        print(f"✅ {ticker}: {len(hist)} dias de dados")
                        break
                    else:
                        print(f"⚠️  {ticker}: Dados insuficientes (tentativa {attempt+1}/3)")
                        time.sleep(2)
                        
                except Exception as e:
                    print(f"❌ {ticker}: Erro na tentativa {attempt+1}/3 - {str(e)[:50]}")
                    time.sleep(2)
            else:
                print(f"❌ {ticker}: FALHOU após 3 tentativas")
        
        if not prices_dict:
            print("\n❌ NENHUM ticker foi baixado com sucesso!")
            return pd.DataFrame()
        
        # Monta DataFrame
        prices_df = pd.DataFrame(prices_dict)
        
        # Remove dias sem dados para TODOS os ativos
        prices_df = prices_df.dropna()
        
        print(f"\n✅ Download concluído!")
        print(f"📊 Ativos baixados: {len(prices_dict)}/{len(self.tickers)}")
        print(f"📊 Shape final: {prices_df.shape}")
        print(f"📅 Data inicial: {prices_df.index[0].date()}")
        print(f"📅 Data final: {prices_df.index[-1].date()}")
        
        return prices_df
    
    def calculate_returns(self, prices_df):
        """Calcula retornos simples e logarítmicos"""
        print("\n🧮 Calculando retornos...")
        
        # Retornos simples
        returns_simple = prices_df.pct_change().dropna()
        
        # Retornos logarítmicos
        returns_log = np.log(prices_df / prices_df.shift(1)).dropna()
        
        print(f"✅ Retornos calculados: {returns_simple.shape}")
        
        return returns_simple, returns_log
    
    def save_data(self, prices_df, returns_simple, returns_log):
        """Salva dados em formato parquet e CSV (backup)"""
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/raw', exist_ok=True)
        
        # Salva parquet (mais eficiente)
        prices_df.to_parquet('data/processed/prices.parquet')
        returns_simple.to_parquet('data/processed/returns_simple.parquet')
        returns_log.to_parquet('data/processed/returns_log.parquet')
        
        # Salva CSV também (backup legível)
        prices_df.to_csv('data/raw/prices.csv')
        
        print("\n💾 Dados salvos:")
        print("   - data/processed/prices.parquet")
        print("   - data/processed/returns_simple.parquet")
        print("   - data/processed/returns_log.parquet")
        print("   - data/raw/prices.csv (backup)")

def main():
    collector = DataCollector()
    
    # Define datas explicitamente
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 anos
    
    # Baixa dados
    prices = collector.download_data_robust(start_date=start_date, end_date=end_date)
    
    if prices.empty:
        print("\n❌ FALHA TOTAL - Nenhum dado foi baixado!")
        print("\n🔧 Possíveis soluções:")
        print("   1. Verifique sua conexão com internet")
        print("   2. Tente novamente em alguns minutos (Yahoo Finance pode estar fora)")
        print("   3. Teste manualmente: yf.Ticker('PETR4.SA').history(period='1mo')")
        return
    
    # Calcular retornos
    returns_simple, returns_log = collector.calculate_returns(prices)
    
    # Salvar dados
    collector.save_data(prices, returns_simple, returns_log)
    
    # Estatísticas finais
    print("\n" + "="*60)
    print("✅ FASE 1 CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print(f"📊 Preços: {prices.shape[0]} dias × {prices.shape[1]} ativos")
    print(f"📈 Retornos: {returns_simple.shape[0]} dias × {returns_simple.shape[1]} ativos")
    
    print("\n📋 Preview dos dados:")
    print(prices.tail())
    
    print("\n📊 Estatísticas básicas:")
    print(returns_simple.describe())

if __name__ == "__main__":
    main()
