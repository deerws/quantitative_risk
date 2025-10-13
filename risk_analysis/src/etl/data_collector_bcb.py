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
        
        # Códigos SGS do Banco Central para diversos ativos
        self.series_bcb = {
            # Índices de Ações
            'IBOVESPA': 7,                    # Ibovespa
            'IBRX50': 7405,                   # IBRX 50
            'IBRX100': 7406,                  # IBRX 100
            
            # Taxas de Juros
            'SELIC': 11,                      # Taxa SELIC
            'CDI': 12,                        # Taxa CDI
            'IPCA': 433,                      # IPCA Mensal
            'IGPM': 189,                      # IGP-M
            
            # Câmbio
            'USD_BRL': 1,                     # USD/BRL
            'EUR_BRL': 21619,                 # EUR/BRL
            
            # Commodities (aproximações)
            'PETROLEO_BRENT': 20742,          # Petróleo Brent
            'OURO': 21614,                    # Ouro
            
            # Títulos Públicos
            'NTN-B': 4390,                    # Tesouro IPCA+
            'LTF': 4391,                      # Tesouro Fixo
        }
    
    def get_bcb_data(self, codigo, start_date=None, end_date=None, retries=3, backoff=2):
        """Busca dados do Banco Central pela API com retry para IBOVESPA e OURO"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%d/%m/%Y')
        if end_date is None:
            end_date = datetime.now().strftime('%d/%m/%Y')
            
        url = self.base_url.format(codigo)
        params = {
            'formato': 'json',
            'dataInicial': start_date,
            'dataFinal': end_date
        }
        
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()  # Levanta exceção para códigos de erro HTTP
                data = response.json()
                
                if data:
                    df = pd.DataFrame(data)
                    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
                    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                    df = df.set_index('data').sort_index()
                    # Renomear coluna usando o nome da série em vez do código
                    series_name = next((name for name, code in self.series_bcb.items() if code == codigo), str(codigo))
                    return df[['valor']].rename(columns={'valor': series_name})
                else:
                    print(f"⚠️  Sem dados para código {codigo} (tentativa {attempt + 1}/{retries})")
                    if attempt < retries - 1:
                        time.sleep(backoff * (attempt + 1))
                    continue
                
            except Exception as e:
                print(f"❌ Erro no código {codigo} (tentativa {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
                continue
        
        print(f"❌ Falha após {retries} tentativas para código {codigo}")
        return None
    
    def download_data(self):
        """Baixa dados de todas as séries do BCB"""
        print("🏦 Iniciando download de dados do Banco Central...")
        
        all_data = []
        
        for nome, codigo in self.series_bcb.items():
            print(f"📥 Baixando {nome} (código {codigo})...")
            data = self.get_bcb_data(codigo)
            
            if data is not None and not data.empty:
                all_data.append(data)
                print(f"✅ {nome}: {len(data)} períodos")
            else:
                print(f"❌ Falha em {nome}")
        
        if all_data:
            # Combinar todos os dados
            combined_df = pd.concat(all_data, axis=1)
            combined_df = combined_df.dropna()
            
            print(f"\n🎯 Download concluído! Shape final: {combined_df.shape}")
            return combined_df
        else:
            print("❌ Nenhum dado foi baixado!")
            return pd.DataFrame()
    
    def create_portfolio_returns(self, prices_df):
        """Cria retornos de portfólio a partir dos dados do BCB"""
        print("📊 Transformando dados em retornos de portfólio...")
        
        # Verificar quais colunas temos disponíveis
        print(f"📋 Colunas disponíveis: {list(prices_df.columns)}")
        
        # Vamos criar um portfólio com o que temos
        # Priorizar: Câmbio, Taxas, Commodities
        available_components = []
        
        if 'USD_BRL' in prices_df.columns:
            available_components.append('USD_BRL')  # Exposição cambial
        if 'SELIC' in prices_df.columns:
            available_components.append('SELIC')    # Taxa livre de risco
        if 'CDI' in prices_df.columns:
            available_components.append('CDI')      # Taxa de juros
        if 'PETROLEO_BRENT' in prices_df.columns:
            available_components.append('PETROLEO_BRENT')  # Commodity
        
        # Se tivermos poucos componentes, usar mais
        if len(available_components) < 3:
            additional = [col for col in prices_df.columns if col not in available_components]
            available_components.extend(additional[:3-len(available_components)])
        
        print(f"🎯 Componentes selecionados: {available_components}")
        
        if len(available_components) < 2:
            print("❌ Componentes insuficientes para criar portfólio")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        portfolio_prices = prices_df[available_components].copy()
        
        # Preencher valores missing com forward fill
        portfolio_prices = portfolio_prices.ffill().dropna()
        
        if portfolio_prices.empty:
            print("❌ Portfolio vazio após limpeza")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        print(f"📊 Portfolio final: {portfolio_prices.shape}")
        
        # Normalizar para base 100 (evita problemas de escala)
        portfolio_normalized = portfolio_prices / portfolio_prices.iloc[0] * 100
        
        # Calcular retornos
        returns_simple = portfolio_normalized.pct_change().dropna()
        returns_log = np.log(portfolio_normalized / portfolio_normalized.shift(1)).dropna()
        
        print(f"📈 Retornos calculados: {returns_simple.shape}")
        
        return portfolio_normalized, returns_simple, returns_log
    
    def save_data(self, prices_df, returns_simple, returns_log):
        """Salva dados"""
        os.makedirs('data/processed', exist_ok=True)
        
        prices_df.to_parquet('data/processed/prices_bcb.parquet')
        returns_simple.to_parquet('data/processed/returns_simple_bcb.parquet')
        returns_log.to_parquet('data/processed/returns_log_bcb.parquet')
        
        # Salvar também como CSV
        prices_df.to_csv('data/processed/prices_bcb.csv')
        returns_simple.to_csv('data/processed/returns_simple_bcb.csv')
        
        print("💾 Dados salvos em data/processed/")

def main():
    collector = BCBDataCollector()
    
    # Baixar dados do BCB
    prices = collector.download_data()
    
    if not prices.empty:
        print("\n✅ DADOS REAIS OBTIDOS DO BANCO CENTRAL!")
        print(f"📊 Shape: {prices.shape}")
        print(f"📅 Período: {prices.index[0]} até {prices.index[-1]}")
        print(f"📈 Séries obtidas: {list(prices.columns)}")
        
        # Estatísticas básicas
        print("\n📋 Estatísticas descritivas:")
        print(prices.describe())
        
        # Criar retornos de portfólio
        portfolio_prices, returns_simple, returns_log = collector.create_portfolio_returns(prices)
        
        if not portfolio_prices.empty:
            collector.save_data(portfolio_prices, returns_simple, returns_log)
            
            print("\n🎯 PORTFÓLIO CRIADO COM SUCESSO!")
            print(f"📊 Componentes do portfólio: {list(portfolio_prices.columns)}")
            print(f"📈 Período do portfólio: {portfolio_prices.index[0]} até {portfolio_prices.index[-1]}")
            
            print("\n📊 Estatísticas dos retornos:")
            print(returns_simple.describe())
            
            # Salvar também os dados brutos para análise
            prices.to_parquet('data/processed/bcb_raw_data.parquet')
            prices.to_csv('data/processed/bcb_raw_data.csv')
            
            print("\n💾 Todos os dados salvos em data/processed/")
            print("🚀 AGORA PODEMOS AVANÇAR PARA AS ANÁLISES DE RISCO!")
            
        else:
            print("❌ Não foi possível criar portfólio, mas temos dados brutos salvos")
            # Salvar dados brutos mesmo sem portfólio
            prices.to_parquet('data/processed/bcb_raw_data.parquet')
            prices.to_csv('data/processed/bcb_raw_data.csv')
            
    else:
        print("❌ Falha completa no download dos dados")

if __name__ == "__main__":
    main()
