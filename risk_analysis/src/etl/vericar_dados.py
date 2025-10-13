import pandas as pd
import os

print("🔍 VERIFICANDO DADOS BAIXADOS...")

# Verificar se os arquivos existem
files = [
    'data/processed/prices_bcb.parquet',
    'data/processed/returns_simple_bcb.parquet', 
    'data/processed/bcb_raw_data.parquet'
]

for file in files:
    if os.path.exists(file):
        print(f"✅ {file} - EXISTE")
        try:
            data = pd.read_parquet(file)
            print(f"   📊 Shape: {data.shape}")
            print(f"   📅 Período: {data.index[0]} até {data.index[-1]}")
            print(f"   📈 Colunas: {list(data.columns)}")
        except Exception as e:
            print(f"   ❌ Erro ao ler: {e}")
    else:
        print(f"❌ {file} - NÃO ENCONTRADO")
