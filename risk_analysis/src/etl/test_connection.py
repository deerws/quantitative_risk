import yfinance as yf
import requests
from datetime import datetime, timedelta
import socket

print("="*60)
print("🔍 DIAGNÓSTICO DE CONEXÃO - Yahoo Finance")
print("="*60)

# Teste 1: Internet básica
print("\n1️⃣ Testando conexão básica com internet...")
try:
    response = requests.get("https://www.google.com", timeout=5)
    print(f"✅ Internet funcionando (status: {response.status_code})")
except Exception as e:
    print(f"❌ Sem internet: {e}")

# Teste 2: Acesso ao Yahoo Finance
print("\n2️⃣ Testando acesso ao Yahoo Finance...")
try:
    response = requests.get("https://finance.yahoo.com", timeout=10)
    print(f"✅ Yahoo Finance acessível (status: {response.status_code})")
except Exception as e:
    print(f"❌ Yahoo Finance bloqueado: {e}")

# Teste 3: API do Yahoo Finance
print("\n3️⃣ Testando API do yfinance...")
try:
    response = requests.get(
        "https://query2.finance.yahoo.com/v8/finance/chart/PETR4.SA",
        timeout=10
    )
    print(f"✅ API Yahoo acessível (status: {response.status_code})")
except Exception as e:
    print(f"❌ API Yahoo bloqueada: {e}")

# Teste 4: Download via yfinance (ticker brasileiro)
print("\n4️⃣ Testando download PETR4.SA...")
try:
    ticker = yf.Ticker("PETR4.SA")
    hist = ticker.history(period="5d")
    if not hist.empty:
        print(f"✅ PETR4.SA funcionando ({len(hist)} dias)")
        print(hist.tail(2))
    else:
        print("❌ PETR4.SA retornou vazio")
except Exception as e:
    print(f"❌ Erro em PETR4.SA: {e}")

# Teste 5: Download via yfinance (ticker americano)
print("\n5️⃣ Testando download AAPL (EUA)...")
try:
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="5d")
    if not hist.empty:
        print(f"✅ AAPL funcionando ({len(hist)} dias)")
        print(hist.tail(2))
    else:
        print("❌ AAPL retornou vazio")
except Exception as e:
    print(f"❌ Erro em AAPL: {e}")

# Teste 6: Índice brasileiro
print("\n6️⃣ Testando índice ^BVSP...")
try:
    ticker = yf.Ticker("^BVSP")
    hist = ticker.history(period="5d")
    if not hist.empty:
        print(f"✅ ^BVSP funcionando ({len(hist)} dias)")
        print(hist.tail(2))
    else:
        print("❌ ^BVSP retornou vazio")
except Exception as e:
    print(f"❌ Erro em ^BVSP: {e}")

# Teste 7: Verificar DNS
print("\n7️⃣ Testando resolução DNS...")
try:
    ip = socket.gethostbyname("finance.yahoo.com")
    print(f"✅ DNS resolvendo: finance.yahoo.com -> {ip}")
except Exception as e:
    print(f"❌ Falha no DNS: {e}")

# Teste 8: Proxy
print("\n8️⃣ Verificando configuração de proxy...")
import os
http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
if http_proxy or https_proxy:
    print(f"⚠️  Proxy detectado:")
    print(f"   HTTP: {http_proxy}")
    print(f"   HTTPS: {https_proxy}")
else:
    print("✅ Nenhum proxy configurado")

print("\n" + "="*60)
print("🏁 DIAGNÓSTICO CONCLUÍDO")
print("="*60)
