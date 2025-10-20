#!/bin/bash
# Script para construir pacote Lambda

mkdir -p package
pip install -r requirements.txt -t package/

# Copiar código
cp risk_pipeline.py package/
cp -r ../../src package/

# Criar zip
cd package
zip -r ../risk-pipeline.zip .
cd ..
