# 🧠 Projeto A — Análise Quantitativa de Risco de Portfólio

---

## 🎯 Objetivo Geral

Desenvolver um **sistema analítico completo para medição, comparação e explicação de risco em portfólios financeiros**, combinando métricas tradicionais e análises modernas baseadas em **simulação e estatística**.

O propósito é criar uma ferramenta que transforme **dados de mercado em informação estratégica** sobre exposição, volatilidade e robustez de portfólios, auxiliando a tomada de decisão e o desenvolvimento de **estratégias quantitativas**.

---

## 🏦 Contexto e Motivação

Em gestão de investimentos, o **risco** não é apenas uma métrica — é um **fator determinante de sobrevivência e performance**.

A maioria dos analistas ainda depende de **planilhas estáticas ou cálculos simplificados** (como desvio-padrão e correlação fixa), o que limita a visão dinâmica do risco.

Este projeto propõe uma **abordagem moderna e reprodutível**, baseada em **dados históricos, estatística computacional e simulação Monte Carlo**, para oferecer uma visão multifatorial e temporal do risco de portfólios reais.

---

## 🧩 Escopo

O projeto abrangerá:

- 📥 **Coleta automatizada** de dados de preços e indicadores de ativos.  
- 🧮 **Construção de um pipeline** para cálculo de métricas de risco.  
- 📊 **Análise dinâmica** de correlação e volatilidade.  
- 🎲 **Simulação de portfólios e backtests.**  
- 📈 **Visualização e relatório interativo** dos resultados.

---

## 📈 Resultados Esperados

- Identificação de **períodos e fatores de maior exposição ao risco**.  
- **Medidas comparativas** de risco por ativo e portfólio.  
- Visualização clara da **relação risco-retorno** ao longo do tempo.  
- **Framework reutilizável** para testar novas estratégias de alocação.  
- Base sólida para integração com **modelos preditivos de retorno (Projeto B)**.

---

## ⚙️ Etapas de Desenvolvimento

### 🧩 Etapa 1 — Coleta e Preparação dos Dados
**Objetivo:** construir um dataset limpo e padronizado de preços e volumes históricos.  
**Tarefas:**
- Coletar dados de ações, ETFs e índices via `yfinance` (BOVA11, PETR4, VALE3, etc.).
- Normalizar períodos e preencher lacunas.
- Calcular **retornos simples e logarítmicos**.
- Estruturar base em formato **tidy data** (`data`, `ativo`, `preço`, `retorno`).

**Ferramentas:** `pandas`, `numpy`, `yfinance`

---

### 📊 Etapa 2 — Cálculo de Métricas de Risco
**Objetivo:** quantificar o risco de forma estática e dinâmica.  
**Tarefas:**
- Calcular **volatilidade rolling** (30d, 90d, 180d).
- Estimar **Beta e correlação rolling** em relação a um benchmark (Ibovespa).
- Construir **matrizes de covariância dinâmicas**.
- Calcular métricas: **VaR**, **CVaR (Expected Shortfall)**, **Drawdown máximo**, **Sharpe** e **Sortino**.

**Ferramentas:** `numpy`, `scipy.stats`, `riskfolio-lib`, `empyrical`

---

### 🎲 Etapa 3 — Simulação de Portfólios e Backtesting
**Objetivo:** avaliar o comportamento de portfólios sob diferentes estratégias de alocação.  
**Tarefas:**
- Criar portfólios **equal-weight**, **minimum-variance** e **risk-parity**.
- Simular trajetória histórica de **retorno acumulado**.
- Executar **simulações de Monte Carlo** (bootstrapping de retornos).
- Calcular métricas agregadas: **Sharpe**, **Drawdown**, **Calmar**.

**Ferramentas:** `vectorbt`, `backtesting.py`, `riskfolio-lib`

---

### 📉 Etapa 4 — Análise e Visualização Interativa
**Objetivo:** traduzir resultados complexos em **insights claros e intuitivos**.  
**Tarefas:**
- Criar **gráficos de evolução temporal** de risco e retorno.
- Plotar **mapas de calor** de correlação dinâmica.
- Visualizar **contribuições ao risco por ativo** (*Marginal Risk Contribution*).
- Desenvolver **dashboard interativo** em Streamlit.

**Ferramentas:** `matplotlib`, `plotly`, `seaborn`, `streamlit`

---

### 🧾 Etapa 5 — Relatório e Comunicação
**Objetivo:** consolidar os achados em formato acessível e profissional.  
**Tarefas:**
- Elaborar **relatório técnico** (Jupyter Notebook + PDF).  
- Criar **sumário executivo** com principais conclusões.  
- Publicar no GitHub com **README completo e código comentado**.  
- *(Opcional)* Gravar vídeo curto mostrando o uso do dashboard.

---

## 📚 Conclusão Esperada

O projeto resultará em uma **plataforma analítica modular** que permite:

- Medir risco em múltiplas dimensões (**volatilidade, correlação, drawdown, VaR, CVaR**).  
- Visualizar de forma interativa as exposições e vulnerabilidades do portfólio.  
- Servir como base para **integração com modelos de Machine Learning preditivos**, criando um **ecossistema quantitativo completo (ligação com o Projeto B)**.

👉 Em resumo, o sistema **não apenas mede risco, mas explica risco** — traduzindo estatística em decisão.  
Isso reforça seu perfil como um **profissional capaz de unir inteligência analítica e visão financeira**, um diferencial direto para qualquer área de **Research, Risk Management ou Data Science aplicada a Finanças**.

---

## 🗓️ Cronograma Prático de Execução

### **Plano passo-a-passo (curto prazo — execução prática)**

#### **Fase 0 — Setup (feito)**
- Criar estrutura de pastas e arquivos (já entregue).  
- Criar ambiente virtual e instalar dependências.

#### **Fase 1 — ETL robusto (1–3 dias)**
- Implementar download com retries e logging (`yfinance` + `requests` backoff).  
- Normalizar colunas (Close / Adj Close) e padronizar índices datetimes.  
- Salvar `raw CSVs` e criar `data/processed/prices.parquet` com tidy format.

#### **Fase 2 — Métricas base e testes (2–4 dias)**
- Implementar cálculo de **retornos simples e log**.  
- **Rolling volatility**, **rolling correlation**, **cov matrix** (Ledoit-Wolf shrinkage).  
- Calcular **VaR**, **CVaR** (histórico e paramétrico).  
- Criar **unit tests** com dados sintéticos.  
- Implementar **max drawdown**, **time-under-water**, **contribution-to-risk**.

#### **Fase 3 — Backtests e simulações (3–6 dias)**
- Implementar **equal-weight**, **min-variance** (convex optimizer) e **risk-parity**.  
- Simulações **Monte Carlo** e **bootstrapping de retornos**.  
- Incluir **custos de transação** e modelo de *slippage*.  
- Executar **walk-forward evaluation** e **purged rolling CV**.

#### **Fase 4 — Visualização & Report (2–4 dias)**
- Criar **dashboard Streamlit** (seleção de portfólio, período, gráficos dinâmicos).  
- Exportar **PDF one-pager** e **notebook interativo**.  
- Adicionar **painéis explicativos** (“por que esta métrica importa”, limitações).

#### **Fase 5 — Hardening & integração com Projeto B (contínuo)**
- Salvar datasets processados (`features`, `returns`, `clusters`) em `data/processed/`.  
- Definir **API/contrato** para Project B (`get_features(start, end, tickers)`).  
- Adicionar **experiment tracking** (`MLflow`) e **CI básica** (GitHub Actions).

---

## 🧰 Tecnologias e Bibliotecas

| Categoria | Ferramentas |
|------------|--------------|
| Coleta e ETL | `yfinance`, `pandas`, `numpy` |
| Estatística e Risco | `scipy.stats`, `empyrical`, `riskfolio-lib` |
| Simulação e Backtesting | `vectorbt`, `backtesting.py` |
| Visualização | `matplotlib`, `plotly`, `seaborn`, `streamlit` |
| Documentação e Reprodutibilidade | `jupyter`, `mlflow`, `pdfkit` |

---

## 📂 Estrutura Recomendada do Projeto

ProjetoA_Risco/
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── external/
│
├── notebooks/
│ ├── 01_ETL.ipynb
│ ├── 02_Metricas_Risco.ipynb
│ ├── 03_Backtests.ipynb
│ └── 04_Dashboard.ipynb
│
├── src/
│ ├── etl/
│ ├── metrics/
│ ├── simulation/
│ └── visualization/
│
├── dashboard/
│ └── app.py
│
├── reports/
│ ├── figures/
│ └── portfolio_risk_report.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
