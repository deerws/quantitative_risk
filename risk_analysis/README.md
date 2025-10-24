# 🧠 Project A — Quantitative Portfolio Risk Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

---

## 🎯 Overview

A **comprehensive analytical system for measuring, comparing, and explaining risk in financial portfolios**, combining traditional metrics with modern simulation-based and statistical analysis.

The goal is to create a tool that transforms **market data into strategic insights** about exposure, volatility, and portfolio robustness, supporting decision-making and the development of **quantitative strategies**.

---

## 🏦 Context and Motivation

In investment management, **risk** isn't just a metric — it's a **determining factor for survival and performance**.

Most analysts still rely on **static spreadsheets or simplified calculations** (such as standard deviation and fixed correlation), which limits the dynamic view of risk.

This project proposes a **modern and reproducible approach**, based on **historical data, computational statistics, and Monte Carlo simulation**, to offer a multifactorial and temporal view of real portfolio risk.

---

## ✨ Key Features

- 📥 **Automated data collection** of asset prices and indicators
- 🧮 **Complete risk metrics pipeline** (VaR, CVaR, Sharpe, Sortino, Drawdown)
- 📊 **Dynamic correlation and volatility analysis**
- 🎲 **Portfolio simulation and backtesting**
- 📈 **Interactive visualization and reporting**
- 🔄 **Framework ready for integration with predictive models**

---

## 🛠️ Technology Stack

| Category | Tools |
|----------|-------|
| Data Collection & ETL | `yfinance`, `pandas`, `numpy` |
| Statistics & Risk | `scipy.stats`, `empyrical`, `riskfolio-lib` |
| Simulation & Backtesting | `vectorbt`, `backtesting.py` |
| Visualization | `matplotlib`, `plotly`, `seaborn`, `streamlit` |
| Documentation & Reproducibility | `jupyter`, `mlflow`, `pdfkit` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/deerws/quantative_risk.git
cd ProjetoA_Risco
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Run data collection**
```bash
python src/etl/download_data.py
```

2. **Calculate risk metrics**
```bash
jupyter notebook notebooks/02_Risk_Metrics.ipynb
```

3. **Launch interactive dashboard**
```bash
streamlit run dashboard/app.py
```

---

## 📊 Core Metrics Implemented

### Volatility Measures
- **Rolling Volatility** (30d, 90d, 180d)
- **Historical Volatility**
- **Exponentially Weighted Moving Average (EWMA)**

### Risk-Adjusted Returns
- **Sharpe Ratio** — Risk-adjusted return metric
- **Sortino Ratio** — Downside risk-adjusted return
- **Calmar Ratio** — Return to maximum drawdown

### Value at Risk (VaR)
- **Historical VaR** (95%, 99% confidence)
- **Parametric VaR** (Normal distribution assumption)
- **Conditional VaR (CVaR)** — Expected Shortfall

### Portfolio Risk
- **Beta** — Systematic risk relative to benchmark
- **Maximum Drawdown** — Peak-to-trough decline
- **Correlation Matrix** — Dynamic asset relationships
- **Marginal Risk Contribution** — Individual asset risk impact

---

## 🎲 Portfolio Strategies

The system supports multiple allocation strategies:

1. **Equal Weight** — Simple diversification
2. **Minimum Variance** — Optimized for lowest volatility
3. **Risk Parity** — Balanced risk contribution
4. **Maximum Sharpe** — Optimal risk-return trade-off

Each strategy includes:
- Historical performance backtesting
- Monte Carlo simulation (10,000+ scenarios)
- Transaction cost modeling
- Walk-forward validation

---

## 📈 Visualization Examples

### Risk Dashboard Features
- **Time series analysis** of returns and volatility
- **Correlation heatmaps** with dynamic updates
- **Risk contribution charts** by asset
- **Drawdown analysis** with underwater plots
- **Efficient frontier** visualization
- **Rolling beta and correlation** plots

---

