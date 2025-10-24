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

## 📂 Project Structure

```
ProjetoA_Risco/
│
├── data/
│   ├── raw/                    # Raw market data
│   ├── processed/              # Cleaned and transformed data
│   └── external/               # External datasets
│
├── notebooks/
│   ├── 01_ETL.ipynb           # Data extraction and transformation
│   ├── 02_Risk_Metrics.ipynb  # Risk calculations
│   ├── 03_Backtests.ipynb     # Portfolio simulations
│   └── 04_Dashboard.ipynb     # Interactive visualizations
│
├── src/
│   ├── etl/                   # Data pipeline modules
│   ├── metrics/               # Risk calculation functions
│   ├── simulation/            # Backtesting and Monte Carlo
│   └── visualization/         # Plotting utilities
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard
│
├── reports/
│   ├── figures/               # Generated plots
│   └── portfolio_risk_report.pdf
│
├── tests/                     # Unit tests
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ProjetoA_Risco.git
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

## 🗓️ Development Roadmap

### Phase 1: ETL Pipeline ✅
- [x] Automated data download with retry logic
- [x] Data normalization and cleaning
- [x] Tidy data format implementation

### Phase 2: Core Metrics 🔄
- [x] Return calculations (simple and logarithmic)
- [x] Rolling statistics
- [ ] Advanced covariance estimation (Ledoit-Wolf)
- [ ] Complete VaR/CVaR implementation

### Phase 3: Backtesting Framework 📋
- [ ] Portfolio optimization algorithms
- [ ] Monte Carlo simulation engine
- [ ] Transaction cost modeling
- [ ] Walk-forward analysis

### Phase 4: Visualization & Reporting 📋
- [ ] Streamlit dashboard development
- [ ] PDF report generation
- [ ] Interactive plots with Plotly

### Phase 5: Integration & Production 📋
- [ ] API development for Project B integration
- [ ] MLflow experiment tracking
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Docker containerization

---

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

---

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

- **[User Guide](docs/user_guide.md)** — How to use the system
- **[API Reference](docs/api_reference.md)** — Function documentation
- **[Methodology](docs/methodology.md)** — Risk calculation explanations
- **[Examples](docs/examples.md)** — Use case scenarios

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**André Pinheiro Paes**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Financial data provided by [Yahoo Finance](https://finance.yahoo.com/)
- Inspired by modern portfolio theory and quantitative finance literature
- Built with open-source tools and libraries

---

## 📧 Contact

For questions or suggestions, please open an issue or contact:
- Email: your.email@example.com
- Project Link: [https://github.com/yourusername/ProjetoA_Risco](https://github.com/yourusername/ProjetoA_Risco)

---

## 🔗 Related Projects

This project is part of a quantitative finance ecosystem:

- **Project B** — Predictive Models for Portfolio Returns (coming soon)
- Integration with machine learning frameworks for portfolio optimization

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/ProjetoA_Risco&type=Date)](https://star-history.com/#yourusername/ProjetoA_Risco&Date)

---

**Made with ❤️ and Python**
