import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import sys
import os

# Adicionar o src ao path para importar nossos módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Análise de Risco de Portfólio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎯 Dashboard de Análise de Risco - Portfólio Macro")
st.markdown("---")

# Função para carregar dados
@st.cache_data
def load_data():
    """Carrega os dados do portfólio"""
    try:
        returns = pd.read_parquet('data/processed/macro_portfolio_returns.parquet')
        prices = pd.read_parquet('data/processed/macro_portfolio_prices.parquet')
        return returns, prices
    except:
        try:
            returns = pd.read_csv('data/processed/macro_portfolio_returns.csv', index_col=0, parse_dates=True)
            prices = pd.read_csv('data/processed/macro_portfolio_prices.csv', index_col=0, parse_dates=True)
            return returns, prices
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {e}")
            return None, None

# Carregar dados
returns, prices = load_data()

if returns is None or prices is None:
    st.error("""
    ⚠️ **Dados não encontrados!**
    
    Execute primeiro o pipeline completo:
    ```bash
    python src/etl/data_collector_bcb.py
    python src/metrics/risk_calculator.py
    ```
    """)
    st.stop()

# Sidebar para controles
st.sidebar.title("⚙️ Controles de Análise")

# Seleção de ativos
st.sidebar.subheader("📈 Seleção de Ativos")
selected_assets = st.sidebar.multiselect(
    "Selecione os ativos para análise:",
    options=returns.columns.tolist(),
    default=returns.columns.tolist()[:4]  # Primeiros 4 por padrão
)

# Filtro de período
st.sidebar.subheader("📅 Período de Análise")
min_date = returns.index.min().date()
max_date = returns.index.max().date()

start_date = st.sidebar.date_input(
    "Data inicial:",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "Data final:",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# Parâmetros de risco
st.sidebar.subheader("🎯 Parâmetros de Risco")
risk_free_rate = st.sidebar.slider(
    "Taxa Livre de Risco (% ao ano):",
    min_value=0.0,
    max_value=20.0,
    value=11.75,
    step=0.1
) / 100

confidence_level = st.sidebar.slider(
    "Nível de Confiança para VaR:",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01
)

# Filtrar dados baseado na seleção
if selected_assets:
    returns_filtered = returns[selected_assets].loc[str(start_date):str(end_date)]
    prices_filtered = prices[selected_assets].loc[str(start_date):str(end_date)]
else:
    returns_filtered = returns.loc[str(start_date):str(end_date)]
    prices_filtered = prices.loc[str(start_date):str(end_date)]

# Layout principal
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Visão Geral", 
    "📊 Métricas de Risco", 
    "🎲 Simulações", 
    "🔍 Análise Detalhada"
])

with tab1:
    st.header("Visão Geral do Portfólio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de preços
        st.subheader("Evolução dos Preços (Base 100)")
        fig_prices = go.Figure()
        
        for asset in prices_filtered.columns:
            fig_prices.add_trace(go.Scatter(
                x=prices_filtered.index,
                y=prices_filtered[asset],
                name=asset,
                line=dict(width=2)
            ))
        
        fig_prices.update_layout(
            height=400,
            xaxis_title="Data",
            yaxis_title="Preço (Base 100)",
            showlegend=True
        )
        st.plotly_chart(fig_prices, use_container_width=True)
    
    with col2:
        # Gráfico de retornos acumulados
        st.subheader("Retornos Acumulados")
        cumulative_returns = (1 + returns_filtered).cumprod()
        
        fig_cumulative = go.Figure()
        
        for asset in cumulative_returns.columns:
            fig_cumulative.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[asset],
                name=asset,
                line=dict(width=2)
            ))
        
        fig_cumulative.update_layout(
            height=400,
            xaxis_title="Data",
            yaxis_title="Retorno Acumulado",
            showlegend=True
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)

with tab2:
    st.header("Métricas de Risco Detalhadas")
    
    # Calcular métricas básicas
    def calculate_metrics(returns_series, risk_free):
        annual_return = returns_series.mean() * 252
        annual_vol = returns_series.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns_series).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # VaR e CVaR
        var = np.percentile(returns_series, (1 - confidence_level) * 100)
        cvar = returns_series[returns_series <= var].mean()
        
        return {
            'Retorno Anual': annual_return,
            'Volatilidade Anual': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            f'VaR {confidence_level:.0%}': var,
            f'CVaR {confidence_level:.0%}': cvar
        }
    
    # Métricas por ativo
    metrics_data = []
    for asset in returns_filtered.columns:
        asset_metrics = calculate_metrics(returns_filtered[asset].dropna(), risk_free_rate)
        asset_metrics['Ativo'] = asset
        metrics_data.append(asset_metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.set_index('Ativo')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Métricas por Ativo")
        st.dataframe(metrics_df.style.format({
            'Retorno Anual': '{:.2%}',
            'Volatilidade Anual': '{:.2%}',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown': '{:.2%}',
            f'VaR {confidence_level:.0%}': '{:.2%}',
            f'CVaR {confidence_level:.0%}': '{:.2%}'
        }))
    
    with col2:
        st.subheader("Matriz de Correlação")
        corr_matrix = returns_filtered.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Correlação entre Ativos'
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.header("Simulações de Monte Carlo")
    
    st.info("""
    🎲 **Simulação de Monte Carlo**
    - Gera milhares de cenários possíveis
    - Calcula probabilidades de ganho/perda
    - Testa resiliência do portfólio
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_simulations = st.slider(
            "Número de Simulações:",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )
        
        time_horizon = st.slider(
            "Horizonte (dias):",
            min_value=30,
            max_value=730,
            value=252,
            step=30
        )
        
        initial_investment = st.number_input(
            "Investimento Inicial (R$):",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
    
    with col2:
        if st.button("🔄 Executar Simulação", type="primary"):
            with st.spinner("Executando simulações de Monte Carlo..."):
                # Simulação simples para demonstração
                portfolio_returns = returns_filtered.mean(axis=1)
                mean_return = portfolio_returns.mean()
                std_return = portfolio_returns.std()
                
                # Gerar simulações
                simulations = np.random.normal(
                    mean_return, 
                    std_return, 
                    (time_horizon, num_simulations)
                )
                
                # Calcular caminhos de preços
                paths = np.zeros((time_horizon + 1, num_simulations))
                paths[0] = initial_investment
                
                for t in range(1, time_horizon + 1):
                    paths[t] = paths[t-1] * (1 + simulations[t-1])
                
                # Métricas das simulações
                final_values = paths[-1]
                returns_simulated = (final_values / initial_investment - 1)
                
                # Exibir resultados
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Valor Médio Final", 
                        f"R$ {np.mean(final_values):,.0f}",
                        delta=f"{np.mean(returns_simulated):.1%}"
                    )
                
                with col2:
                    prob_loss = (returns_simulated < 0).mean()
                    st.metric(
                        "Probabilidade de Prejuízo",
                        f"{prob_loss:.1%}",
                        delta="Risco" if prob_loss > 0.1 else "Baixo Risco"
                    )
                
                with col3:
                    var_95 = np.percentile(returns_simulated, 5)
                    st.metric(
                        f"VaR {confidence_level:.0%}",
                        f"{var_95:.1%}",
                        delta="Perda Máxima Esperada"
                    )
                
                # Gráfico das simulações
                fig_sim = go.Figure()
                
                # Plotar alguns caminhos
                for i in range(min(100, num_simulations)):
                    fig_sim.add_trace(go.Scatter(
                        x=list(range(time_horizon + 1)),
                        y=paths[:, i],
                        mode='lines',
                        line=dict(width=1, color='lightblue'),
                        showlegend=False
                    ))
                
                # Linha da mediana
                fig_sim.add_trace(go.Scatter(
                    x=list(range(time_horizon + 1)),
                    y=np.median(paths, axis=1),
                    mode='lines',
                    line=dict(width=3, color='red'),
                    name='Mediana'
                ))
                
                fig_sim.update_layout(
                    title=f"Simulação de Monte Carlo - {num_simulations} Cenários",
                    xaxis_title="Dias",
                    yaxis_title="Valor do Portfólio (R$)",
                    height=400
                )
                
                st.plotly_chart(fig_sim, use_container_width=True)

with tab4:
    st.header("Análise Técnica Detalhada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volatilidade Móvel")
        window = st.slider("Janela para Volatilidade (dias):", 10, 252, 63)
        
        rolling_vol = returns_filtered.rolling(window).std() * np.sqrt(252)
        
        fig_vol = go.Figure()
        for asset in rolling_vol.columns:
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol[asset],
                name=asset,
                line=dict(width=2)
            ))
        
        fig_vol.update_layout(
            height=400,
            title=f"Volatilidade Móvel ({window} dias - Anualizada)",
            xaxis_title="Data",
            yaxis_title="Volatilidade Anualizada"
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        st.subheader("Drawdowns")
        portfolio_returns = returns_filtered.mean(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red', width=1),
            name='Drawdown'
        ))
        
        fig_dd.update_layout(
            height=400,
            title="Drawdown do Portfólio",
            xaxis_title="Data",
            yaxis_title="Drawdown",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig_dd, use_container_width=True)

# Rodapé
st.markdown("---")
st.markdown(
    "**Desenvolvido com ❤️ usando Streamlit | "
    "Dados: Banco Central do Brasil | "
    "Análise de Risco Quantitativa**"
)
