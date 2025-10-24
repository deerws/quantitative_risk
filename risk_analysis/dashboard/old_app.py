import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys
import os
from datetime import datetime, timedelta

# Adicionar o src ao path para importar nossos módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Quantum Risk Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .simulation-card {
        background-color: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🧠 Quantum Risk Analytics</h1>', unsafe_allow_html=True)
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
st.sidebar.title("⚙️ Painel de Controle")

# Seleção de ativos
st.sidebar.subheader("📈 Configuração do Portfólio")
selected_assets = st.sidebar.multiselect(
    "Selecione os ativos:",
    options=returns.columns.tolist(),
    default=returns.columns.tolist()[:4]
)

# Pesos personalizados
st.sidebar.subheader("⚖️ Alocação de Pesos")
weights = {}
if selected_assets:
    st.sidebar.write("Defina os pesos (%) para cada ativo:")
    total_weight = 0
    for asset in selected_assets:
        weight = st.sidebar.slider(f"{asset}", 0, 100, 100//len(selected_assets), key=f"weight_{asset}")
        weights[asset] = weight / 100
        total_weight += weight
    
    if abs(total_weight - 100) > 1:
        st.sidebar.warning(f"⚠️ Pesos totalizam {total_weight}% - ajuste para 100%")

# Filtro de período
st.sidebar.subheader("📅 Período de Análise")
min_date = returns.index.min().date()
max_date = returns.index.max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Início", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("Fim", value=max_date, min_value=min_date, max_value=max_date)

# Parâmetros de risco
st.sidebar.subheader("🎯 Parâmetros de Risco")
risk_free_rate = st.sidebar.slider("Taxa Livre de Risco (% a.a):", 0.0, 20.0, 11.75, 0.1) / 100
confidence_level = st.sidebar.slider("Confiança VaR:", 0.90, 0.99, 0.95, 0.01)

# Configuração de Simulações
st.sidebar.subheader("🎲 Configuração de Simulações")
num_simulations = st.sidebar.selectbox("Nº de Simulações:", [500, 1000, 2000, 5000], index=1)
time_horizon = st.sidebar.selectbox("Horizonte (dias):", [30, 90, 180, 252, 504], index=3)
initial_investment = st.sidebar.number_input("Investimento Inicial (R$):", 1000, 1000000, 10000, 1000)

# Filtrar dados
if selected_assets:
    returns_filtered = returns[selected_assets].loc[str(start_date):str(end_date)]
    prices_filtered = prices[selected_assets].loc[str(start_date):str(end_date)]
else:
    returns_filtered = returns.loc[str(start_date):str(end_date)]
    prices_filtered = prices.loc[str(start_date):str(end_date)]

# Calcular retorno do portfólio com pesos
if weights and len(weights) == len(selected_assets):
    # Normalizar pesos
    weight_sum = sum(weights.values())
    normalized_weights = {k: v/weight_sum for k, v in weights.items()}
    portfolio_returns = (returns_filtered * pd.Series(normalized_weights)).sum(axis=1)
else:
    # Pesos iguais
    portfolio_returns = returns_filtered.mean(axis=1)

# Layout principal com abas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Dashboard", 
    "📊 Análise de Risco", 
    "🎲 Simulações Avançadas",
    "📈 Análise Técnica", 
    "📋 Relatório"
])

with tab1:
    st.header("📈 Visão Geral do Portfólio")
    
    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = (1 + portfolio_returns).prod() - 1
        st.metric("Retorno Total", f"{total_return:.2%}")
    
    with col2:
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        st.metric("Volatilidade Anual", f"{annual_vol:.2%}")
    
    with col3:
        sharpe = (portfolio_returns.mean() * 252 - risk_free_rate) / annual_vol
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col4:
        var_95 = np.percentile(portfolio_returns, 5)
        st.metric(f"VaR {confidence_level:.0%}", f"{var_95:.2%}")
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evolução dos Preços")
        fig_prices = go.Figure()
        for asset in prices_filtered.columns:
            fig_prices.add_trace(go.Scatter(
                x=prices_filtered.index,
                y=prices_filtered[asset],
                name=asset,
                line=dict(width=2)
            ))
        fig_prices.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_prices, use_container_width=True)
    
    with col2:
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
        fig_cumulative.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Composição do portfólio
    st.subheader("📊 Composição do Portfólio")
    if weights:
        fig_pie = px.pie(
            values=list(normalized_weights.values()),
            names=list(normalized_weights.keys()),
            title="Alocação do Portfólio"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.header("📊 Análise Detalhada de Risco")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Métricas por Ativo")
        
        def calculate_asset_metrics(returns_series, risk_free):
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
            
            return [annual_return, annual_vol, sharpe, max_drawdown, var, cvar]
        
        metrics_data = []
        for asset in returns_filtered.columns:
            metrics = calculate_asset_metrics(returns_filtered[asset].dropna(), risk_free_rate)
            metrics_data.append([asset] + metrics)
        
        metrics_df = pd.DataFrame(
            metrics_data,
            columns=['Ativo', 'Retorno Anual', 'Volatilidade', 'Sharpe', 'Max DD', f'VaR {confidence_level:.0%}', f'CVaR {confidence_level:.0%}']
        ).set_index('Ativo')
        
        st.dataframe(metrics_df.style.format({
            'Retorno Anual': '{:.2%}',
            'Volatilidade': '{:.2%}',
            'Sharpe': '{:.2f}',
            'Max DD': '{:.2%}',
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
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Análise de Drawdown
    st.subheader("📉 Análise de Drawdown")
    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_portfolio.expanding().max()
    drawdown_series = (cumulative_portfolio - rolling_max) / rolling_max
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red', width=1),
        name='Drawdown'
    ))
    fig_dd.update_layout(
        height=300,
        title="Drawdown do Portfólio",
        yaxis_tickformat='.1%'
    )
    st.plotly_chart(fig_dd, use_container_width=True)

with tab3:
    st.header("🎲 Simulações Avançadas de Monte Carlo")
    
    st.info("""
    **Algoritmos Disponíveis:**
    - 🎲 **Monte Carlo Clássico**: Simulação básica com distribuição normal
    - 🔄 **Bootstrapping**: Reamostragem dos dados históricos
    - 🌊 **Merton Jump**: Inclui saltos para eventos extremos
    - 📈 **GARCH**: Volatilidade variável no tempo
    - 🎯 **Cópula Gaussiana**: Modelagem de dependência multivariada
    """)
    
    # Seleção do algoritmo
    algorithm = st.selectbox(
        "Selecione o algoritmo de simulação:",
        [
            "Monte Carlo Clássico",
            "Bootstrapping", 
            "Merton Jump Diffusion",
            "GARCH",
            "Cópula Gaussiana"
        ]
    )
    
    if st.button("🚀 Executar Simulação", type="primary"):
        with st.spinner(f"Executando {algorithm}..."):
            try:
                # Importar e executar simulação baseada na seleção
                from src.simulation.advanced_simulators import AdvancedRiskSimulators
                
                simulator = AdvancedRiskSimulators(returns_filtered)
                
                if algorithm == "Monte Carlo Clássico":
                    paths, metrics = simulator.monte_carlo_baseline(
                        initial_investment, num_simulations, time_horizon
                    )
                elif algorithm == "Bootstrapping":
                    paths, metrics = simulator.historical_bootstrapping(num_simulations)
                elif algorithm == "Merton Jump Diffusion":
                    paths, metrics = simulator.merton_jump_diffusion(
                        initial_investment, time_horizon/252, num_simulations
                    )
                elif algorithm == "GARCH":
                    paths, metrics = simulator.garch_simulation(num_simulations)
                elif algorithm == "Cópula Gaussiana":
                    sim_df, metrics = simulator.gaussian_copula_simulation(num_simulations)
                    # Para cópula, usar portfolio simulado
                    portfolio_simulated = sim_df.mean(axis=1)
                    paths = np.array([portfolio_simulated.values * initial_investment]).T
                
                # Exibir resultados
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Valor Médio Final", 
                        f"R$ {metrics.get('Retorno_Medio_Simulado', 0) * initial_investment + initial_investment:,.0f}",
                        delta=f"{metrics.get('Retorno_Medio_Simulado', 0):.1%}"
                    )
                
                with col2:
                    prob_loss = metrics.get('Prob_Prejuizo', 0)
                    st.metric(
                        "Prob. Prejuízo",
                        f"{prob_loss:.1%}",
                        delta="Alto Risco" if prob_loss > 0.2 else "Baixo Risco",
                        delta_color="inverse"
                    )
                
                with col3:
                    var_95 = metrics.get('VaR_95%', 0)
                    st.metric(
                        f"VaR {confidence_level:.0%}",
                        f"R$ {abs(var_95 * initial_investment):,.0f}",
                        delta="Perda Esperada"
                    )
                
                with col4:
                    if 'Volatilidade_Simulada' in metrics:
                        st.metric(
                            "Volatilidade Simulada",
                            f"{metrics['Volatilidade_Simulada']:.1%}",
                            delta="Anualizada"
                        )
                
                # Gráfico das simulações
                st.subheader(f"Simulação - {algorithm}")
                
                fig_sim = go.Figure()
                
                # Plotar alguns caminhos (máximo 100 para performance)
                n_paths_to_plot = min(100, num_simulations)
                for i in range(n_paths_to_plot):
                    fig_sim.add_trace(go.Scatter(
                        x=list(range(len(paths))),
                        y=paths[:, i] if len(paths.shape) > 1 else paths,
                        mode='lines',
                        line=dict(width=1, color='lightblue'),
                        showlegend=False
                    ))
                
                # Linha da mediana
                if len(paths.shape) > 1:
                    median_path = np.median(paths, axis=1)
                else:
                    median_path = paths
                    
                fig_sim.add_trace(go.Scatter(
                    x=list(range(len(median_path))),
                    y=median_path,
                    mode='lines',
                    line=dict(width=3, color='red'),
                    name='Mediana'
                ))
                
                fig_sim.update_layout(
                    title=f"{algorithm} - {num_simulations} Simulações",
                    xaxis_title="Dias",
                    yaxis_title="Valor do Portfólio (R$)",
                    height=400
                )
                
                st.plotly_chart(fig_sim, use_container_width=True)
                
                # Distribuição dos valores finais
                if len(paths.shape) > 1:
                    final_values = paths[-1]
                else:
                    final_values = paths
                    
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=final_values,
                    nbinsx=50,
                    name='Distribuição Final',
                    opacity=0.7
                ))
                
                fig_dist.add_vline(
                    x=np.mean(final_values), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Média"
                )
                
                fig_dist.update_layout(
                    title="Distribuição dos Valores Finais",
                    xaxis_title="Valor Final (R$)",
                    yaxis_title="Frequência",
                    height=300
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Erro na simulação: {e}")
                st.info("💡 Tente instalar as dependências: `pip install arch statsmodels`")

with tab4:
    st.header("📈 Análise Técnica Avançada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volatilidade Móvel")
        vol_window = st.slider("Janela Volatilidade:", 10, 252, 63)
        
        rolling_vol = returns_filtered.rolling(vol_window).std() * np.sqrt(252)
        
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
            title=f"Volatilidade Móvel ({vol_window} dias)",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        st.subheader("Retornos vs Volatilidade")
        annual_returns = returns_filtered.mean() * 252
        annual_volatility = returns_filtered.std() * np.sqrt(252)
        
        fig_scatter = px.scatter(
            x=annual_volatility,
            y=annual_returns,
            text=returns_filtered.columns,
            title="Retorno vs Risco"
        )
        
        fig_scatter.update_traces(
            marker=dict(size=20, opacity=0.7),
            textposition='top center'
        )
        
        fig_scatter.update_layout(
            height=400,
            xaxis_title="Volatilidade Anual",
            yaxis_title="Retorno Anual",
            xaxis_tickformat='.1%',
            yaxis_tickformat='.1%'
        )
        
        # Adicionar linha do Sharpe
        max_vol = annual_volatility.max()
        sharpe_line = np.linspace(0, max_vol, 100) * (annual_returns.mean() / annual_volatility.mean())
        fig_scatter.add_trace(go.Scatter(
            x=np.linspace(0, max_vol, 100),
            y=sharpe_line,
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Linha Sharpe'
        ))
        
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab5:
    st.header("📋 Relatório Executivo")
    
    # Gerar relatório automático
    st.subheader("📊 Resumo do Portfólio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
        <h3>🎯 Performance</h3>
        <ul>
        <li><b>Retorno Total:</b> {:.2%}</li>
        <li><b>Retorno Anualizado:</b> {:.2%}</li>
        <li><b>Sharpe Ratio:</b> {:.2f}</li>
        <li><b>Alpha (vs RF):</b> {:.2%}</li>
        </ul>
        </div>
        """.format(
            total_return,
            portfolio_returns.mean() * 252,
            sharpe,
            portfolio_returns.mean() * 252 - risk_free_rate
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h3>📉 Medidas de Risco</h3>
        <ul>
        <li><b>Volatilidade Anual:</b> {:.2%}</li>
        <li><b>Max Drawdown:</b> {:.2%}</li>
        <li><b>VaR 95%:</b> {:.2%}</li>
        <li><b>CVaR 95%:</b> {:.2%}</li>
        </ul>
        </div>
        """.format(
            annual_vol,
            drawdown_series.min(),
            var_95,
            portfolio_returns[portfolio_returns <= var_95].mean()
        ), unsafe_allow_html=True)
    
    # Recomendações
    st.subheader("💡 Recomendações")
    
    if sharpe > 1.0:
        st.success("**✅ Portfólio Eficiente** - Sharpe Ratio acima de 1.0 indica boa relação risco-retorno")
    else:
        st.warning("**⚠️ Oportunidade de Melhoria** - Considere otimizar a alocação para melhorar o Sharpe Ratio")
    
    if annual_vol > 0.20:
        st.warning("**📈 Alta Volatilidade** - Portfólio apresenta risco elevado, considere diversificação")
    else:
        st.success("**✅ Volatilidade Controlada** - Nível de risco dentro de parâmetros conservadores")
    
    # Botão para exportar relatório
    if st.button("📥 Exportar Relatório PDF"):
        st.info("""
        **Recurso em Desenvolvimento**
        
        Em versões futuras, você poderá exportar:
        - Relatório PDF completo
        - Análise detalhada em Excel
        - Gráficos em alta resolução
        - Apresentação executiva
        """)

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <b>🧠 Quantum Risk Analytics</b> | 
    Desenvolvido com Python + Streamlit | 
    Dados: Banco Central do Brasil
    </div>
    """, 
    unsafe_allow_html=True
)
