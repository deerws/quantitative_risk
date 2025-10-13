import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

class AdvancedRiskVisualizer:
    def __init__(self, returns_df, prices_df):
        self.returns = returns_df
        self.prices = prices_df
        self.portfolio_returns = returns_df.mean(axis=1)
        
        # Configuração de tema escuro
        self.set_dark_theme()
    
    def set_dark_theme(self):
        """Configura tema escuro para os gráficos"""
        plt.style.use('dark_background')
        self.colors = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
        
    def create_comprehensive_dashboard(self, save_path=None):
        """Dashboard completo no estilo da sua imagem"""
        fig = plt.figure(figsize=(20, 16))
        
        # Definir layout do grid
        gs = fig.add_gridspec(3, 3)
        
        # 1. Evolução de Preços com Bandas de Bollinger (topo)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_evolution_with_bollinger(ax1)
        
        # 2. Distribuição de Retornos (esquerda meio)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_aggregate_returns_distribution(ax2)
        
        # 3. Autocorrelação (meio centro)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_autocorrelation(ax3)
        
        # 4. Rolling Beta (direita meio)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_rolling_beta(ax4)
        
        # 5. Heatmap de Correlação (inferior esquerda)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_correlation_heatmap(ax5)
        
        # 6. Drawdowns (inferior centro)
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_drawdowns(ax6)
        
        # 7. Métricas de Risco (inferior direita)
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_risk_metrics_table(ax7)
        
        plt.suptitle('ANÁLISE QUANTITATIVA COMPLETA DE PREÇOS E RETORNOS', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0E1117')
            print(f"💾 Dashboard salvo: {save_path}")
        
        plt.show()
        return fig
    
    def _plot_price_evolution_with_bollinger(self, ax):
        """Preços com Bandas de Bollinger"""
        # Plotar preços principais
        main_assets = ['SELIC', 'USD_BRL', 'EUR_BRL', 'PETROLEO_BRENT']
        
        for i, asset in enumerate(main_assets):
            if asset in self.prices.columns:
                ax.plot(self.prices.index, self.prices[asset], 
                       label=asset, color=self.colors[i], linewidth=2)
        
        # Adicionar Bandas de Bollinger para o portfólio (exemplo)
        portfolio_prices = self.prices.mean(axis=1)
        window = 20
        rolling_mean = portfolio_prices.rolling(window=window).mean()
        rolling_std = portfolio_prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        ax.plot(portfolio_prices.index, rolling_mean, 'white', 
               linestyle='--', linewidth=1, label='Média Móvel (20)')
        ax.fill_between(portfolio_prices.index, lower_band, upper_band, 
                       alpha=0.2, color='gray', label='Banda Bollinger')
        
        ax.set_title('EVOLUÇÃO DE PREÇOS + BANDAS DE BOLLINGER', fontweight='bold')
        ax.set_ylabel('Preço (Base 100)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_aggregate_returns_distribution(self, ax):
        """Distribuição agregada de retornos"""
        portfolio_returns = self.portfolio_returns.dropna()
        
        # Histograma
        ax.hist(portfolio_returns, bins=50, density=True, alpha=0.7, 
               color='cyan', edgecolor='white')
        
        # KDE
        from scipy import stats
        kde = stats.gaussian_kde(portfolio_returns)
        x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
        ax.plot(x, kde(x), 'white', linewidth=2, label='KDE')
        
        # Distribuição normal
        normal = stats.norm(portfolio_returns.mean(), portfolio_returns.std())
        ax.plot(x, normal.pdf(x), 'yellow', linestyle='--', linewidth=1, label='Normal')
        
        # VaR 95%
        var_95 = np.percentile(portfolio_returns, 5)
        ax.axvline(x=var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.3f}')
        
        ax.set_title('Distribuição Agregada de Retorno', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_autocorrelation(self, ax):
        """Autocorrelação dos retornos"""
        from pandas.plotting import autocorrelation_plot
        
        portfolio_returns = self.portfolio_returns.dropna()
        
        # Plotar autocorrelação
        autocorrelation_plot(portfolio_returns, ax=ax)
        ax.set_title('Autocorrelação (Serial Correlation)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_rolling_beta(self, ax):
        """Rolling Beta vs Market Proxy"""
        # Usar SELIC como proxy de mercado para exemplo
        market_proxy = self.returns.get('SELIC', self.returns.iloc[:, 0])
        
        window = 63
        betas = {}
        
        for asset in self.returns.columns[:4]:  # Apenas primeiros 4 ativos
            if asset != market_proxy.name:
                rolling_beta = []
                for i in range(window, len(self.returns)):
                    asset_window = self.returns[asset].iloc[i-window:i]
                    market_window = market_proxy.iloc[i-window:i]
                    
                    covariance = np.cov(asset_window, market_window)[0,1]
                    variance = np.var(market_window)
                    beta = covariance / variance if variance != 0 else np.nan
                    rolling_beta.append(beta)
                
                if rolling_beta:
                    betas[asset] = pd.Series(rolling_beta, 
                                           index=self.returns.index[window:])
        
        for i, (asset, beta_series) in enumerate(betas.items()):
            ax.plot(beta_series.index, beta_series.values, 
                   label=asset, color=self.colors[i], linewidth=2)
        
        ax.set_title('Rolling Beta (63D vs Market Proxy)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_heatmap(self, ax):
        """Heatmap de correlação compacto"""
        corr_matrix = self.returns.corr()
        
        # Plotar heatmap
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Configurar ticks
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels([col[:8] for col in corr_matrix.columns], rotation=45)
        ax.set_yticklabels([col[:8] for col in corr_matrix.columns])
        
        # Adicionar valores
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                       ha='center', va='center', fontsize=8, 
                       color='white' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'black')
        
        ax.set_title('Matriz de Correlação', fontweight='bold')
    
    def _plot_drawdowns(self, ax):
        """Gráfico de drawdowns"""
        portfolio_cumulative = (1 + self.portfolio_returns).cumprod()
        rolling_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - rolling_max) / rolling_max
        
        ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.5)
        ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        
        ax.set_title('Drawdown do Portfólio', fontweight='bold')
        ax.set_ylabel('Drawdown')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_metrics_table(self, ax):
        """Tabela de métricas de risco"""
        # Calcular métricas básicas
        metrics_data = []
        for asset in self.returns.columns[:4]:  # Apenas primeiros 4
            returns = self.returns[asset].dropna()
            metrics_data.append([
                asset,
                f'{returns.mean() * 252:.2%}',
                f'{returns.std() * np.sqrt(252):.2%}',
                f'{(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}' if returns.std() > 0 else 'N/A',
                f'{np.percentile(returns, 5):.2%}'
            ])
        
        # Criar tabela
        column_labels = ['Ativo', 'Retorno', 'Vol', 'Sharpe', 'VaR 95%']
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=metrics_data,
                        colLabels=column_labels,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        ax.set_title('Métricas de Risco', fontweight='bold')

# Função principal atualizada
def main():
    """Função principal com dashboard avançado"""
    print("🚀 GERADOR DE DASHBOARD AVANÇADO DE RISCO")
    print("=" * 60)
    
    # Carregar dados
    returns, prices = load_data_for_visualization()
    
    if returns is not None and prices is not None:
        # Inicializar visualizador avançado
        visualizer = AdvancedRiskVisualizer(returns, prices)
        
        # Criar dashboard completo
        print("🎨 CRIANDO DASHBOARD AVANÇADO...")
        visualizer.create_comprehensive_dashboard('reports/figures/advanced_dashboard.png')
        
        print("\n✅ DASHBOARD AVANÇADO CRIADO!")
        print("📊 PRÓXIMA ETAPA: Simulações de Monte Carlo")
        print("💡 Execute: python src/simulation/monte_carlo.py")
        
    else:
        print("❌ Não foi possível carregar os dados")

# Mantém a função de carregamento de dados
def load_data_for_visualization():
    """Carrega dados para visualização"""
    try:
        returns = pd.read_parquet('data/processed/macro_portfolio_returns.parquet')
        prices = pd.read_parquet('data/processed/macro_portfolio_prices.parquet')
        print(f"✅ Dados carregados: {returns.shape}")
        return returns, prices
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None, None

if __name__ == "__main__":
    main()
