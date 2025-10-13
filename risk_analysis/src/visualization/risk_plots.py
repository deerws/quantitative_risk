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

# Configuração de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RiskVisualizer:
    def __init__(self, returns_df, prices_df):
        self.returns = returns_df
        self.prices = prices_df
        self.portfolio_returns = returns_df.mean(axis=1)  # Portfolio equal-weight
        
    def plot_price_evolution(self, save_path=None):
        """Gráfico da evolução dos preços normalizados"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plotar cada ativo
        for column in self.prices.columns:
            ax.plot(self.prices.index, self.prices[column], label=column, linewidth=2)
        
        ax.set_title('📈 Evolução dos Preços - Base 100', fontsize=16, fontweight='bold')
        ax.set_ylabel('Preço (Base 100)', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Formatar eixo Y como percentual
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Gráfico salvo: {save_path}")
        
        plt.show()
        return fig
    
    def plot_returns_distribution(self, save_path=None):
        """Distribuição dos retornos com estatísticas"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Selecionar os 4 primeiros ativos para exemplo
        assets_to_plot = self.returns.columns[:4] if len(self.returns.columns) >= 4 else self.returns.columns
        
        for i, asset in enumerate(assets_to_plot):
            if i < len(axes):
                returns = self.returns[asset].dropna()
                
                # Histograma com curva normal
                axes[i].hist(returns, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Adicionar curva normal
                xmin, xmax = axes[i].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, returns.mean(), returns.std())
                axes[i].plot(x, p, 'k', linewidth=2, label='Distribuição Normal')
                
                # Estatísticas no gráfico
                stats_text = f'Média: {returns.mean():.4f}\nStd: {returns.std():.4f}\nSkew: {returns.skew():.2f}'
                axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                axes[i].set_title(f'Distribuição de Retornos - {asset}', fontweight='bold')
                axes[i].set_xlabel('Retorno Diário')
                axes[i].set_ylabel('Densidade')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Gráfico salvo: {save_path}")
        
        plt.show()
        return fig
    
    def plot_correlation_heatmap(self, save_path=None):
        """Heatmap da matriz de correlação"""
        corr_matrix = self.returns.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Criar heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={"shrink": .8})
        
        ax.set_title('🔥 Matriz de Correlação - Retornos Diários', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Gráfico salvo: {save_path}")
        
        plt.show()
        return fig
    
    def plot_rolling_volatility(self, window=63, save_path=None):
        """Volatilidade móvel anualizada"""
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(252)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for column in rolling_vol.columns:
            ax.plot(rolling_vol.index, rolling_vol[column], label=column, linewidth=2)
        
        ax.set_title(f'📉 Volatilidade Móvel ({window} dias - Anualizada)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Volatilidade Anualizada', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Gráfico salvo: {save_path}")
        
        plt.show()
        return fig
    
    def plot_drawdowns(self, save_path=None):
        """Gráfico de drawdowns do portfólio"""
        portfolio_cumulative = (1 + self.portfolio_returns).cumprod()
        rolling_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - rolling_max) / rolling_max
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        
        ax.set_title('📉 Drawdown do Portfólio', fontsize=16, fontweight='bold')
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.set_xlabel('Data', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Adicionar linha do drawdown máximo
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.axhline(y=max_dd, color='black', linestyle='--', alpha=0.7, 
                  label=f'Drawdown Máximo: {max_dd:.2%}')
        ax.axvline(x=max_dd_date, color='black', linestyle='--', alpha=0.7)
        
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Gráfico salvo: {save_path}")
        
        plt.show()
        return fig
    
    def plot_risk_metrics_dashboard(self, save_path=None):
        """Dashboard com múltiplas métricas de risco"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Retorno vs Volatilidade',
                'Razão Sharpe por Ativo', 
                'Value at Risk (95%)',
                'Drawdown Máximo'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Calcular métricas
        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - 0.1175) / annual_volatility
        var_95 = self.returns.apply(lambda x: np.percentile(x, 5))
        max_drawdowns = self.returns.apply(self._calculate_max_drawdown)
        
        # Gráfico 1: Retorno vs Volatilidade
        for i, asset in enumerate(self.returns.columns):
            fig.add_trace(
                go.Scatter(x=[annual_volatility[asset]], y=[annual_returns[asset]],
                          mode='markers+text', name=asset, marker=dict(size=12),
                          text=asset, textposition="top center"),
                row=1, col=1
            )
        
        # Gráfico 2: Sharpe Ratio
        fig.add_trace(
            go.Bar(x=list(sharpe_ratios.index), y=sharpe_ratios.values,
                  marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Gráfico 3: VaR
        fig.add_trace(
            go.Bar(x=list(var_95.index), y=var_95.values,
                  marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Gráfico 4: Max Drawdown
        fig.add_trace(
            go.Bar(x=list(max_drawdowns.index), y=max_drawdowns.values,
                  marker_color='lightsalmon'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="📊 Dashboard de Métricas de Risco", 
                         showlegend=False)
        
        # Atualizar eixos
        fig.update_xaxes(title_text="Volatilidade Anual", row=1, col=1)
        fig.update_yaxes(title_text="Retorno Anual", row=1, col=1)
        fig.update_xaxes(title_text="Ativos", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_xaxes(title_text="Ativos", row=2, col=1)
        fig.update_yaxes(title_text="VaR 95%", row=2, col=1)
        fig.update_xaxes(title_text="Ativos", row=2, col=2)
        fig.update_yaxes(title_text="Max Drawdown", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Dashboard salvo: {save_path}")
        
        fig.show()
        return fig
    
    def _calculate_max_drawdown(self, returns):
        """Calcula drawdown máximo para uma série"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def create_all_visualizations(self):
        """Cria todas as visualizações automaticamente"""
        print("🎨 CRIANDO TODAS AS VISUALIZAÇÕES DE RISCO...")
        
        os.makedirs('reports/figures', exist_ok=True)
        
        # Lista de visualizações para criar
        visualizations = [
            ('price_evolution.png', self.plot_price_evolution),
            ('returns_distribution.png', self.plot_returns_distribution),
            ('correlation_heatmap.png', self.plot_correlation_heatmap),
            ('rolling_volatility.png', self.plot_rolling_volatility),
            ('drawdowns.png', self.plot_drawdowns),
            ('risk_dashboard.html', self.plot_risk_metrics_dashboard)
        ]
        
        for filename, plot_function in visualizations:
            save_path = f'reports/figures/{filename}'
            try:
                plot_function(save_path=save_path)
                print(f"✅ {filename}")
            except Exception as e:
                print(f"❌ Erro em {filename}: {e}")
        
        print("\n🎯 VISUALIZAÇÕES CONCLUÍDAS!")
        print("📁 Verifique a pasta 'reports/figures/'")

# Funções auxiliares
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

def main():
    """Função principal"""
    print("🚀 GERADOR DE VISUALIZAÇÕES DE RISCO")
    print("=" * 60)
    
    # Carregar dados
    returns, prices = load_data_for_visualization()
    
    if returns is not None and prices is not None:
        # Inicializar visualizador
        visualizer = RiskVisualizer(returns, prices)
        
        # Criar todas as visualizações
        visualizer.create_all_visualizations()
        
        print("\n" + "=" * 60)
        print("📊 PRÓXIMA ETAPA: Simulações de Monte Carlo")
        print("💡 Execute: python src/simulation/monte_carlo.py")
        print("=" * 60)
        
    else:
        print("❌ Não foi possível carregar os dados para visualização")

if __name__ == "__main__":
    main()
