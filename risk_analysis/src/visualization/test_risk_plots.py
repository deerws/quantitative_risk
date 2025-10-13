import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
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
        self.colors = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF6B6B', 
                      '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
        
        # Configurações globais do matplotlib
        plt.rcParams.update({
            'figure.facecolor': '#0E1117',
            'axes.facecolor': '#1a1d24',
            'axes.edgecolor': '#2d3748',
            'axes.labelcolor': '#e0e0e0',
            'text.color': '#e0e0e0',
            'xtick.color': '#e0e0e0',
            'ytick.color': '#e0e0e0',
            'grid.color': '#2d3748',
            'grid.alpha': 0.3,
            'font.size': 9,
            'axes.titlesize': 11,
            'axes.titleweight': 'bold',
            'axes.labelsize': 9,
        })
        
    def create_comprehensive_dashboard(self, save_path=None):
        """Dashboard completo otimizado sem sobreposições"""
        fig = plt.figure(figsize=(24, 14))
        
        # Definir layout do grid com mais espaçamento
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25,
                             top=0.94, bottom=0.05, left=0.06, right=0.98)
        
        # 1. Evolução de Preços com Bandas de Bollinger (topo - ocupa 2 colunas)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_price_evolution_with_bollinger(ax1)
        
        # 2. Volatilidade Realizada (topo direita)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_realized_volatility(ax2)
        
        # 3. Distribuição de Retornos (meio esquerda)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_aggregate_returns_distribution(ax3)
        
        # 4. Autocorrelação (meio centro)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_autocorrelation(ax4)
        
        # 5. Rolling Beta (meio direita)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_rolling_beta(ax5)
        
        # 6. Heatmap de Correlação (inferior esquerda)
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_correlation_heatmap(ax6)
        
        # 7. Drawdowns (inferior centro)
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_drawdowns(ax7)
        
        # 8. Métricas de Risco (inferior direita)
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_risk_metrics_table(ax8)
        
        # Título principal com mais espaço
        plt.suptitle('ANÁLISE QUANTITATIVA COMPLETA DE PREÇOS E RETORNOS', 
                    fontsize=16, fontweight='bold', y=0.98, color='#00FFFF')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='#0E1117', edgecolor='none')
            print(f"💾 Dashboard salvo: {save_path}")
        
        plt.show()
        return fig
    
    def _plot_price_evolution_with_bollinger(self, ax):
        """Preços com Bandas de Bollinger - Otimizado"""
        # Selecionar ativos principais (máximo 4 para legibilidade)
        main_assets = []
        for asset in ['SELIC', 'USD_BRL', 'EUR_BRL', 'PETROLEO_BRENT']:
            if asset in self.prices.columns:
                main_assets.append(asset)
        
        # Se não encontrar, usar primeiros 4
        if not main_assets:
            main_assets = list(self.prices.columns[:4])
        
        # Plotar preços principais
        for i, asset in enumerate(main_assets):
            ax.plot(self.prices.index, self.prices[asset], 
                   label=asset, color=self.colors[i], linewidth=2, alpha=0.9)
        
        # Adicionar Bandas de Bollinger para média do portfólio
        portfolio_prices = self.prices[main_assets].mean(axis=1)
        window = 20
        rolling_mean = portfolio_prices.rolling(window=window).mean()
        rolling_std = portfolio_prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        # Banda com transparência
        ax.fill_between(portfolio_prices.index, lower_band, upper_band, 
                       alpha=0.15, color='gray', label='Bollinger Bands (±2σ)')
        ax.plot(portfolio_prices.index, rolling_mean, 'white', 
               linestyle='--', linewidth=1.5, label='MA(20)', alpha=0.7)
        
        ax.set_title('EVOLUÇÃO DE PREÇOS + BANDAS DE BOLLINGER', 
                    fontweight='bold', pad=10, fontsize=12)
        ax.set_ylabel('Preço Normalizado (Base 100)', fontsize=10)
        ax.set_xlabel('Data', fontsize=10)
        
        # Legenda otimizada
        ax.legend(loc='upper left', fontsize=8, ncol=3, framealpha=0.9,
                 fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Remover bordas superiores e direitas
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def _plot_realized_volatility(self, ax):
        """Volatilidade Realizada - Novo painel"""
        window = 21
        
        # Selecionar 3 ativos principais
        main_assets = list(self.returns.columns[:3])
        
        for i, asset in enumerate(main_assets):
            vol = self.returns[asset].rolling(window).std() * np.sqrt(252)
            ax.plot(vol.index, vol, label=asset, 
                   color=self.colors[i], linewidth=2, alpha=0.85)
        
        ax.set_title('VOLATILIDADE REALIZADA (21d)', 
                    fontweight='bold', pad=10, fontsize=11)
        ax.set_ylabel('Vol. Anualizada', fontsize=9)
        ax.set_xlabel('Data', fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def _plot_aggregate_returns_distribution(self, ax):
        """Distribuição agregada de retornos - Otimizada"""
        portfolio_returns = self.portfolio_returns.dropna()
        
        # Histograma com menos bins para clareza
        n, bins, patches = ax.hist(portfolio_returns, bins=40, density=True, 
                                   alpha=0.6, color='cyan', edgecolor='white', 
                                   linewidth=0.5)
        
        # KDE suavizada
        kde = stats.gaussian_kde(portfolio_returns)
        x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 200)
        ax.plot(x, kde(x), 'white', linewidth=2.5, label='KDE', alpha=0.9)
        
        # Distribuição normal teórica
        normal = stats.norm(portfolio_returns.mean(), portfolio_returns.std())
        ax.plot(x, normal.pdf(x), 'yellow', linestyle='--', 
               linewidth=2, label='Normal Teórica', alpha=0.8)
        
        # VaR 95% com anotação
        var_95 = np.percentile(portfolio_returns, 5)
        ax.axvline(x=var_95, color='red', linestyle='--', linewidth=2, 
                  label=f'VaR 95%', alpha=0.9)
        
        # Adicionar box com estatísticas
        stats_text = (f'μ = {portfolio_returns.mean():.4f}\n'
                     f'σ = {portfolio_returns.std():.4f}\n'
                     f'Skew = {portfolio_returns.skew():.2f}\n'
                     f'Kurt = {portfolio_returns.kurtosis():.2f}')
        
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='#1a1d24', 
                        edgecolor='cyan', alpha=0.9, linewidth=2),
               fontsize=7, family='monospace')
        
        ax.set_title('DISTRIBUIÇÃO AGREGADA DE RETORNOS', 
                    fontweight='bold', pad=10, fontsize=11)
        ax.set_xlabel('Retorno Diário', fontsize=9)
        ax.set_ylabel('Densidade', fontsize=9)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def _plot_autocorrelation(self, ax):
        """Autocorrelação dos retornos - Otimizada"""
        portfolio_returns = self.portfolio_returns.dropna()
        
        # Calcular autocorrelação manualmente para melhor controle
        max_lags = min(100, len(portfolio_returns)//4)
        lags = range(1, max_lags)
        acf_values = [portfolio_returns.autocorr(lag=lag) for lag in lags]
        
        # Plotar autocorrelação
        ax.plot(lags, acf_values, color='cyan', linewidth=2, alpha=0.9)
        ax.fill_between(lags, 0, acf_values, alpha=0.3, color='cyan')
        
        # Adicionar linha zero
        ax.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
        
        # Bandas de confiança (95%)
        confidence = 1.96 / np.sqrt(len(portfolio_returns))
        ax.axhline(y=confidence, color='yellow', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label='95% Confidence')
        ax.axhline(y=-confidence, color='yellow', linestyle='--', 
                  linewidth=1.5, alpha=0.7)
        
        ax.set_title('AUTOCORRELAÇÃO (Serial Correlation)', 
                    fontweight='bold', pad=10, fontsize=11)
        ax.set_xlabel('Lag (dias)', fontsize=9)
        ax.set_ylabel('ACF', fontsize=9)
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, max_lags)
    
    def _plot_rolling_beta(self, ax):
        """Rolling Beta vs Market Proxy - Otimizado"""
        # Usar primeiro ativo como proxy de mercado
        market_proxy = self.returns.iloc[:, 0]
        
        window = 63
        
        # Selecionar apenas 3 ativos para clareza
        assets_to_plot = [col for col in self.returns.columns[1:4] 
                         if col != market_proxy.name]
        
        for i, asset in enumerate(assets_to_plot):
            # Calcular covariância e variância rolantes
            cov_rolling = self.returns[asset].rolling(window).cov(market_proxy)
            var_rolling = market_proxy.rolling(window).var()
            beta_rolling = cov_rolling / var_rolling
            
            ax.plot(beta_rolling.index, beta_rolling.values, 
                   label=asset[:15], color=self.colors[i+1], 
                   linewidth=2, alpha=0.85)
        
        # Linha beta = 1
        ax.axhline(y=1, color='white', linestyle='--', 
                  linewidth=1.5, alpha=0.6, label='β = 1')
        ax.axhline(y=0, color='gray', linestyle='-', 
                  linewidth=1, alpha=0.4)
        
        ax.set_title(f'ROLLING BETA ({window}d vs Market Proxy)', 
                    fontweight='bold', pad=10, fontsize=11)
        ax.set_ylabel('Beta (β)', fontsize=9)
        ax.set_xlabel('Data', fontsize=9)
        ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=1)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def _plot_correlation_heatmap(self, ax):
        """Heatmap de correlação - Otimizado"""
        corr_matrix = self.returns.corr()
        
        # Limitar a 8 ativos para legibilidade
        if len(corr_matrix) > 8:
            corr_matrix = corr_matrix.iloc[:8, :8]
        
        # Usar seaborn para melhor aparência
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, square=True,
                   linewidths=1, linecolor='#0E1117',
                   cbar_kws={'shrink': 0.8, 'label': 'Correlação'},
                   annot_kws={'size': 8, 'weight': 'bold'},
                   vmin=-1, vmax=1, ax=ax)
        
        # Ajustar labels
        ax.set_xticklabels([col[:10] for col in corr_matrix.columns], 
                          rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([col[:10] for col in corr_matrix.index], 
                          rotation=0, fontsize=8)
        
        ax.set_title('MATRIZ DE CORRELAÇÃO', 
                    fontweight='bold', pad=10, fontsize=11)
    
    def _plot_drawdowns(self, ax):
        """Gráfico de drawdowns - Otimizado"""
        portfolio_cumulative = (1 + self.portfolio_returns).cumprod()
        rolling_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - rolling_max) / rolling_max
        
        # Preencher área de drawdown
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                       color='red', alpha=0.4, label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, 
               color='darkred', linewidth=1.5, alpha=0.9)
        
        # Marcar drawdown máximo
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        ax.scatter([max_dd_date], [max_dd], color='yellow', 
                  s=100, zorder=5, edgecolors='white', linewidth=2,
                  label=f'Max DD: {max_dd:.2%}')
        
        ax.axhline(y=max_dd, color='yellow', linestyle='--', 
                  linewidth=1.5, alpha=0.6)
        
        ax.set_title('DRAWDOWN DO PORTFÓLIO', 
                    fontweight='bold', pad=10, fontsize=11)
        ax.set_ylabel('Drawdown (%)', fontsize=9)
        ax.set_xlabel('Data', fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(fontsize=7, loc='lower left', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def _plot_risk_metrics_table(self, ax):
        """Tabela de métricas de risco - Otimizada"""
        # Calcular métricas para até 6 ativos
        n_assets = min(6, len(self.returns.columns))
        metrics_data = []
        
        for asset in self.returns.columns[:n_assets]:
            returns = self.returns[asset].dropna()
            
            # Retorno anualizado
            annual_return = returns.mean() * 252
            
            # Volatilidade anualizada
            annual_vol = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio (assumindo risk-free = 11.75%)
            sharpe = (annual_return - 0.1175) / annual_vol if annual_vol > 0 else 0
            
            # VaR 95%
            var_95 = np.percentile(returns, 5)
            
            # Max Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            metrics_data.append([
                asset[:12],  # Truncar nome
                f'{annual_return:.1%}',
                f'{annual_vol:.1%}',
                f'{sharpe:.2f}',
                f'{var_95:.2%}',
                f'{max_dd:.1%}'
            ])
        
        # Criar tabela estilizada
        column_labels = ['Ativo', 'Ret.', 'Vol.', 'Sharpe', 'VaR95%', 'MaxDD']
        
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=metrics_data,
                        colLabels=column_labels,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 2)
        
        # Estilizar células do cabeçalho
        for i in range(len(column_labels)):
            cell = table[(0, i)]
            cell.set_facecolor('#00FFFF')
            cell.set_text_props(weight='bold', color='black')
            cell.set_linewidth(2)
        
        # Estilizar células de dados
        for i in range(1, len(metrics_data) + 1):
            for j in range(len(column_labels)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#1a1d24')
                else:
                    cell.set_facecolor('#242831')
                cell.set_linewidth(1)
                cell.set_edgecolor('#2d3748')
        
        ax.set_title('MÉTRICAS DE RISCO', 
                    fontweight='bold', pad=10, fontsize=11, color='#00FFFF')

# Funções auxiliares
def load_data_for_visualization():
    """Carrega dados para visualização"""
    try:
        returns = pd.read_parquet('data/processed/macro_portfolio_returns.parquet')
        prices = pd.read_parquet('data/processed/macro_portfolio_prices.parquet')
        print(f"✅ Dados carregados: {returns.shape[0]} períodos, {returns.shape[1]} ativos")
        return returns, prices
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None, None

def main():
    """Função principal com dashboard avançado"""
    print("\n" + "="*80)
    print("🚀 GERADOR DE DASHBOARD AVANÇADO DE RISCO - TEMA ESCURO OTIMIZADO")
    print("="*80 + "\n")
    
    # Carregar dados
    returns, prices = load_data_for_visualization()
    
    if returns is not None and prices is not None:
        # Inicializar visualizador avançado
        visualizer = AdvancedRiskVisualizer(returns, prices)
        
        # Criar diretório se não existir
        os.makedirs('reports/figures', exist_ok=True)
        
        # Criar dashboard completo
        print("🎨 CRIANDO DASHBOARD AVANÇADO...")
        visualizer.create_comprehensive_dashboard('reports/figures/advanced_dashboard.png')
        
        print("\n" + "="*80)
        print("✅ DASHBOARD AVANÇADO CRIADO COM SUCESSO!")
        print("="*80)
        print("\n📊 Componentes incluídos:")
        print("   ✓ Evolução de Preços + Bandas de Bollinger")
        print("   ✓ Volatilidade Realizada (21d)")
        print("   ✓ Distribuição de Retornos (KDE + Normal + VaR)")
        print("   ✓ Autocorrelação (com bandas de confiança)")
        print("   ✓ Rolling Beta vs Market Proxy")
        print("   ✓ Matriz de Correlação (heatmap)")
        print("   ✓ Drawdown Timeline com ponto máximo")
        print("   ✓ Tabela de Métricas Quantitativas")
        print("\n📁 Arquivo salvo: reports/figures/advanced_dashboard.png")
        print("💡 Resolução: 300 DPI (pronto para impressão)")
        print("="*80 + "\n")
        
    else:
        print("❌ Não foi possível carregar os dados")

if __name__ == "__main__":
    main()
