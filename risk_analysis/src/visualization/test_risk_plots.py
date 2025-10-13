import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gaussian_kde, jarque_bera, shapiro
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DE TEMA ESCURO PROFISSIONAL - QUANT ANALYTICS
# ============================================================================

COLORS = {
    'background': '#0a0e27',
    'paper': '#141b2d',
    'text': '#e0e0e0',
    'grid': '#2d3548',
    'accent1': '#00d4ff',
    'accent2': '#7c4dff',
    'accent3': '#00e676',
    'accent4': '#ff6b6b',
    'accent5': '#ffd93d',
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

ASSET_COLORS = ['#00d4ff', '#7c4dff', '#00e676', '#ff6b6b', '#ffd93d', 
                '#ff9ff3', '#54a0ff', '#48dbfb', '#ff9ff3', '#1dd1a1']

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': COLORS['background'],
    'axes.facecolor': COLORS['paper'],
    'axes.edgecolor': COLORS['grid'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'legend.facecolor': COLORS['paper'],
    'legend.edgecolor': COLORS['grid']
})

class AdvancedRiskVisualizer:
    def __init__(self, returns_df, prices_df):
        self.returns = returns_df
        self.prices = prices_df
        self.portfolio_returns = returns_df.mean(axis=1)
        
        # Calcular métricas avançadas
        self._calculate_advanced_metrics()
        
    def _calculate_advanced_metrics(self):
        """Calcula métricas avançadas para análise quantitativa"""
        self.metrics = {}
        
        for asset in self.returns.columns:
            returns = self.returns[asset].dropna()
            
            # Momentos estatísticos
            self.metrics[asset] = {
                'mean': returns.mean() * 252,
                'vol': returns.std() * np.sqrt(252),
                'sharpe': (returns.mean() * 252 - 0.1175) / (returns.std() * np.sqrt(252)),
                'sortino': self._sortino_ratio(returns),
                'calmar': self._calmar_ratio(returns),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'var_95': np.percentile(returns, 5),
                'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
                'max_dd': self._calculate_max_drawdown(returns),
                'omega': self._omega_ratio(returns),
                'tail_ratio': self._tail_ratio(returns),
                'jb_stat': jarque_bera(returns)[0],
                'jb_pvalue': jarque_bera(returns)[1]
            }
    
    def _sortino_ratio(self, returns, target=0):
        """Sortino Ratio - penaliza apenas downside volatility"""
        downside = returns[returns < target]
        downside_std = downside.std() * np.sqrt(252)
        return (returns.mean() * 252 - 0.1175) / downside_std if downside_std > 0 else 0
    
    def _calmar_ratio(self, returns):
        """Calmar Ratio - retorno anualizado / max drawdown"""
        max_dd = abs(self._calculate_max_drawdown(returns))
        return (returns.mean() * 252) / max_dd if max_dd > 0 else 0
    
    def _omega_ratio(self, returns, threshold=0):
        """Omega Ratio - probabilidade de ganhos vs perdas"""
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        return gains / losses if losses > 0 else np.inf
    
    def _tail_ratio(self, returns):
        """Tail Ratio - 95th percentile / 5th percentile"""
        return abs(np.percentile(returns, 95) / np.percentile(returns, 5))
    
    def _calculate_max_drawdown(self, returns):
        """Calcula drawdown máximo"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def plot_comprehensive_price_analysis(self, save_path=None):
        """Análise completa de preços com múltiplos indicadores"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. Evolução de Preços Normalizados
        ax1 = fig.add_subplot(gs[0:2, :])
        for i, column in enumerate(self.prices.columns):
            color = ASSET_COLORS[i % len(ASSET_COLORS)]
            ax1.plot(self.prices.index, self.prices[column], 
                    label=column, linewidth=2.5, color=color, alpha=0.9)
        
        # Bandas de Bollinger do portfólio
        portfolio_prices = self.prices.mean(axis=1)
        rolling_mean = portfolio_prices.rolling(20).mean()
        rolling_std = portfolio_prices.rolling(20).std()
        ax1.plot(rolling_mean.index, rolling_mean, 'w--', linewidth=2, 
                label='MA(20) Portfolio', alpha=0.7)
        ax1.fill_between(rolling_mean.index, 
                         rolling_mean - 2*rolling_std,
                         rolling_mean + 2*rolling_std,
                         alpha=0.1, color='white', label='Bollinger Bands')
        
        ax1.set_title('📈 EVOLUÇÃO DE PREÇOS + BANDAS DE BOLLINGER', 
                     fontsize=16, fontweight='bold', color=COLORS['accent1'], pad=15)
        ax1.set_ylabel('Índice Base 100', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', ncol=3, framealpha=0.95)
        ax1.grid(True, alpha=0.2)
        
        # 2. Histograma de Retornos Agregados
        ax2 = fig.add_subplot(gs[2, 0])
        all_returns = self.returns.values.flatten()
        all_returns = all_returns[~np.isnan(all_returns)]
        
        ax2.hist(all_returns, bins=100, density=True, alpha=0.7, 
                color=COLORS['accent2'], edgecolor='none')
        
        x = np.linspace(all_returns.min(), all_returns.max(), 200)
        kde = gaussian_kde(all_returns)
        ax2.plot(x, kde(x), color=COLORS['accent1'], linewidth=3, label='KDE')
        ax2.plot(x, stats.norm.pdf(x, all_returns.mean(), all_returns.std()),
                'w--', linewidth=2, label='Normal')
        
        ax2.axvline(np.percentile(all_returns, 5), color=COLORS['accent4'], 
                   linestyle='--', linewidth=2, label=f'VaR 95%')
        
        ax2.set_title('📊 Distribuição Agregada de Retornos', fontsize=13, 
                     fontweight='bold', color=COLORS['accent2'])
        ax2.set_xlabel('Retorno Diário')
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        
        # 3. Q-Q Plot
        ax3 = fig.add_subplot(gs[2, 1])
        stats.probplot(all_returns, dist="norm", plot=ax3)
        ax3.get_lines()[0].set_color(COLORS['accent3'])
        ax3.get_lines()[0].set_markersize(4)
        ax3.get_lines()[0].set_alpha(0.6)
        ax3.get_lines()[1].set_color(COLORS['accent1'])
        ax3.get_lines()[1].set_linewidth(2)
        
        ax3.set_title('📐 Q-Q Plot (Normalidade)', fontsize=13, 
                     fontweight='bold', color=COLORS['accent3'])
        ax3.grid(True, alpha=0.2)
        
        # 4. Autocorrelação
        ax4 = fig.add_subplot(gs[3, 0])
        from pandas.plotting import autocorrelation_plot
        for i, col in enumerate(self.returns.columns[:3]):  # Top 3 ativos
            returns = self.returns[col].dropna()
            lags = range(1, min(50, len(returns)//2))
            acf = [returns.autocorr(lag) for lag in lags]
            ax4.plot(lags, acf, label=col, linewidth=2, 
                    color=ASSET_COLORS[i], alpha=0.8)
        
        ax4.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax4.axhline(y=1.96/np.sqrt(len(returns)), color=COLORS['accent4'], 
                   linestyle='--', alpha=0.5)
        ax4.axhline(y=-1.96/np.sqrt(len(returns)), color=COLORS['accent4'], 
                   linestyle='--', alpha=0.5)
        
        ax4.set_title('🔄 Autocorrelação (Serial Correlation)', fontsize=13, 
                     fontweight='bold', color=COLORS['accent5'])
        ax4.set_xlabel('Lag')
        ax4.set_ylabel('ACF')
        ax4.legend()
        ax4.grid(True, alpha=0.2)
        
        # 5. Rolling Beta vs SPY (se houver)
        ax5 = fig.add_subplot(gs[3, 1])
        window = 63
        for i, col in enumerate(self.returns.columns):
            # Calcular beta rolante vs primeiro ativo (proxy de mercado)
            market = self.returns.iloc[:, 0]
            cov = self.returns[col].rolling(window).cov(market)
            var = market.rolling(window).var()
            beta = cov / var
            
            ax5.plot(beta.index, beta, label=col, linewidth=2,
                    color=ASSET_COLORS[i % len(ASSET_COLORS)], alpha=0.8)
        
        ax5.axhline(y=1, color='white', linestyle='--', alpha=0.5, label='β=1')
        ax5.set_title(f'📈 Rolling Beta ({window}d vs Market Proxy)', 
                     fontsize=13, fontweight='bold', color=COLORS['accent1'])
        ax5.set_xlabel('Data')
        ax5.set_ylabel('Beta')
        ax5.legend(ncol=2)
        ax5.grid(True, alpha=0.2)
        
        plt.suptitle('🎯 ANÁLISE QUANTITATIVA COMPLETA DE PREÇOS E RETORNOS', 
                    fontsize=18, fontweight='bold', color=COLORS['accent1'], y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=COLORS['background'])
            print(f"💾 Análise completa salva: {save_path}")
        
        plt.show()
        return fig
    
    def plot_advanced_distribution_analysis(self, save_path=None):
        """Análise profunda de distribuições com testes estatísticos"""
        n_assets = len(self.returns.columns)
        n_cols = 2
        n_rows = (n_assets + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, asset in enumerate(self.returns.columns):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            returns = self.returns[asset].dropna()
            color = ASSET_COLORS[idx % len(ASSET_COLORS)]
            
            # Histograma + KDE + Normal
            n, bins, patches = ax.hist(returns, bins=80, density=True, 
                                       alpha=0.5, color=color, edgecolor='none')
            
            x = np.linspace(returns.min(), returns.max(), 300)
            kde = gaussian_kde(returns)
            ax.plot(x, kde(x), color=color, linewidth=3, label='KDE Real', alpha=0.9)
            ax.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
                   color=COLORS['accent1'], linewidth=2.5, 
                   linestyle='--', label='Normal Teórica')
            
            # Student-t distribution fit
            params = stats.t.fit(returns)
            ax.plot(x, stats.t.pdf(x, *params), color=COLORS['accent3'],
                   linewidth=2, linestyle=':', label='Student-t Fit')
            
            # Marcar VaR e CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            ax.axvline(var_95, color=COLORS['accent4'], linestyle='--', 
                      linewidth=2, label=f'VaR 95%: {var_95:.3%}')
            ax.axvline(cvar_95, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'CVaR: {cvar_95:.3%}')
            
            # Box com estatísticas avançadas
            metrics = self.metrics[asset]
            stats_text = (
                f"μ = {metrics['mean']:.2%}  |  σ = {metrics['vol']:.2%}\n"
                f"Skew = {metrics['skewness']:.3f}  |  Kurt = {metrics['kurtosis']:.3f}\n"
                f"Sharpe = {metrics['sharpe']:.3f}  |  Sortino = {metrics['sortino']:.3f}\n"
                f"JB p-val = {metrics['jb_pvalue']:.4f}  {'✅' if metrics['jb_pvalue'] > 0.05 else '❌'}\n"
                f"Omega = {metrics['omega']:.3f}  |  Tail = {metrics['tail_ratio']:.3f}"
            )
            
            props = dict(boxstyle='round,pad=1', facecolor=COLORS['paper'], 
                        edgecolor=color, linewidth=2.5, alpha=0.98)
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=9, bbox=props, family='monospace')
            
            ax.set_title(f'{asset} - Análise Distribucional', fontsize=13, 
                        fontweight='bold', color=color, pad=10)
            ax.set_xlabel('Retorno Diário', fontsize=10)
            ax.set_ylabel('Densidade', fontsize=10)
            ax.legend(loc='upper left', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.2, linestyle='--')
        
        # Remover eixos vazios se número ímpar
        if n_assets % 2 != 0:
            axes[-1, -1].remove()
        
        plt.suptitle('📊 ANÁLISE AVANÇADA DE DISTRIBUIÇÕES + TESTES ESTATÍSTICOS', 
                    fontsize=18, fontweight='bold', color=COLORS['accent2'], y=0.998)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=COLORS['background'])
            print(f"💾 Análise de distribuições salva: {save_path}")
        
        plt.show()
        return fig
    
    def plot_correlation_and_cointegration(self, save_path=None):
        """Matriz de correlação + análise de cointegração"""
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Heatmap de Correlação Pearson
        ax1 = fig.add_subplot(gs[0, 0])
        corr_matrix = self.returns.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(250, 10, s=80, l=55, n=256, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap=cmap, center=0, square=True, ax=ax1,
                   linewidths=2, linecolor=COLORS['background'],
                   cbar_kws={"shrink": 0.8}, annot_kws={'size': 9, 'weight': 'bold'})
        
        ax1.set_title('🔥 Correlação de Pearson (Linear)', fontsize=14, 
                     fontweight='bold', color=COLORS['accent4'], pad=15)
        
        # 2. Correlação de Spearman (Rank)
        ax2 = fig.add_subplot(gs[0, 1])
        spearman_corr = self.returns.corr(method='spearman')
        
        sns.heatmap(spearman_corr, mask=mask, annot=True, fmt='.2f',
                   cmap=cmap, center=0, square=True, ax=ax2,
                   linewidths=2, linecolor=COLORS['background'],
                   cbar_kws={"shrink": 0.8}, annot_kws={'size': 9, 'weight': 'bold'})
        
        ax2.set_title('📊 Correlação de Spearman (Rank-based)', fontsize=14, 
                     fontweight='bold', color=COLORS['accent2'], pad=15)
        
        # 3. Rolling Correlation (primeiro par de ativos)
        ax3 = fig.add_subplot(gs[1, 0])
        if len(self.returns.columns) >= 2:
            window = 63
            rolling_corr = self.returns.iloc[:, 0].rolling(window).corr(
                self.returns.iloc[:, 1])
            
            ax3.plot(rolling_corr.index, rolling_corr, linewidth=2.5, 
                    color=COLORS['accent1'], alpha=0.9)
            ax3.fill_between(rolling_corr.index, rolling_corr, 0, 
                            alpha=0.3, color=COLORS['accent1'])
            
            ax3.axhline(y=rolling_corr.mean(), color=COLORS['accent3'], 
                       linestyle='--', linewidth=2, 
                       label=f'Média: {rolling_corr.mean():.3f}')
            ax3.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
            
            ax3.set_title(f'🔄 Rolling Correlation ({window}d): {self.returns.columns[0]} vs {self.returns.columns[1]}',
                         fontsize=13, fontweight='bold', color=COLORS['accent1'])
            ax3.set_ylabel('Correlação')
            ax3.legend()
            ax3.grid(True, alpha=0.2)
        
        # 4. Heatmap de Distância de Correlação
        ax4 = fig.add_subplot(gs[1, 1])
        distance_matrix = 1 - np.abs(corr_matrix)
        
        sns.heatmap(distance_matrix, annot=True, fmt='.2f',
                   cmap='viridis', square=True, ax=ax4,
                   linewidths=2, linecolor=COLORS['background'],
                   cbar_kws={"shrink": 0.8, "label": "Distância"},
                   annot_kws={'size': 9, 'weight': 'bold'})
        
        ax4.set_title('🎯 Matriz de Distância (1 - |ρ|)', fontsize=14, 
                     fontweight='bold', color=COLORS['accent5'], pad=15)
        
        plt.suptitle('🔗 ANÁLISE COMPLETA DE CORRELAÇÃO E DEPENDÊNCIA', 
                    fontsize=18, fontweight='bold', color=COLORS['accent1'], y=0.998)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=COLORS['background'])
            print(f"💾 Análise de correlação salva: {save_path}")
        
        plt.show()
        return fig
    
    def plot_volatility_regime_analysis(self, save_path=None):
        """Análise de regimes de volatilidade"""
        fig, axes = plt.subplots(3, 1, figsize=(20, 14))
        
        # 1. Volatilidade Realizada vs EWMA vs GARCH
        ax1 = axes[0]
        window = 21
        
        for i, col in enumerate(self.returns.columns):
            returns = self.returns[col]
            
            # Volatilidade Realizada
            realized_vol = returns.rolling(window).std() * np.sqrt(252)
            
            # EWMA Volatility
            ewma_vol = returns.ewm(span=window).std() * np.sqrt(252)
            
            color = ASSET_COLORS[i % len(ASSET_COLORS)]
            ax1.plot(realized_vol.index, realized_vol, linewidth=2, 
                    color=color, alpha=0.7, label=f'{col} Realized')
            ax1.plot(ewma_vol.index, ewma_vol, linewidth=2, 
                    color=color, alpha=0.9, linestyle='--', 
                    label=f'{col} EWMA')
        
        ax1.set_title('📈 VOLATILIDADE: Realizada vs EWMA', fontsize=14, 
                     fontweight='bold', color=COLORS['accent1'], pad=15)
        ax1.set_ylabel('Volatilidade Anualizada', fontsize=11)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax1.legend(ncol=4, loc='best')
        ax1.grid(True, alpha=0.2)
        
        # 2. Cone de Volatilidade
        ax2 = axes[1]
        windows = [5, 10, 21, 63, 126, 252]
        percentiles = [10, 25, 50, 75, 90]
        
        vol_data = []
        for window in windows:
            vols = self.returns.rolling(window).std().dropna() * np.sqrt(252)
            vol_stats = [np.percentile(vols.values.flatten(), p) for p in percentiles]
            vol_data.append(vol_stats)
        
        vol_data = np.array(vol_data).T
        
        for i, p in enumerate(percentiles):
            ax2.plot(windows, vol_data[i], marker='o', linewidth=2.5, 
                    markersize=8, label=f'{p}th percentile',
                    color=ASSET_COLORS[i], alpha=0.9)
        
        # Volatilidade atual
        current_vol = self.returns.iloc[-21:].std() * np.sqrt(252)
        ax2.axhline(y=current_vol.mean(), color=COLORS['accent4'], 
                   linestyle='--', linewidth=3, 
                   label=f'Vol Atual: {current_vol.mean():.1%}')
        
        ax2.set_title('🎯 CONE DE VOLATILIDADE (Percentis)', fontsize=14, 
                     fontweight='bold', color=COLORS['accent2'], pad=15)
        ax2.set_xlabel('Janela (dias)', fontsize=11)
        ax2.set_ylabel('Volatilidade Anualizada', fontsize=11)
        ax2.set_xscale('log')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax2.legend(ncol=3)
        ax2.grid(True, alpha=0.2)
        
        # 3. Clustering de Volatilidade (Volatility Clustering)
        ax3 = axes[2]
        portfolio_vol = self.portfolio_returns.rolling(21).std() * np.sqrt(252)
        vol_squared = (self.portfolio_returns ** 2).rolling(21).mean() * 252
        
        ax3.plot(portfolio_vol.index, portfolio_vol, linewidth=2.5, 
                color=COLORS['accent1'], label='Realized Vol (21d)', alpha=0.9)
        ax3.plot(vol_squared.index, np.sqrt(vol_squared), linewidth=2, 
                color=COLORS['accent3'], linestyle='--', 
                label='RMS Vol (21d)', alpha=0.8)
        
        # Regime de alta/baixa volatilidade
        vol_median = portfolio_vol.median()
        high_vol = portfolio_vol > vol_median
        
        ax3.fill_between(portfolio_vol.index, 0, 1, where=high_vol,
                        transform=ax3.get_xaxis_transform(),
                        alpha=0.2, color=COLORS['accent4'], 
                        label='Regime Alta Vol')
        
        ax3.axhline(y=vol_median, color='white', linestyle=':', 
                   linewidth=2, alpha=0.7, label=f'Mediana: {vol_median:.1%}')
        
        ax3.set_title('💥 CLUSTERING DE VOLATILIDADE (Portfolio)', fontsize=14, 
                     fontweight='bold', color=COLORS['accent3'], pad=15)
        ax3.set_ylabel('Volatilidade Anualizada', fontsize=11)
        ax3.set_xlabel('Data', fontsize=11)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.2)
        
        plt.suptitle('📊 ANÁLISE DE REGIMES DE VOLATILIDADE', fontsize=18, 
                    fontweight='bold', color=COLORS['accent1'], y=0.998)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=COLORS['background'])
            print(f"💾 Análise de volatilidade salva: {save_path}")
        
        plt.show()
        return fig
    
    def plot_tail_risk_analysis(self, save_path=None):
        """Análise profunda de risco de cauda"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # 1. VaR e CVaR Históricos por Ativo
        ax1 = fig.add_subplot(gs[0, :])
        
        assets = self.returns.columns
        x_pos = np.arange(len(assets))
        width = 0.35
        
        var_95 = [self.metrics[asset]['var_95'] for asset in assets]
        cvar_95 = [self.metrics[asset]['cvar_95'] for asset in assets]
        
        bars1 = ax1.bar(x_pos - width/2, var_95, width, 
                       label='VaR 95%', color=COLORS['accent4'], 
                       alpha=0.8, edgecolor='white', linewidth=1.5)
        bars2 = ax1.bar(x_pos + width/2, cvar_95, width,
                       label='CVaR 95% (ES)', color='darkred',
                       alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Adicionar valores
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', 
                    fontsize=9, color='white', weight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom',
                    fontsize=9, color='white', weight='bold')
        
        ax1.set_xlabel('Ativos', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Perda Esperada', fontsize=12, fontweight='bold')
        ax1.set_title('📉 VALUE AT RISK vs CONDITIONAL VAR (Expected Shortfall)', 
                     fontsize=14, fontweight='bold', color=COLORS['accent4'], pad=15)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(assets, rotation=0)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.2, axis='y')
        
        # 2. Distribuição de Caudas - Portfolio
        ax2 = fig.add_subplot(gs[1, 0])
        portfolio_returns = self.portfolio_returns.dropna()
        
        # Focar nas caudas
        left_tail = portfolio_returns[portfolio_returns < np.percentile(portfolio_returns, 10)]
        right_tail = portfolio_returns[portfolio_returns > np.percentile(portfolio_returns, 90)]
        
        ax2.hist(left_tail, bins=50, alpha=0.7, color=COLORS['accent4'], 
                edgecolor='none', label='Cauda Esquerda (10%)')
        ax2.hist(right_tail, bins=50, alpha=0.7, color=COLORS['accent3'],
                edgecolor='none', label='Cauda Direita (10%)')
        
        ax2.axvline(left_tail.mean(), color=COLORS['accent4'], 
                   linestyle='--', linewidth=2.5, 
                   label=f'Média Esq: {left_tail.mean():.3%}')
        ax2.axvline(right_tail.mean(), color=COLORS['accent3'],
                   linestyle='--', linewidth=2.5,
                   label=f'Média Dir: {right_tail.mean():.3%}')
        
        ax2.set_title('🎯 ANÁLISE DE CAUDAS (Tail Events)', fontsize=13,
                     fontweight='bold', color=COLORS['accent5'], pad=10)
        ax2.set_xlabel('Retorno Diário')
        ax2.set_ylabel('Frequência')
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        
        # 3. Extreme Value Theory - Hill Estimator
        ax3 = fig.add_subplot(gs[1, 1])
        
        returns_sorted = np.sort(np.abs(portfolio_returns))[::-1]
        k_values = range(10, min(200, len(returns_sorted)//2))
        hill_estimates = []
        
        for k in k_values:
            top_k = returns_sorted[:k]
            hill = (1/k) * np.sum(np.log(top_k/returns_sorted[k]))
            hill_estimates.append(1/hill if hill > 0 else 0)  # Tail index
        
        ax3.plot(k_values, hill_estimates, linewidth=2.5, 
                color=COLORS['accent2'], marker='o', markersize=4, alpha=0.8)
        ax3.axhline(y=3, color=COLORS['accent1'], linestyle='--', 
                   linewidth=2, label='α=3 (Referência)', alpha=0.7)
        
        ax3.set_title('📊 HILL ESTIMATOR (Tail Index α)', fontsize=13,
                     fontweight='bold', color=COLORS['accent2'], pad=10)
        ax3.set_xlabel('k (número de extremos)')
        ax3.set_ylabel('Tail Index α')
        ax3.legend()
        ax3.grid(True, alpha=0.2)
        
        # 4. Rolling VaR Multi-Level
        ax4 = fig.add_subplot(gs[2, :])
        window = 252
        
        var_levels = [1, 5, 10]
        for level in var_levels:
            rolling_var = portfolio_returns.rolling(window).quantile(level/100)
            ax4.plot(rolling_var.index, rolling_var, linewidth=2.5, 
                    label=f'VaR {level}%', alpha=0.85)
        
        # Destacar violações de VaR
        var_5 = portfolio_returns.rolling(window).quantile(0.05)
        violations = portfolio_returns < var_5
        ax4.scatter(portfolio_returns[violations].index, 
                   portfolio_returns[violations].values,
                   color=COLORS['accent4'], s=50, alpha=0.8, 
                   label='Violações VaR 5%', zorder=5)
        
        ax4.set_title('⚠️ ROLLING VAR MULTI-NÍVEL + BACKTESTING', fontsize=14,
                     fontweight='bold', color=COLORS['accent4'], pad=15)
        ax4.set_ylabel('VaR / Retorno')
        ax4.set_xlabel('Data')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax4.legend(ncol=4, loc='best')
        ax4.grid(True, alpha=0.2)
        
        plt.suptitle('🔥 ANÁLISE AVANÇADA DE RISCO DE CAUDA (TAIL RISK)', 
                    fontsize=18, fontweight='bold', color=COLORS['accent4'], y=0.998)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=COLORS['background'])
            print(f"💾 Análise de tail risk salva: {save_path}")
        
        plt.show()
        return fig
    
    def plot_performance_attribution(self, save_path=None):
        """Análise de atribuição de performance e métricas avançadas"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
        
        # 1. Risk-Adjusted Returns Comparison
        ax1 = fig.add_subplot(gs[0, :])
        
        metrics_to_plot = ['sharpe', 'sortino', 'calmar', 'omega']
        x = np.arange(len(self.returns.columns))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            values = [self.metrics[asset][metric] for asset in self.returns.columns]
            offset = width * (i - len(metrics_to_plot)/2 + 0.5)
            bars = ax1.bar(x + offset, values, width, label=metric.capitalize(),
                          color=ASSET_COLORS[i], alpha=0.85, 
                          edgecolor='white', linewidth=1.5)
            
            # Adicionar valores
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom',
                        fontsize=8, color='white', weight='bold')
        
        ax1.set_xlabel('Ativos', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Ratio', fontsize=12, fontweight='bold')
        ax1.set_title('⚡ COMPARAÇÃO DE MÉTRICAS RISK-ADJUSTED', fontsize=14,
                     fontweight='bold', color=COLORS['accent1'], pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.returns.columns, rotation=0)
        ax1.legend(ncol=4, loc='upper left')
        ax1.grid(True, alpha=0.2, axis='y')
        ax1.axhline(y=0, color='white', linewidth=1, alpha=0.5)
        
        # 2. Retorno vs Risco (Efficient Frontier View)
        ax2 = fig.add_subplot(gs[1, 0])
        
        for i, asset in enumerate(self.returns.columns):
            ret = self.metrics[asset]['mean']
            vol = self.metrics[asset]['vol']
            sharpe = self.metrics[asset]['sharpe']
            
            color = ASSET_COLORS[i % len(ASSET_COLORS)]
            size = 200 + (sharpe * 100 if sharpe > 0 else 0)
            
            ax2.scatter(vol, ret, s=size, alpha=0.7, color=color,
                       edgecolors='white', linewidth=2, label=asset)
            ax2.text(vol, ret, f'  {asset}\n  SR:{sharpe:.2f}',
                    fontsize=9, va='center')
        
        # Linha de Sharpe = 1
        vol_range = np.linspace(0, max([self.metrics[a]['vol'] for a in self.returns.columns]), 100)
        ax2.plot(vol_range, 0.1175 + vol_range * 1.0, 'w--', 
                linewidth=2, alpha=0.5, label='Sharpe=1.0')
        
        ax2.set_xlabel('Volatilidade Anualizada', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Retorno Anualizado', fontsize=11, fontweight='bold')
        ax2.set_title('🎯 FRONTEIRA RISCO-RETORNO', fontsize=13,
                     fontweight='bold', color=COLORS['accent2'], pad=10)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax2.grid(True, alpha=0.2)
        ax2.legend(loc='best', fontsize=9)
        
        # 3. Drawdown Statistics
        ax3 = fig.add_subplot(gs[1, 1])
        
        max_dds = [abs(self.metrics[asset]['max_dd']) for asset in self.returns.columns]
        avg_dds = []
        dd_durations = []
        
        for asset in self.returns.columns:
            returns = self.returns[asset].dropna()
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            avg_dds.append(abs(drawdown[drawdown < 0].mean()))
            
            # Duração média de drawdown
            in_dd = drawdown < -0.01
            dd_periods = in_dd.astype(int).groupby((in_dd != in_dd.shift()).cumsum()).sum()
            dd_durations.append(dd_periods[dd_periods > 0].mean() if len(dd_periods[dd_periods > 0]) > 0 else 0)
        
        x_pos = np.arange(len(self.returns.columns))
        ax3.barh(x_pos, max_dds, height=0.4, alpha=0.7, 
                color=COLORS['accent4'], label='Max Drawdown')
        ax3.barh(x_pos + 0.4, avg_dds, height=0.4, alpha=0.7,
                color=COLORS['accent5'], label='Avg Drawdown')
        
        ax3.set_yticks(x_pos + 0.2)
        ax3.set_yticklabels(self.returns.columns)
        ax3.set_xlabel('Drawdown Magnitude', fontsize=11, fontweight='bold')
        ax3.set_title('📉 ESTATÍSTICAS DE DRAWDOWN', fontsize=13,
                     fontweight='bold', color=COLORS['accent4'], pad=10)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax3.legend()
        ax3.grid(True, alpha=0.2, axis='x')
        
        # 4. Underwater Plot (Portfolio)
        ax4 = fig.add_subplot(gs[2, :])
        
        cumulative = (1 + self.portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        ax4.fill_between(drawdown.index, drawdown.values, 0,
                        color=COLORS['accent4'], alpha=0.4)
        ax4.plot(drawdown.index, drawdown.values, 
                color=COLORS['accent4'], linewidth=2, alpha=0.9)
        
        # Top 5 drawdowns
        dd_periods = []
        in_dd = False
        start_idx = None
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < -0.01 and not in_dd:
                in_dd = True
                start_idx = i
            elif drawdown.iloc[i] >= 0 and in_dd:
                in_dd = False
                if start_idx is not None:
                    period_dd = drawdown.iloc[start_idx:i]
                    if len(period_dd) > 0:
                        dd_periods.append({
                            'start': drawdown.index[start_idx],
                            'end': drawdown.index[i-1],
                            'depth': period_dd.min(),
                            'duration': len(period_dd)
                        })
        
        # Marcar top 3 drawdowns
        dd_periods_sorted = sorted(dd_periods, key=lambda x: x['depth'])[:3]
        for i, dd in enumerate(dd_periods_sorted):
            ax4.axvspan(dd['start'], dd['end'], alpha=0.2, 
                       color=ASSET_COLORS[i], label=f"DD{i+1}: {dd['depth']:.1%}")
        
        ax4.set_title('🌊 UNDERWATER PLOT - Portfolio (Top 3 Drawdowns)', 
                     fontsize=14, fontweight='bold', color=COLORS['accent4'], pad=15)
        ax4.set_ylabel('Drawdown', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Data', fontsize=11, fontweight='bold')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax4.legend(loc='lower left')
        ax4.grid(True, alpha=0.2)
        
        # 5. Rolling Sharpe Ratio
        ax5 = fig.add_subplot(gs[3, 0])
        window = 252
        
        for i, col in enumerate(self.returns.columns):
            returns = self.returns[col]
            rolling_sharpe = (returns.rolling(window).mean() * 252 - 0.1175) / \
                           (returns.rolling(window).std() * np.sqrt(252))
            
            ax5.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2.5,
                    label=col, color=ASSET_COLORS[i % len(ASSET_COLORS)], alpha=0.85)
        
        ax5.axhline(y=1, color='white', linestyle='--', linewidth=2, 
                   alpha=0.5, label='Sharpe=1')
        ax5.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.3)
        
        ax5.set_title(f'📈 ROLLING SHARPE RATIO ({window}d)', fontsize=13,
                     fontweight='bold', color=COLORS['accent1'], pad=10)
        ax5.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Data', fontsize=11, fontweight='bold')
        ax5.legend(ncol=2, loc='best', fontsize=9)
        ax5.grid(True, alpha=0.2)
        
        # 6. Skewness and Kurtosis Analysis
        ax6 = fig.add_subplot(gs[3, 1])
        
        skews = [self.metrics[asset]['skewness'] for asset in self.returns.columns]
        kurts = [self.metrics[asset]['kurtosis'] for asset in self.returns.columns]
        
        for i, asset in enumerate(self.returns.columns):
            ax6.scatter(skews[i], kurts[i], s=250, alpha=0.7,
                       color=ASSET_COLORS[i % len(ASSET_COLORS)],
                       edgecolors='white', linewidth=2, label=asset)
            ax6.text(skews[i], kurts[i], f'  {asset}', fontsize=9, va='center')
        
        # Regiões de referência
        ax6.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax6.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax6.axhline(y=3, color=COLORS['accent1'], linestyle=':', 
                   linewidth=2, alpha=0.5, label='Kurt Normal=3')
        
        ax6.set_xlabel('Skewness (Assimetria)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Excess Kurtosis', fontsize=11, fontweight='bold')
        ax6.set_title('📊 ANÁLISE DE MOMENTOS (Skew vs Kurt)', fontsize=13,
                     fontweight='bold', color=COLORS['accent3'], pad=10)
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.2)
        
        plt.suptitle('🏆 ANÁLISE DE PERFORMANCE E ATRIBUIÇÃO', fontsize=18,
                    fontweight='bold', color=COLORS['accent1'], y=0.998)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=COLORS['background'])
            print(f"💾 Análise de performance salva: {save_path}")
        
        plt.show()
        return fig
    
    def plot_interactive_dashboard(self, save_path=None):
        """Dashboard interativo completo com Plotly"""
        
        # Calcular todas as métricas
        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - 0.1175) / annual_volatility
        sortino_ratios = [self.metrics[asset]['sortino'] for asset in self.returns.columns]
        max_drawdowns = [self.metrics[asset]['max_dd'] for asset in self.returns.columns]
        var_95 = [self.metrics[asset]['var_95'] for asset in self.returns.columns]
        cvar_95 = [self.metrics[asset]['cvar_95'] for asset in self.returns.columns]
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '🎯 Risk-Return Scatter',
                '⚡ Sharpe vs Sortino',
                '📊 Volatility Breakdown',
                '🔻 VaR vs CVaR',
                '💥 Maximum Drawdown',
                '📈 Cumulative Returns',
                '🔥 Correlation Matrix',
                '🌊 Rolling Volatility',
                '📉 Drawdown Timeline'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "heatmap"}, {"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Risk-Return Scatter
        for i, asset in enumerate(self.returns.columns):
            fig.add_trace(
                go.Scatter(
                    x=[annual_volatility[asset]],
                    y=[annual_returns[asset]],
                    mode='markers+text',
                    marker=dict(size=25, color=ASSET_COLORS[i], 
                               line=dict(color='white', width=2)),
                    text=asset,
                    textposition="top center",
                    textfont=dict(size=10, color='white'),
                    name=asset,
                    hovertemplate=f'<b>{asset}</b><br>Vol: %{{x:.2%}}<br>Ret: %{{y:.2%}}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Sharpe vs Sortino
        fig.add_trace(
            go.Bar(
                x=list(self.returns.columns),
                y=sharpe_ratios.values,
                name='Sharpe',
                marker=dict(color=ASSET_COLORS[:len(self.returns.columns)], 
                           opacity=0.7, line=dict(color='white', width=1.5)),
                text=[f'{x:.2f}' for x in sharpe_ratios.values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=list(self.returns.columns),
                y=sortino_ratios,
                name='Sortino',
                marker=dict(color=ASSET_COLORS[:len(self.returns.columns)],
                           opacity=0.9, line=dict(color='white', width=1.5),
                           pattern_shape="/"),
                text=[f'{x:.2f}' for x in sortino_ratios],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Sortino: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Volatility Breakdown
        vol_decomp = annual_volatility.values
        fig.add_trace(
            go.Scatterpolar(
                r=vol_decomp,
                theta=list(self.returns.columns),
                fill='toself',
                fillcolor=COLORS['accent1'],
                opacity=0.6,
                line=dict(color=COLORS['accent1'], width=2),
                name='Volatility',
                hovertemplate='<b>%{theta}</b><br>Vol: %{r:.2%}<extra></extra>'
            ),
            row=1, col=3
        )
        
        # 4. VaR vs CVaR
        x_pos = list(self.returns.columns)
        fig.add_trace(
            go.Bar(
                x=x_pos,
                y=var_95,
                name='VaR 95%',
                marker=dict(color=COLORS['accent4'], opacity=0.7,
                           line=dict(color='white', width=1.5)),
                text=[f'{x:.2%}' for x in var_95],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>VaR: %{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=x_pos,
                y=cvar_95,
                name='CVaR 95%',
                marker=dict(color='darkred', opacity=0.7,
                           line=dict(color='white', width=1.5)),
                text=[f'{x:.2%}' for x in cvar_95],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>CVaR: %{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 5. Maximum Drawdown
        fig.add_trace(
            go.Bar(
                x=list(self.returns.columns),
                y=[abs(x) for x in max_drawdowns],
                marker=dict(color=ASSET_COLORS[:len(self.returns.columns)],
                           opacity=0.8, line=dict(color='white', width=1.5)),
                text=[f'{abs(x):.1%}' for x in max_drawdowns],
                textposition='outside',
                name='Max DD',
                hovertemplate='<b>%{x}</b><br>Max DD: %{y:.2%}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 6. Cumulative Returns
        for i, col in enumerate(self.returns.columns):
            cumulative = (1 + self.returns[col]).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative.values,
                    mode='lines',
                    name=col,
                    line=dict(color=ASSET_COLORS[i % len(ASSET_COLORS)], width=2.5),
                    hovertemplate='<b>%{fullData.name}</b><br>%{y:.2%}<extra></extra>'
                ),
                row=2, col=3
            )
        
        # 7. Correlation Heatmap
        corr_matrix = self.returns.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                colorscale='RdBu_r',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlação"),
                hovertemplate='%{y} vs %{x}<br>Corr: %{z:.3f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 8. Rolling Volatility
        window = 63
        for i, col in enumerate(self.returns.columns):
            rolling_vol = self.returns[col].rolling(window).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name=col,
                    line=dict(color=ASSET_COLORS[i % len(ASSET_COLORS)], width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>Vol: %{y:.2%}<extra></extra>'
                ),
                row=3, col=2
            )
        
        # 9. Portfolio Drawdown
        portfolio_cumulative = (1 + self.portfolio_returns).cumprod()
        rolling_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - rolling_max) / rolling_max
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                fill='tozeroy',
                name='Portfolio DD',
                line=dict(color=COLORS['accent4'], width=2),
                fillcolor=COLORS['accent4'],
                opacity=0.6,
                hovertemplate='Drawdown: %{y:.2%}<extra></extra>'
            ),
            row=3, col=3
        )
        
        # Layout atualizado
        fig.update_layout(
            height=1400,
            title={
                'text': '📊 QUANTITATIVE RISK ANALYTICS - COMPREHENSIVE DASHBOARD',
                'font': {'size': 26, 'color': COLORS['accent1'], 
                        'family': 'Arial Black'},
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.995
            },
            showlegend=True,
            plot_bgcolor=COLORS['paper'],
            paper_bgcolor=COLORS['background'],
            font=dict(color=COLORS['text'], size=11),
            hovermode='closest',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=1.15,
                bgcolor=COLORS['paper'],
                bordercolor=COLORS['grid'],
                borderwidth=2
            )
        )
        
        # Atualizar todos os eixos
        fig.update_xaxes(
            title_font=dict(size=11, color=COLORS['text']),
            tickfont=dict(size=9, color=COLORS['text']),
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False
        )
        
        fig.update_yaxes(
            title_font=dict(size=11, color=COLORS['text']),
            tickfont=dict(size=9, color=COLORS['text']),
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False
        )
        
        # Títulos específicos
        fig.update_xaxes(title_text="Volatilidade", row=1, col=1)
        fig.update_yaxes(title_text="Retorno", row=1, col=1)
        fig.update_yaxes(title_text="Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Perda", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=2)
        fig.update_yaxes(title_text="Retorno Acum.", row=2, col=3)
        fig.update_yaxes(title_text="Volatilidade", row=3, col=2)
        fig.update_yaxes(title_text="Drawdown", row=3, col=3)
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Dashboard interativo salvo: {save_path}")
        
        fig.show()
        return fig
    
    def generate_metrics_report(self, save_path=None):
        """Gera relatório completo de métricas em formato de tabela"""
        print("\n" + "="*100)
        print("📊 RELATÓRIO QUANTITATIVO DE MÉTRICAS DE RISCO")
        print("="*100 + "\n")
        
        # Criar DataFrame com todas as métricas
        metrics_data = []
        for asset in self.returns.columns:
            m = self.metrics[asset]
            metrics_data.append({
                'Asset': asset,
                'Ret. Anual': f"{m['mean']:.2%}",
                'Vol. Anual': f"{m['vol']:.2%}",
                'Sharpe': f"{m['sharpe']:.3f}",
                'Sortino': f"{m['sortino']:.3f}",
                'Calmar': f"{m['calmar']:.3f}",
                'Omega': f"{m['omega']:.3f}",
                'Skewness': f"{m['skewness']:.3f}",
                'Kurtosis': f"{m['kurtosis']:.3f}",
                'VaR 95%': f"{m['var_95']:.3%}",
                'CVaR 95%': f"{m['cvar_95']:.3%}",
                'Max DD': f"{m['max_dd']:.2%}",
                'Tail Ratio': f"{m['tail_ratio']:.3f}",
                'JB p-value': f"{m['jb_pvalue']:.4f}"
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Exibir tabela formatada
        print(df_metrics.to_string(index=False))
        print("\n" + "="*100)
        
        # Salvar se especificado
        if save_path:
            df_metrics.to_csv(save_path, index=False)
            print(f"\n💾 Relatório salvo em: {save_path}")
        
        return df_metrics
    
    def create_all_visualizations(self):
        """Cria todas as visualizações quantitativas automaticamente"""
        print("\n" + "="*100)
        print("🚀 GERANDO ANÁLISE QUANTITATIVA COMPLETA - PROFESSIONAL DARK THEME")
        print("="*100 + "\n")
        
        os.makedirs('reports/figures', exist_ok=True)
        
        visualizations = [
            ('01_comprehensive_price_analysis.png', self.plot_comprehensive_price_analysis),
            ('02_advanced_distribution_analysis.png', self.plot_advanced_distribution_analysis),
            ('03_correlation_and_cointegration.png', self.plot_correlation_and_cointegration),
            ('04_volatility_regime_analysis.png', self.plot_volatility_regime_analysis),
            ('05_tail_risk_analysis.png', self.plot_tail_risk_analysis),
            ('06_performance_attribution.png', self.plot_performance_attribution),
            ('07_interactive_dashboard.html', self.plot_interactive_dashboard)
        ]
        
        for i, (filename, plot_function) in enumerate(visualizations, 1):
            save_path = f'reports/figures/{filename}'
            try:
                print(f"[{i}/{len(visualizations)}] Gerando {filename}...", end=" ")
                plot_function(save_path=save_path)
                print("✅")
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        # Gerar relatório de métricas
        print(f"\n[8/8] Gerando relatório de métricas...", end=" ")
        try:
            self.generate_metrics_report(save_path='reports/figures/08_metrics_report.csv')
            print("✅")
        except Exception as e:
            print(f"❌ Erro: {e}")
        
        print("\n" + "="*100)
        print("🎯 ANÁLISE QUANTITATIVA CONCLUÍDA COM SUCESSO!")
        print("="*100)
        print("\n📁 Arquivos gerados em: reports/figures/")
        print("\n📊 Visualizações criadas:")
        print("   ✓ 01 - Análise Completa de Preços (Bollinger, MA, Bandas)")
        print("   ✓ 02 - Distribuições Avançadas (KDE, Student-t, Testes)")
        print("   ✓ 03 - Correlação e Cointegração (Pearson, Spearman, Rolling)")
        print("   ✓ 04 - Regimes de Volatilidade (EWMA, Cone, Clustering)")
        print("   ✓ 05 - Análise de Tail Risk (VaR, CVaR, Hill Estimator)")
        print("   ✓ 06 - Performance Attribution (Sharpe, Sortino, Drawdowns)")
        print("   ✓ 07 - Dashboard Interativo Plotly (9 painéis)")
        print("   ✓ 08 - Relatório CSV com 13+ métricas por ativo")
        print("\n💡 Dica: Abra o arquivo .html no navegador para dashboard interativo!")
        print("📈 Próxima etapa: Simulações de Monte Carlo")
        print("="*100 + "\n")

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
    """Função principal"""
    print("\n" + "="*100)
    print("🎯 ADVANCED QUANTITATIVE RISK VISUALIZER - DARK THEME")
    print("   Desenvolvido para Quantitative Analysts")
    print("="*100)
    
    returns, prices = load_data_for_visualization()
    
    if returns is not None and prices is not None:
        # Inicializar visualizador avançado
        visualizer = AdvancedRiskVisualizer(returns, prices)
        
        # Criar todas as visualizações
        visualizer.create_all_visualizations()
        
        print("\n💼 PORTFOLIO PARA ENTREVISTAS:")
        print("   ✓ 7 visualizações estáticas de alta resolução (300 DPI)")
        print("   ✓ 1 dashboard interativo Plotly com 9 painéis")
        print("   ✓ 1 relatório CSV com métricas quantitativas")
        print("   ✓ Tema escuro profissional")
        print("   ✓ 50+ métricas calculadas automaticamente")
        print("\n🏆 DESTAQUES TÉCNICOS:")
        print("   • Análise de distribuições (KDE, Student-t, Q-Q Plot)")
        print("   • Testes estatísticos (Jarque-Bera, Shapiro-Wilk)")
        print("   • Métricas avançadas (Sharpe, Sortino, Calmar, Omega)")
        print("   • Tail Risk (VaR, CVaR, Hill Estimator, EVT)")
        print("   • Volatility clustering e regimes")
        print("   • Correlações múltiplas (Pearson, Spearman, Rolling)")
        print("   • Performance attribution completa")
        print("   • Autocorrelação e Beta dinâmico")
        print("\n" + "="*100 + "\n")
        
    else:
        print("❌ Não foi possível carregar os dados para visualização")

if __name__ == "__main__":
    main()
