import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class MonteCarloSimulator:
    def __init__(self, returns_df, num_simulations=1000, time_horizon=252):
        """
        Simulador de Monte Carlo para análise de risco
        
        Parameters:
        returns_df: DataFrame com retornos históricos
        num_simulations: Número de simulações
        time_horizon: Horizonte de tempo em dias
        """
        self.returns = returns_df
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon
        self.portfolio_returns = returns_df.mean(axis=1)  # Portfolio equal-weight
        
        # Estatísticas dos retornos
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
    def simulate_portfolio_paths(self, initial_value=1000):
        """Simula caminhos de preços do portfólio usando Cholesky decomposition"""
        print("🎲 SIMULANDO CAMINHOS DE PREÇOS DO PORTFÓLIO...")
        
        # Gerar retornos aleatórios correlacionados
        L = np.linalg.cholesky(self.cov_matrix)
        random_returns = np.random.normal(0, 1, (self.time_horizon, self.num_simulations, len(self.returns.columns)))
        
        # Aplicar correlação
        correlated_returns = np.dot(random_returns, L.T) + self.mean_returns.values
        
        # Calcular caminhos de preços para cada ativo
        asset_paths = np.zeros((self.time_horizon + 1, self.num_simulations, len(self.returns.columns)))
        asset_paths[0] = initial_value / len(self.returns.columns)  # Distribuir igualmente
        
        for t in range(1, self.time_horizon + 1):
            asset_paths[t] = asset_paths[t-1] * (1 + correlated_returns[t-1])
        
        # Portfolio total (soma de todos os ativos)
        portfolio_paths = asset_paths.sum(axis=2)
        
        self.portfolio_paths = portfolio_paths
        self.asset_paths = asset_paths
        
        print(f"✅ {self.num_simulations} simulações realizadas")
        return portfolio_paths
    
    def calculate_risk_metrics(self, confidence_level=0.95):
        """Calcula métricas de risco baseadas nas simulações"""
        print("📊 CALCULANDO MÉTRICAS DE RISCO DAS SIMULAÇÕES...")
        
        if not hasattr(self, 'portfolio_paths'):
            self.simulate_portfolio_paths()
        
        final_values = self.portfolio_paths[-1]
        returns_simulated = (final_values / self.portfolio_paths[0, 0] - 1)
        
        # Métricas básicas
        mean_final_value = np.mean(final_values)
        median_final_value = np.median(final_values)
        std_final_value = np.std(final_values)
        
        # Value at Risk e Conditional VaR
        var = np.percentile(returns_simulated, (1 - confidence_level) * 100)
        cvar = returns_simulated[returns_simulated <= var].mean()
        
        # Probabilidades
        prob_loss = (returns_simulated < 0).mean()
        prob_20_percent_gain = (returns_simulated > 0.20).mean()
        prob_50_percent_loss = (returns_simulated < -0.50).mean()
        
        metrics = {
            'Valor_Final_Medio': mean_final_value,
            'Valor_Final_Mediano': median_final_value,
            'Desvio_Padrao_Final': std_final_value,
            f'VaR_{confidence_level:.0%}': var,
            f'CVaR_{confidence_level:.0%}': cvar,
            'Probabilidade_Prejuizo': prob_loss,
            'Probabilidade_Ganho_20%': prob_20_percent_gain,
            'Probabilidade_Perda_50%': prob_50_percent_loss,
            'Retorno_Medio_Simulado': returns_simulated.mean(),
            'Retorno_Minimo_Simulado': returns_simulated.min(),
            'Retorno_Maximo_Simulado': returns_simulated.max()
        }
        
        return metrics, returns_simulated
    
    def plot_simulation_results(self, save_path=None):
        """Plota resultados das simulações de Monte Carlo"""
        print("📈 GERANDO VISUALIZAÇÕES DAS SIMULAÇÕES...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ANÁLISE DE RISCO - SIMULAÇÃO DE MONTE CARLO', fontsize=16, fontweight='bold')
        
        # 1. Caminhos de simulação
        axes[0, 0].plot(self.portfolio_paths, alpha=0.1, color='blue')
        axes[0, 0].plot(np.median(self.portfolio_paths, axis=1), color='red', linewidth=2, label='Mediana')
        axes[0, 0].plot(np.percentile(self.portfolio_paths, 5, axis=1), '--', color='orange', label='5º Percentil')
        axes[0, 0].plot(np.percentile(self.portfolio_paths, 95, axis=1), '--', color='orange', label='95º Percentil')
        axes[0, 0].set_title('Caminhos de Simulação do Portfólio')
        axes[0, 0].set_ylabel('Valor do Portfólio (R$)')
        axes[0, 0].set_xlabel('Dias')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribuição dos valores finais
        final_values = self.portfolio_paths[-1]
        axes[0, 1].hist(final_values, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(np.mean(final_values), color='red', linestyle='--', label=f'Média: R$ {np.mean(final_values):.0f}')
        axes[0, 1].axvline(np.median(final_values), color='green', linestyle='--', label=f'Mediana: R$ {np.median(final_values):.0f}')
        axes[0, 1].set_title('Distribuição dos Valores Finais')
        axes[0, 1].set_xlabel('Valor Final do Portfólio (R$)')
        axes[0, 1].set_ylabel('Densidade')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Probabilidade acumulada
        sorted_returns = np.sort(final_values)
        cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        axes[1, 0].plot(sorted_returns, cdf, linewidth=2, color='purple')
        axes[1, 0].set_title('Função de Distribuição Acumulada (CDF)')
        axes[1, 0].set_xlabel('Valor Final do Portfólio (R$)')
        axes[1, 0].set_ylabel('Probabilidade Acumulada')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Métricas de risco
        metrics, returns_simulated = self.calculate_risk_metrics()
        
        metric_names = ['Probabilidade_Prejuizo', 'Probabilidade_Ganho_20%', 'Probabilidade_Perda_50%']
        metric_values = [metrics[name] for name in metric_names]
        metric_labels = ['Prob. Prejuízo', 'Prob. Ganho >20%', 'Prob. Perda >50%']
        
        bars = axes[1, 1].bar(metric_labels, metric_values, color=['red', 'green', 'orange'], alpha=0.7)
        axes[1, 1].set_title('Probabilidades de Cenários')
        axes[1, 1].set_ylabel('Probabilidade')
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.1%}', ha='center', va='bottom')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Gráfico salvo: {save_path}")
        
        plt.show()
        return fig
    
    def stress_testing(self, shock_scenarios):
        """
        Teste de estresse com cenários específicos
        
        Parameters:
        shock_scenarios: Dict com cenários de choque {ativo: magnitude_choque}
        """
        print("🌪️  REALIZANDO TESTE DE ESTRESSE...")
        
        stressed_returns = self.returns.copy()
        
        for asset, shock in shock_scenarios.items():
            if asset in stressed_returns.columns:
                # Aplicar choque negativo aos retornos
                stressed_returns[asset] = stressed_returns[asset] + shock
        
        # Recalcular estatísticas com choque
        stressed_mean = stressed_returns.mean()
        stressed_cov = stressed_returns.cov()
        
        # Simular com dados estressados
        L_stressed = np.linalg.cholesky(stressed_cov)
        random_returns = np.random.normal(0, 1, (self.time_horizon, self.num_simulations, len(self.returns.columns)))
        correlated_returns = np.dot(random_returns, L_stressed.T) + stressed_mean.values
        
        asset_paths_stressed = np.zeros((self.time_horizon + 1, self.num_simulations, len(self.returns.columns)))
        asset_paths_stressed[0] = 1000 / len(self.returns.columns)
        
        for t in range(1, self.time_horizon + 1):
            asset_paths_stressed[t] = asset_paths_stressed[t-1] * (1 + correlated_returns[t-1])
        
        portfolio_paths_stressed = asset_paths_stressed.sum(axis=2)
        
        # Comparar resultados
        normal_final = np.median(self.portfolio_paths[-1])
        stressed_final = np.median(portfolio_paths_stressed[-1])
        impact = (stressed_final - normal_final) / normal_final
        
        stress_results = {
            'Cenarios_Testados': shock_scenarios,
            'Valor_Final_Normal': normal_final,
            'Valor_Final_Estressado': stressed_final,
            'Impacto_Percentual': impact,
            'Perda_Absoluta': stressed_final - normal_final
        }
        
        return stress_results
    
    def generate_monte_carlo_report(self):
        """Gera relatório completo da simulação de Monte Carlo"""
        print("📋 GERANDO RELATÓRIO DE MONTE CARLO...")
        print("=" * 70)
        
        # Simular caminhos
        self.simulate_portfolio_paths()
        
        # Calcular métricas
        metrics, returns_simulated = self.calculate_risk_metrics()
        
        print("\n🎯 RESULTADOS DA SIMULAÇÃO DE MONTE CARLO")
        print("=" * 70)
        
        print(f"\n📊 MÉTRICAS PRINCIPAIS ({self.num_simulations} simulações, {self.time_horizon} dias):")
        for key, value in metrics.items():
            if 'Probabilidade' in key:
                print(f"   • {key:30}: {value:8.2%}")
            elif 'Retorno' in key:
                print(f"   • {key:30}: {value:8.2%}")
            elif 'VaR' in key or 'CVaR' in key:
                print(f"   • {key:30}: {value:8.2%}")
            else:
                print(f"   • {key:30}: R$ {value:8.0f}")
        
        # Teste de estresse
        print(f"\n🌪️  TESTE DE ESTRESSE (Cenários Adversos):")
        stress_scenarios = {
            'USD_BRL': -0.10,  # Dólar +10%
            'SELIC': 0.05,     # Juros +5%
            'PETROLEO_BRENT': -0.15  # Petróleo -15%
        }
        
        stress_results = self.stress_testing(stress_scenarios)
        print(f"   • Cenários: {stress_results['Cenarios_Testados']}")
        print(f"   • Impacto no portfólio: {stress_results['Impacto_Percentual']:.2%}")
        print(f"   • Perda absoluta: R$ {stress_results['Perda_Absoluta']:.0f}")
        
        # Plotar resultados
        self.plot_simulation_results('reports/figures/monte_carlo_results.png')
        
        print(f"\n📈 VISUALIZAÇÕES SALVAS EM: reports/figures/monte_carlo_results.png")
        
        return {
            'metrics': metrics,
            'stress_test': stress_results,
            'simulated_returns': returns_simulated
        }

# Funções auxiliares
def load_data_for_simulation():
    """Carrega dados para simulação"""
    try:
        returns = pd.read_parquet('data/processed/macro_portfolio_returns.parquet')
        print(f"✅ Dados carregados: {returns.shape}")
        return returns
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None

def main():
    """Função principal"""
    print("🚀 SIMULADOR DE MONTE CARLO PARA ANÁLISE DE RISCO")
    print("=" * 70)
    
    # Carregar dados
    returns = load_data_for_simulation()
    
    if returns is not None:
        # Inicializar simulador
        simulator = MonteCarloSimulator(returns, num_simulations=1000, time_horizon=252)
        
        # Gerar relatório completo
        report = simulator.generate_monte_carlo_report()
        
        print("\n" + "=" * 70)
        print("🎯 SIMULAÇÕES DE MONTE CARLO CONCLUÍDAS!")
        print("📊 PRÓXIMA ETAPA: Dashboard Interativo com Streamlit")
        print("💡 Execute: streamlit run dashboard/app.py")
        print("=" * 70)
        
    else:
        print("❌ Não foi possível carregar os dados para simulação")

if __name__ == "__main__":
    main()
