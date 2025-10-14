import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import sys

# Adicionar o diretório raiz ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

warnings.filterwarnings('ignore')

# Tentar importar arch e statsmodels, mas continuar mesmo se falhar
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    print("⚠️  arch não instalado, usando GARCH simplificado")
    HAS_ARCH = False

try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    HAS_STATSMODELS = True
except ImportError:
    print("⚠️  statsmodels não instalado, usando métodos alternativos")
    HAS_STATSMODELS = False

class AdvancedRiskSimulators:
    """Coleção de algoritmos avançados de simulação para risco"""
    
    def __init__(self, returns_df):
        self.returns = returns_df
        self.portfolio_returns = returns_df.mean(axis=1)
    
    # 1. BOOTSTRAPPING
    def historical_bootstrapping(self, num_simulations=1000, block_length=5):
        """Bootstrapping com blocos para preservar estrutura temporal"""
        print("🔄 Executando Bootstrapping...")
        
        n_obs = len(self.portfolio_returns)
        simulated_paths = []
        
        for sim in range(num_simulations):
            path = []
            i = 0
            while i < n_obs:
                start_idx = np.random.randint(0, n_obs - block_length)
                block = self.portfolio_returns.iloc[start_idx:start_idx + block_length]
                path.extend(block.values)
                i += block_length
            
            # Manter mesmo comprimento
            simulated_paths.append(path[:n_obs])
        
        paths_array = np.array(simulated_paths)
        
        # Calcular métricas
        final_values = (1 + paths_array).prod(axis=1)
        metrics = {
            'Retorno_Medio_Simulado': final_values.mean() - 1,
            'Volatilidade_Simulada': paths_array.std(axis=1).mean() * np.sqrt(252),
            'Prob_Prejuizo': (final_values < 1).mean(),
            'VaR_95%': np.percentile(final_values - 1, 5)
        }
        
        return paths_array, metrics
    
    # 2. MODELO DE MERTON (SALTOS)
    def merton_jump_diffusion(self, S0=100, T=1, num_simulations=1000):
        """Modelo de Merton com saltos para eventos extremos"""
        print("🌊 Executando Merton Jump Diffusion...")
        
        # Estimar parâmetros dos retornos
        returns = self.portfolio_returns
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        
        # Estimar parâmetros de saltos (simplificado)
        threshold = 2 * sigma / np.sqrt(252)  # Retornos extremos
        extreme_returns = returns[np.abs(returns) > threshold]
        lambd = len(extreme_returns) / len(returns) * 252  # Intensidade anual
        mu_j = extreme_returns.mean() if len(extreme_returns) > 0 else 0
        sigma_j = extreme_returns.std() if len(extreme_returns) > 0 else sigma * 2
        
        dt = 1/252
        n_steps = int(T/dt)
        
        paths = np.zeros((n_steps + 1, num_simulations))
        paths[0] = S0
        
        for t in range(1, n_steps + 1):
            # Componente Browniano
            Z = np.random.normal(0, 1, num_simulations)
            
            # Componente de saltos
            N = np.random.poisson(lambd * dt, num_simulations)
            jumps = np.random.normal(mu_j, sigma_j, num_simulations) * N
            
            paths[t] = paths[t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + 
                sigma * np.sqrt(dt) * Z + 
                jumps
            )
        
        returns_simulated = (paths[-1] / S0 - 1)
        metrics = {
            'Retorno_Medio_Simulado': returns_simulated.mean(),
            'Volatilidade_Simulada': returns_simulated.std(),
            'Prob_Prejuizo': (returns_simulated < 0).mean(),
            'Prob_Evento_Extremo': (np.abs(returns_simulated) > 0.3).mean()
        }
        
        return paths, metrics
    
    # 3. GARCH PARA VOLATILIDADE ESTOCÁSTICA
    def garch_simulation(self, num_simulations=1000):
        """Simulação GARCH com volatilidade variável no tempo"""
        print("📈 Executando Simulação GARCH...")
        
        if HAS_ARCH:
            try:
                # Estimar modelo GARCH(1,1)
                model = arch_model(self.portfolio_returns * 100, vol='Garch', p=1, q=1)
                fitted_model = model.fit(disp='off')
                
                omega = fitted_model.params['omega'] / 10000
                alpha = fitted_model.params['alpha[1]']
                beta = fitted_model.params['beta[1]']
                mu = self.portfolio_returns.mean()
                
            except Exception as e:
                print(f"⚠️  GARCH falhou: {e}, usando parâmetros simplificados")
                omega, alpha, beta, mu = self._get_simple_garch_params()
        else:
            omega, alpha, beta, mu = self._get_simple_garch_params()
        
        n_steps = 252
        paths = np.zeros((n_steps, num_simulations))
        
        for i in range(num_simulations):
            returns_sim = []
            h_t = self.portfolio_returns.var()  # Volatilidade inicial
            
            for t in range(n_steps):
                # Atualizar volatilidade GARCH
                z_t = np.random.normal(0, 1)
                h_t = omega + alpha * (self.portfolio_returns.iloc[-1]**2 if t == 0 else returns_sim[-1]**2) + beta * h_t
                
                # Gerar retorno
                return_t = mu + np.sqrt(h_t) * z_t
                returns_sim.append(return_t)
            
            paths[:, i] = returns_sim
        
        # Converter para preços
        price_paths = np.zeros((n_steps + 1, num_simulations))
        price_paths[0] = 100
        for t in range(1, n_steps + 1):
            price_paths[t] = price_paths[t-1] * (1 + paths[t-1])
        
        final_returns = (price_paths[-1] / 100 - 1)
        metrics = {
            'Retorno_Medio_Simulado': final_returns.mean(),
            'Volatilidade_Media_Simulada': paths.std(axis=0).mean() * np.sqrt(252),
            'Prob_Prejuizo': (final_returns < 0).mean(),
            'Persistencia_Volatilidade': alpha + beta  # Quão persistente é a vol
        }
        
        return price_paths, metrics
    
    def _get_simple_garch_params(self):
        """Parâmetros GARCH simplificados quando o pacote não está disponível"""
        returns = self.portfolio_returns
        return 0.0001, 0.1, 0.85, returns.mean()
    
    # 4. CÓPULA GAUSSIANA (MULTIVARIADA)
    def gaussian_copula_simulation(self, num_simulations=1000):
        """Simulação multivariada usando cópula Gaussiana"""
        print("🎯 Executando Cópula Gaussiana...")
        
        # Transformar para uniformes usando CDF empírica
        uniform_data = self.returns.apply(
            lambda x: (x.rank() - 0.5) / len(x)
        )
        
        # Transformar para normais
        normal_data = uniform_data.apply(stats.norm.ppf)
        
        # Matriz de correlação
        corr_matrix = normal_data.corr()
        
        # Simular da cópula
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # Se matriz não for positiva definida, usar correlação aproximada
            print("⚠️  Matriz não é positiva definida, usando correlação diagonal")
            L = np.linalg.cholesky(corr_matrix + np.eye(len(corr_matrix)) * 0.01)
        
        Z = np.random.normal(0, 1, (num_simulations, len(self.returns.columns)))
        correlated_normals = Z @ L.T
        
        # Voltar para uniformes
        uniform_sim = stats.norm.cdf(correlated_normals)
        
        # Voltar para distribuição original
        simulated_returns = []
        for i, col in enumerate(self.returns.columns):
            sorted_returns = self.returns[col].sort_values()
            sim_col = np.interp(uniform_sim[:, i], 
                              np.linspace(0, 1, len(sorted_returns)),
                              sorted_returns)
            simulated_returns.append(sim_col)
        
        sim_df = pd.DataFrame(np.array(simulated_returns).T, 
                            columns=self.returns.columns,
                            index=range(num_simulations))
        
        # Calcular portfolio simulado
        portfolio_simulated = sim_df.mean(axis=1)
        
        metrics = {
            'Retorno_Medio_Simulado': portfolio_simulated.mean() * 252,
            'Volatilidade_Simulada': portfolio_simulated.std() * np.sqrt(252),
            'Prob_Prejuizo': (portfolio_simulated < 0).mean(),
            'Correlacao_Preservada': np.mean(np.abs(sim_df.corr().values - corr_matrix.values))  # Erro de correlação
        }
        
        return sim_df, metrics
    
    # 5. MONTE CARLO CLÁSSICO (PARA COMPARAÇÃO)
    def monte_carlo_baseline(self, initial_value=1000, num_simulations=1000, time_horizon=252):
        """Monte Carlo clássico para comparação"""
        print("🎲 Executando Monte Carlo Clássico...")
        
        # Estatísticas do portfolio
        portfolio_returns = self.portfolio_returns
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Simular caminhos
        simulations = np.random.normal(
            mean_return, 
            std_return, 
            (time_horizon, num_simulations)
        )
        
        paths = np.zeros((time_horizon + 1, num_simulations))
        paths[0] = initial_value
        
        for t in range(1, time_horizon + 1):
            paths[t] = paths[t-1] * (1 + simulations[t-1])
        
        # Calcular métricas
        final_values = paths[-1]
        returns_simulated = (final_values / initial_value - 1)
        
        metrics = {
            'Retorno_Medio_Simulado': returns_simulated.mean(),
            'Volatilidade_Simulada': returns_simulated.std(),
            'Prob_Prejuizo': (returns_simulated < 0).mean(),
            'VaR_95%': np.percentile(returns_simulated, 5),
            'CVaR_95%': returns_simulated[returns_simulated <= np.percentile(returns_simulated, 5)].mean()
        }
        
        return paths, metrics
    
    def compare_all_methods(self, initial_value=1000):
        """Compara todos os métodos de simulação"""
        print("🔍 COMPARANDO TODOS OS MÉTODOS DE SIMULAÇÃO...")
        print("=" * 60)
        
        methods = {
            'Monte Carlo Clássico': self.monte_carlo_baseline,
            'Bootstrapping': self.historical_bootstrapping,
            'Merton Jump': self.merton_jump_diffusion,
            'GARCH': self.garch_simulation,
            'Cópula Gaussiana': self.gaussian_copula_simulation
        }
        
        results = {}
        
        for name, method in methods.items():
            print(f"\n📊 Executando {name}...")
            try:
                if name == 'Monte Carlo Clássico':
                    paths, metrics = method(initial_value)
                else:
                    paths, metrics = method()
                
                results[name] = metrics
                print(f"✅ {name}:")
                print(f"   Retorno: {metrics.get('Retorno_Medio_Simulado', 0):.2%}")
                print(f"   Vol: {metrics.get('Volatilidade_Simulada', 0):.2%}")
                print(f"   Prob Prejuízo: {metrics.get('Prob_Prejuizo', 0):.2%}")
                
            except Exception as e:
                print(f"❌ {name} falhou: {e}")
                results[name] = {'Erro': str(e)}
        
        # Criar DataFrame comparativo
        comparison_df = pd.DataFrame(results).T
        
        print("\n" + "=" * 60)
        print("🎯 RESUMO DA COMPARAÇÃO:")
        print("=" * 60)
        print(comparison_df.round(4))
        
        return comparison_df

# Função auxiliar para carregar dados
def load_data_for_simulation():
    """Carrega dados para simulação com fallback para CSV"""
    try:
        # Primeiro tenta parquet
        returns = pd.read_parquet('data/processed/macro_portfolio_returns.parquet')
        print(f"✅ Dados carregados (parquet): {returns.shape}")
        return returns
    except Exception as e:
        print(f"⚠️  Erro com parquet: {e}")
        try:
            # Fallback para CSV
            returns = pd.read_csv('data/processed/macro_portfolio_returns.csv', index_col=0, parse_dates=True)
            print(f"✅ Dados carregados (CSV): {returns.shape}")
            return returns
        except Exception as e2:
            print(f"❌ Erro ao carregar dados: {e2}")
            # Criar dados de exemplo para teste
            print("🔄 Criando dados de exemplo para demonstração...")
            dates = pd.date_range('2020-01-01', periods=500, freq='D')
            np.random.seed(42)
            example_data = {
                'SELIC': np.random.normal(0.0003, 0.005, 500),
                'USD_BRL': np.random.normal(0.0005, 0.008, 500),
                'PETROLEO_BRENT': np.random.normal(0.0008, 0.012, 500),
                'IBOVESPA': np.random.normal(0.001, 0.015, 500)
            }
            returns = pd.DataFrame(example_data, index=dates)
            return returns

# Função de exemplo
def demo_advanced_simulators():
    """Demonstração dos simuladores avançados"""
    print("🚀 DEMONSTRAÇÃO - SIMULADORES AVANÇADOS DE RISCO")
    print("=" * 60)
    
    # Carregar dados
    returns = load_data_for_simulation()
    if returns is None:
        print("❌ Dados não carregados")
        return
    
    print(f"📊 Dados: {returns.shape}")
    print(f"📈 Ativos: {list(returns.columns)}")
    
    # Inicializar simulador
    advanced_sim = AdvancedRiskSimulators(returns)
    
    # Comparar todos os métodos
    comparison = advanced_sim.compare_all_methods()
    
    # Salvar resultados
    os.makedirs('reports', exist_ok=True)
    comparison.to_csv('reports/comparacao_simuladores.csv')
    comparison.to_html('reports/comparacao_simuladores.html')
    
    print(f"\n💾 Resultados salvos em:")
    print("   - reports/comparacao_simuladores.csv")
    print("   - reports/comparacao_simuladores.html")
    
    return comparison

if __name__ == "__main__":
    demo_advanced_simulators()
