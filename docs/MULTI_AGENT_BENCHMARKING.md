# AtlasFX - Multi-Agent Benchmarking Strategy

**Versión:** 1.0  
**Fecha:** 20 de Octubre, 2025  
**Propósito:** Definir estrategia para comparar SAC con otros algoritmos de RL (TD3, PPO, etc.)

---

## Executive Summary

Este documento define la estrategia para implementar un **framework de benchmarking** que compare el rendimiento de **SAC (Soft Actor-Critic)** con otros algoritmos de RL en condiciones controladas. El objetivo es validar que SAC es la mejor elección para trading algorítmico y obtener insights sobre cuándo otros agentes podrían funcionar mejor.

**Agentes a Comparar:**
1. **SAC (Soft Actor-Critic)** - Baseline principal
2. **TD3 (Twin Delayed DDPG)** - Política determinista, más estable
3. **PPO (Proximal Policy Optimization)** - On-policy, robusto
4. **A2C (Advantage Actor-Critic)** - Baseline simple on-policy
5. **DQN (Deep Q-Network)** - Baseline discreto (opcional)

---

## Motivación

### ¿Por Qué Benchmarking?

1. **Validación Científica:**
   - Probar que SAC es objetivamente mejor (no solo intuición)
   - Publicar resultados → Credibilidad en comunidad ML/finance

2. **Detectar Regímenes de Mercado:**
   - ¿Hay momentos donde TD3 funciona mejor? (ej. trending markets)
   - ¿PPO es mejor en markets volátiles?

3. **Insights para Mejora:**
   - Entender trade-offs (sample efficiency vs. stability)
   - Informar decisiones de arquitectura

4. **Aprendizaje:**
   - Experimentar con múltiples algoritmos
   - Comparar on-policy vs. off-policy
   - Comparar determinístico vs. estocástico

---

## Agentes a Benchmarking

### 1. SAC (Soft Actor-Critic) - Baseline

**Características:**
- **Tipo:** Off-policy, stochastic policy
- **Policy:** Gaussian (reparameterization trick)
- **Critics:** Twin Q-networks (reduce overestimation)
- **Exploration:** Entropy regularization (automatic temperature tuning)

**Ventajas:**
- ✅ Exploración estocástica (mejor para non-stationary markets)
- ✅ Off-policy (sample efficient)
- ✅ Entropy regularization (previene convergencia prematura)
- ✅ Twin critics (reduce bias)

**Desventajas:**
- ⚠️ Más hiperparámetros que tunar
- ⚠️ Entropía puede ser demasiado alta (sub-optimal exploitation)

**Cuándo es Mejor:**
- Markets con alta incertidumbre
- Cuando exploración es crítica
- Cuando hay múltiples estrategias óptimas (multi-modal rewards)

---

### 2. TD3 (Twin Delayed DDPG)

**Características:**
- **Tipo:** Off-policy, deterministic policy
- **Policy:** Determinística con noise añadido manualmente
- **Critics:** Twin Q-networks
- **Updates:** Delayed policy updates

**Ventajas:**
- ✅ Muy estable (menos variance que SAC)
- ✅ Off-policy (sample efficient)
- ✅ Menos hiperparámetros sensibles
- ✅ Convergencia más suave

**Desventajas:**
- ❌ Exploración limitada (deterministic policy)
- ❌ Puede quedarse en local optima
- ❌ Requiere noise manual (no adaptive como SAC)

**Cuándo es Mejor:**
- Markets estables (trending, low volatility)
- Cuando sample efficiency es crítica
- Cuando quieres convergencia rápida

**Paper:** [TD3: Addressing Function Approximation Error](https://arxiv.org/abs/1802.09477) (Fujimoto et al., 2018)

---

### 3. PPO (Proximal Policy Optimization)

**Características:**
- **Tipo:** On-policy, stochastic policy
- **Policy:** Gaussian
- **Constraint:** Clipped surrogate objective
- **Updates:** Multiple epochs on same batch

**Ventajas:**
- ✅ Muy robusto (hard to break)
- ✅ Pocos hiperparámetros
- ✅ Stable training (constraint on policy updates)
- ✅ Buen balance exploration/exploitation

**Desventajas:**
- ❌ On-policy (less sample efficient)
- ❌ Requiere más data que SAC/TD3
- ❌ Más lento para entrenar

**Cuándo es Mejor:**
- Cuando sample efficiency no es limitante
- Cuando quieres training ultra-stable
- Cuando tienes mucho compute

**Paper:** [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)

---

### 4. A2C (Advantage Actor-Critic)

**Características:**
- **Tipo:** On-policy, stochastic policy
- **Policy:** Gaussian
- **Critics:** Single value network
- **Updates:** Synchronous

**Ventajas:**
- ✅ Simple (baseline sólido)
- ✅ Fácil de implementar
- ✅ Buen para entender trade-offs

**Desventajas:**
- ❌ Menos stable que PPO
- ❌ Menos sample efficient que off-policy
- ❌ Requiere muchos workers paralelos

**Cuándo es Mejor:**
- Como baseline de comparación
- Para debugging (si todos fallan, A2C también)

**Paper:** [Asynchronous Methods for Deep RL](https://arxiv.org/abs/1602.01783) (Mnih et al., 2016)

---

### 5. DQN (Deep Q-Network) - Opcional

**Características:**
- **Tipo:** Off-policy, discrete actions
- **Policy:** ε-greedy Q-function
- **Experience:** Replay buffer
- **Target:** Fixed target network

**Ventajas:**
- ✅ Simple y bien entendido
- ✅ Off-policy (sample efficient)
- ✅ Baseline clásico

**Desventajas:**
- ❌ Solo acciones discretas (requiere discretizar position sizing)
- ❌ Exploración ε-greedy es naive
- ❌ Superado por métodos modernos

**Cuándo es Mejor:**
- Baseline de comparación histórica
- Si queremos discretizar action space

**Nota:** Probablemente no usar DQN porque:
- Position sizing es naturalmente continuo
- Discretización pierde granularidad
- SAC/TD3/PPO son superiores para continuous actions

---

## Framework de Benchmarking

### Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│              Benchmarking Framework                      │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    ┌───▼────┐      ┌─────▼─────┐      ┌────▼─────┐
    │  SAC   │      │    TD3    │      │   PPO    │
    │ Agent  │      │   Agent   │      │  Agent   │
    └───┬────┘      └─────┬─────┘      └────┬─────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                ┌──────────▼──────────┐
                │  Trading Env        │
                │  (Same for all)     │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Evaluation Suite   │
                │  (Metrics, Plots)   │
                └─────────────────────┘
```

### Componentes

#### 1. Agente Wrapper Unificado

```python
# src/atlasfx/agents/base_agent.py
from abc import ABC, abstractmethod
import torch

class BaseRLAgent(ABC):
    """Base class for all RL agents."""
    
    @abstractmethod
    def select_action(self, state: torch.Tensor, evaluate: bool = False):
        """Select action given state."""
        pass
    
    @abstractmethod
    def update(self, batch):
        """Update agent with batch of transitions."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent checkpoint."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent checkpoint."""
        pass
```

#### 2. Implementaciones Específicas

```python
# src/atlasfx/agents/sac_agent.py
class SACAgent(BaseRLAgent):
    def __init__(self, config: SACConfig):
        self.actor = GaussianPolicy(...)
        self.critic1 = QNetwork(...)
        self.critic2 = QNetwork(...)
        # ... initialization
    
    def select_action(self, state, evaluate=False):
        if evaluate:
            # Deterministic (mean of Gaussian)
            action, _ = self.actor.sample(state)
        else:
            # Stochastic sampling
            action, _ = self.actor.sample(state)
        return action
    
    def update(self, batch):
        # SAC update logic
        ...

# src/atlasfx/agents/td3_agent.py
class TD3Agent(BaseRLAgent):
    def __init__(self, config: TD3Config):
        self.actor = DeterministicPolicy(...)
        self.critic1 = QNetwork(...)
        self.critic2 = QNetwork(...)
        # ... initialization
    
    def select_action(self, state, evaluate=False):
        action = self.actor(state)
        if not evaluate:
            # Add exploration noise
            noise = torch.randn_like(action) * self.noise_scale
            action = action + noise
        return torch.clamp(action, -1, 1)
    
    def update(self, batch):
        # TD3 update logic
        ...

# Similar for PPO, A2C
```

#### 3. Benchmarking Script

```python
# scripts/benchmark_agents.py
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd

from atlasfx.agents import SACAgent, TD3Agent, PPOAgent, A2CAgent
from atlasfx.environments import TradingEnv
from atlasfx.evaluation import Evaluator, MetricsCalculator
from atlasfx.utils import setup_logger, set_seed

def benchmark_agents(
    agents: Dict[str, BaseRLAgent],
    env_config: dict,
    train_episodes: int = 10000,
    eval_episodes: int = 100,
    seeds: List[int] = [42, 123, 456],
):
    """
    Benchmark multiple RL agents.
    
    Args:
        agents: Dict of agent_name -> agent_instance
        env_config: Config for trading environment
        train_episodes: Number of training episodes
        eval_episodes: Number of evaluation episodes
        seeds: Random seeds for reproducibility
    
    Returns:
        results: DataFrame with metrics per agent
    """
    results = []
    
    for agent_name, agent_class in agents.items():
        logger.info(f"Benchmarking {agent_name}...")
        
        agent_results = []
        
        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            
            # Set seed
            set_seed(seed)
            
            # Create fresh environment and agent
            env = TradingEnv(env_config)
            agent = agent_class(config)
            
            # Train agent
            logger.info(f"    Training {agent_name}...")
            train_metrics = train_agent(
                agent, env, n_episodes=train_episodes
            )
            
            # Evaluate agent
            logger.info(f"    Evaluating {agent_name}...")
            eval_env = TradingEnv(env_config, split='test')
            eval_metrics = evaluate_agent(
                agent, eval_env, n_episodes=eval_episodes
            )
            
            # Store results
            agent_results.append({
                'agent': agent_name,
                'seed': seed,
                **train_metrics,
                **eval_metrics,
            })
        
        results.extend(agent_results)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    return df_results


def train_agent(agent, env, n_episodes):
    """Train agent and return metrics."""
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, _ = env.step(action)
            
            agent.update({'state': state, 'action': action, 
                         'reward': reward, 'next_state': next_state, 
                         'done': done})
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        if episode % 100 == 0:
            logger.info(f"      Episode {episode}, Reward: {episode_reward:.2f}")
    
    return {
        'train_mean_reward': np.mean(episode_rewards[-100:]),
        'train_std_reward': np.std(episode_rewards[-100:]),
    }


def evaluate_agent(agent, env, n_episodes):
    """Evaluate agent and return comprehensive metrics."""
    episode_rewards = []
    sharpe_ratios = []
    max_drawdowns = []
    win_rates = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        trades = []
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            
            if info.get('trade_executed'):
                trades.append(info['trade_pnl'])
        
        episode_rewards.append(episode_reward)
        
        # Calculate metrics
        if len(trades) > 0:
            returns = pd.Series(trades)
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            sharpe_ratios.append(sharpe)
            
            cumulative = returns.cumsum()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdowns.append(drawdown.min())
            
            win_rate = (returns > 0).sum() / len(returns)
            win_rates.append(win_rate)
    
    return {
        'eval_mean_reward': np.mean(episode_rewards),
        'eval_std_reward': np.std(episode_rewards),
        'eval_sharpe_ratio': np.mean(sharpe_ratios),
        'eval_max_drawdown': np.mean(max_drawdowns),
        'eval_win_rate': np.mean(win_rates),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to benchmark config')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Define agents to benchmark
    agents = {
        'SAC': SACAgent,
        'TD3': TD3Agent,
        'PPO': PPOAgent,
        'A2C': A2CAgent,
    }
    
    # Run benchmark
    results = benchmark_agents(
        agents=agents,
        env_config=config.env,
        train_episodes=config.train_episodes,
        eval_episodes=config.eval_episodes,
        seeds=config.seeds,
    )
    
    # Save results
    results.to_csv('benchmark_results.csv', index=False)
    
    # Generate comparison plots
    plot_benchmark_results(results)
    
    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
```

---

## Métricas de Comparación

### Métricas Primarias

1. **Sharpe Ratio** (risk-adjusted returns)
   - Fórmula: `mean(returns) / std(returns) * sqrt(252)`
   - Target: > 1.0

2. **Maximum Drawdown** (risk metric)
   - Fórmula: `max(running_max - cumulative_returns)`
   - Target: < 20%

3. **Win Rate** (trading consistency)
   - Fórmula: `n_winning_trades / n_total_trades`
   - Target: > 50%

4. **Profit Factor**
   - Fórmula: `sum(winning_trades) / abs(sum(losing_trades))`
   - Target: > 1.5

### Métricas Secundarias

5. **Sample Efficiency**
   - Episodes to reach threshold Sharpe ratio
   - Menos es mejor

6. **Training Stability**
   - Variance de rewards across seeds
   - Menos es mejor

7. **Convergence Speed**
   - Episodes to converge (change < ε)
   - Menos es mejor

8. **Robustness**
   - Performance consistency across different market periods
   - Más es mejor

---

## Visualizaciones

### 1. Learning Curves

```python
def plot_learning_curves(results):
    """Plot reward over episodes for each agent."""
    plt.figure(figsize=(12, 6))
    
    for agent_name in results['agent'].unique():
        agent_data = results[results['agent'] == agent_name]
        
        # Plot mean and std across seeds
        mean_rewards = agent_data.groupby('episode')['reward'].mean()
        std_rewards = agent_data.groupby('episode')['reward'].std()
        
        plt.plot(mean_rewards, label=agent_name)
        plt.fill_between(
            mean_rewards.index,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.3
        )
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png', dpi=300)
```

### 2. Performance Comparison (Violin Plots)

```python
def plot_performance_comparison(results):
    """Violin plot of Sharpe ratios."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sharpe Ratio
    sns.violinplot(data=results, x='agent', y='eval_sharpe_ratio', ax=axes[0])
    axes[0].set_title('Sharpe Ratio Distribution')
    axes[0].axhline(y=1.0, color='r', linestyle='--', label='Target')
    axes[0].legend()
    
    # Max Drawdown
    sns.violinplot(data=results, x='agent', y='eval_max_drawdown', ax=axes[1])
    axes[1].set_title('Max Drawdown Distribution')
    axes[1].axhline(y=-0.20, color='r', linestyle='--', label='Threshold')
    axes[1].legend()
    
    # Win Rate
    sns.violinplot(data=results, x='agent', y='eval_win_rate', ax=axes[2])
    axes[2].set_title('Win Rate Distribution')
    axes[2].axhline(y=0.50, color='r', linestyle='--', label='Target')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
```

### 3. Statistical Comparison

```python
def statistical_comparison(results):
    """Perform statistical tests to compare agents."""
    from scipy.stats import ttest_ind
    
    sac_sharpe = results[results['agent'] == 'SAC']['eval_sharpe_ratio']
    td3_sharpe = results[results['agent'] == 'TD3']['eval_sharpe_ratio']
    ppo_sharpe = results[results['agent'] == 'PPO']['eval_sharpe_ratio']
    
    # SAC vs. TD3
    t_stat, p_value = ttest_ind(sac_sharpe, td3_sharpe)
    print(f"SAC vs. TD3: t={t_stat:.3f}, p={p_value:.3f}")
    
    # SAC vs. PPO
    t_stat, p_value = ttest_ind(sac_sharpe, ppo_sharpe)
    print(f"SAC vs. PPO: t={t_stat:.3f}, p={p_value:.3f}")
    
    # Effect size (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (x.mean() - y.mean()) / np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / dof)
    
    d_sac_td3 = cohens_d(sac_sharpe, td3_sharpe)
    print(f"Effect size (SAC vs. TD3): d={d_sac_td3:.3f}")
```

---

## Experimentos Propuestos

### Experimento 1: Baseline Comparison

**Objetivo:** Comparar rendimiento básico de todos los agentes

**Setup:**
- Mismos hiperparámetros (lr, batch_size, etc.)
- Mismo environment (7 pares forex, 5min intervals)
- Mismo training budget (10K episodes)
- 5 seeds diferentes

**Métricas:**
- Sharpe ratio (test set)
- Max drawdown
- Win rate
- Sample efficiency

**Duración:** 2-3 días (con GPU)

---

### Experimento 2: Regime Analysis

**Objetivo:** ¿Qué agente funciona mejor en qué régimen?

**Setup:**
- Dividir test set en regímenes:
  - Trending (high autocorrelation)
  - Ranging (low volatility)
  - Volatile (high volatility)
- Evaluar cada agente en cada régimen

**Hipótesis:**
- TD3 mejor en trending (deterministic policy aprovecha momentum)
- SAC mejor en volatile (exploration ayuda)
- PPO más robusto across regimes

**Duración:** 1 día

---

### Experimento 3: Hyperparameter Sensitivity

**Objetivo:** ¿Qué agente es más robusto a cambios de hiperparámetros?

**Setup:**
- Variar learning rate (1e-5 a 1e-2)
- Variar batch size (64 a 512)
- Variar network size
- Medir variance en performance

**Métricas:**
- Std dev de Sharpe across configs
- Worst-case Sharpe

**Duración:** 3-5 días (requiere grid search)

---

### Experimento 4: Sample Efficiency

**Objetivo:** ¿Qué agente aprende más rápido?

**Setup:**
- Limitar training data (1K, 2K, 5K episodes)
- Evaluar en test set después de cada budget
- Plot sample efficiency curves

**Hipótesis:**
- Off-policy (SAC, TD3) más sample efficient que on-policy (PPO)

**Duración:** 2 días

---

## Timeline de Implementación

### Fase 1: Infrastructure (1 semana)
- [ ] Implementar `BaseRLAgent` interface
- [ ] Implementar wrappers para SAC, TD3, PPO
- [ ] Crear benchmarking script
- [ ] Setup logging y tracking

### Fase 2: Baseline Experiment (1 semana)
- [ ] Correr experimento 1 (baseline comparison)
- [ ] Generar visualizaciones
- [ ] Análisis estadístico
- [ ] Documentar resultados

### Fase 3: Advanced Experiments (2-3 semanas)
- [ ] Experimento 2 (regime analysis)
- [ ] Experimento 3 (hyperparameter sensitivity)
- [ ] Experimento 4 (sample efficiency)
- [ ] Consolidar resultados

### Fase 4: Documentation (1 semana)
- [ ] Escribir report técnico
- [ ] Crear presentación
- [ ] Publicar en blog/paper

**Total:** 5-6 semanas (post-MVP)

---

## Ubicación en el Repositorio

### Opción A: Módulo Dedicado

```
src/atlasfx/
├── agents/
│   ├── __init__.py
│   ├── base.py          # BaseRLAgent interface
│   ├── sac.py           # SAC implementation
│   ├── td3.py           # TD3 implementation
│   ├── ppo.py           # PPO implementation
│   └── a2c.py           # A2C implementation
├── benchmarking/        # ✅ NUEVO
│   ├── __init__.py
│   ├── benchmark.py     # Main benchmarking script
│   ├── evaluator.py     # Evaluation utilities
│   └── visualizer.py    # Plotting functions
```

### Opción B: Scripts Separados

```
scripts/
├── benchmark/           # ✅ NUEVO
│   ├── run_benchmark.py
│   ├── analyze_results.py
│   └── plot_comparison.py
```

**Recomendación:** Opción A (módulo dedicado)
- Código reutilizable
- Mejor organización
- Fácil mantener

---

## Entregables

### 1. Código
- [ ] `BaseRLAgent` interface
- [ ] Wrappers para SAC, TD3, PPO, A2C
- [ ] Benchmarking framework
- [ ] Evaluation utilities
- [ ] Plotting functions

### 2. Experimentos
- [ ] Resultados de 4 experimentos (CSV + plots)
- [ ] Statistical analysis
- [ ] Regime analysis

### 3. Documentación
- [ ] Technical report (10-15 pages)
- [ ] API documentation
- [ ] Usage examples

### 4. Presentación (Opcional)
- [ ] Slides con resultados
- [ ] Blog post
- [ ] Paper draft

---

## Recursos Computacionales

### GPU Requirements
- **Training:** 1x RTX 3090 (24GB VRAM)
- **Parallel experiments:** 4x GPUs ideal
- **Cloud alternative:** AWS p3.2xlarge ($3/hr)

### Storage
- Raw results: ~10GB (logs, checkpoints)
- Plots: ~100MB
- Total: ~10-15GB

### Time
- Single agent, single seed: ~6 hours
- Full benchmark (4 agents, 5 seeds): ~120 hours
- Con paralelización (4 GPUs): ~30 hours

---

## Conclusión

**¿Vale la Pena?**

✅ **SÍ**, porque:
1. Valida científicamente la elección de SAC
2. Insights valiosos sobre trade-offs
3. Detecta regímenes donde otros agentes son mejores
4. Material para publicación/blog
5. Aprendizaje profundo de RL algorithms

**Cuándo Hacerlo:**
- ✅ **Post-MVP** (no durante MVP)
- Después de tener SAC funcionando sólidamente
- Cuando tienes tiempo/recursos para experimentos

**Prioridad:** ⭐⭐⭐⭐ Alta (pero post-MVP)

---

## Referencias

### Papers
- **SAC:** Haarnoja et al. (2018) - Soft Actor-Critic
- **TD3:** Fujimoto et al. (2018) - Addressing Function Approximation Error
- **PPO:** Schulman et al. (2017) - Proximal Policy Optimization
- **A2C:** Mnih et al. (2016) - Asynchronous Methods for Deep RL

### Librerías
- **Stable-Baselines3:** Implementaciones de referencia
- **RLlib:** Framework de benchmarking
- **Optuna:** Hyperparameter tuning

---

**Autor:** AtlasFX Benchmarking Team  
**Fecha:** 20 de Octubre, 2025  
**Versión:** 1.0  
**Status:** Propuesta para Post-MVP
