# AtlasFX - Backlog & Future Roadmap

**Version:** 1.0  
**Last Updated:** 20 de Octubre, 2025  
**Status:** Planning

---

## Purpose

Este documento captura ideas, mejoras y características planificadas para versiones futuras del MVP y fases posteriores del proyecto AtlasFX. Mantiene un registro organizado de todo lo que se quiere implementar pero que está fuera del alcance inmediato del MVP actual.

---

## Cómo Usar Este Backlog

### Categorías de Prioridad

- **🔥 Alta Prioridad (Post-MVP Inmediato):** Características críticas para pasar de MVP a producción
- **⭐ Media Prioridad (Versión 2.0):** Mejoras importantes pero no bloqueantes
- **💡 Baja Prioridad (Futuro):** Ideas interesantes para explorar eventualmente
- **🔬 Investigación:** Requiere más investigación antes de decidir implementación

### Estados

- **📋 Backlog:** Idea capturada, no iniciada
- **🔍 En Análisis:** Evaluando viabilidad y diseño
- **✅ Completado:** Implementado y desplegado
- **❌ Descartado:** Decidido no implementar (con razón documentada)

---

## MVP Current Scope (Baseline)

### Core Components (En Desarrollo)

- **Data Pipeline:** Procesamiento de tick data Level 1 (7 pares forex)
- **VAE:** Representación latente del estado de mercado
- **TFT:** Pronósticos multi-horizonte con incertidumbre
- **SAC:** Agente de trading con política estocástica
- **Backtesting:** Sistema de evaluación en datos históricos

### MVP Features

- 7 pares de forex (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD)
- Ventanas temporales: 1-10 minutos
- Horizones de predicción: 1, 5, 10 minutos
- Métricas: Sharpe ratio, drawdown, win rate, profit factor
- Reproducibilidad: Seeds fijos, DVC, MLflow

---

## 🔥 Post-MVP Inmediato (Prioridad Alta)

### 1. Expansión de Instrumentos

**Status:** 📋 Backlog  
**Categoría:** Datos  
**Esfuerzo Estimado:** 2-3 semanas

**Descripción:**
Agregar 3 instrumentos adicionales mencionados en el problema statement original.

**Candidatos:**
- **Commodities:** Gold (XAU/USD), Oil (WTI)
- **Índices:** S&P 500 futures (ES)
- **Crypto:** Bitcoin (BTC/USD) - si disponible en Dukascopy

**Razón:**
- Diversificación de portafolio
- Reducción de correlación entre assets
- Mejor gestión de riesgo

**Implementación:**
1. Extender pipeline de datos para soportar múltiples tipos de instrumentos
2. Ajustar features (algunos no aplicables a todos los instrumentos)
3. Actualizar VAE para manejar características específicas por instrumento
4. Entrenar modelos multi-asset

**Consideraciones:**
- ¿Usar un solo modelo o modelos especializados por tipo de asset?
- ¿Cómo normalizar diferentes escalas de precios?
- ¿Distintas estructuras de costos de transacción?

---

### 2. Slippage y Market Impact Models

**Status:** 📋 Backlog  
**Categoría:** Trading Environment  
**Esfuerzo Estimado:** 1-2 semanas

**Descripción:**
Agregar modelos realistas de slippage y market impact al entorno de trading.

**Razón:**
- MVP usa simplificaciones (ejecución instantánea a mid price)
- En producción, trades tienen slippage y mueven el mercado
- Resultados de backtest serían optimistas sin esto

**Implementación:**
```python
class RealisticExecutionModel:
    def execute_trade(self, size, current_spread, recent_volume):
        # Modelo de slippage
        base_slippage = self.spread_slippage(current_spread)
        volume_slippage = self.volume_slippage(size, recent_volume)
        
        # Market impact (proporcional a tamaño relativo)
        market_impact = self.calculate_impact(size, recent_volume)
        
        total_cost = base_slippage + volume_slippage + market_impact
        return total_cost
```

**Referencias:**
- Almgren & Chriss (2000) - Optimal Execution
- Gatheral (2010) - Market Impact Models

---

### 3. Sistema de Gestión de Riesgo en Tiempo Real

**Status:** 📋 Backlog  
**Categoría:** Risk Management  
**Esfuerzo Estimado:** 2-3 semanas

**Descripción:**
Implementar sistema de límites y circuit breakers para producción.

**Componentes:**
- **Stop-loss dinámico:** Basado en volatilidad reciente
- **Position limits:** Máximo por par, máximo total
- **Drawdown limits:** Pausa trading si drawdown > threshold
- **Exposure limits:** Balance long/short
- **Velocity checks:** Detectar comportamiento anormal del agente

**Implementación:**
```python
class RiskManager:
    def check_trade_allowed(self, action, portfolio_state):
        # Check position limits
        if self.exceeds_position_limit(action, portfolio_state):
            return False, "Position limit exceeded"
        
        # Check drawdown threshold
        if self.in_drawdown_protection(portfolio_state):
            return False, "Drawdown protection active"
        
        # Check exposure limits
        if self.exceeds_exposure_limit(action, portfolio_state):
            return False, "Exposure limit exceeded"
        
        return True, None
```

**Parámetros Sugeridos:**
- Max position per pair: 10% del capital
- Max total exposure: 30% del capital
- Drawdown threshold: -15% desde peak
- Daily loss limit: -5% del capital

---

### 4. Deployment Pipeline (Producción)

**Status:** 📋 Backlog  
**Categoría:** MLOps  
**Esfuerzo Estimado:** 3-4 semanas

**Descripción:**
Sistema completo de deployment para trading en vivo.

**Arquitectura:**
```
┌─────────────────┐
│  Data Ingestion │  ← WebSocket feed (Dukascopy/broker API)
└────────┬────────┘
         │
┌────────▼────────┐
│ Feature Engine  │  ← Real-time pipeline (mismas features que entrenamiento)
└────────┬────────┘
         │
┌────────▼────────┐
│  Model Serving  │  ← VAE → TFT → SAC (TorchScript optimizado)
└────────┬────────┘
         │
┌────────▼────────┐
│  Risk Manager   │  ← Validación de trades
└────────┬────────┘
         │
┌────────▼────────┐
│ Order Execution │  ← Broker API (OANDA, Interactive Brokers)
└────────┬────────┘
         │
┌────────▼────────┐
│   Monitoring    │  ← Prometheus + Grafana dashboards
└─────────────────┘
```

**Tecnologías:**
- **Serving:** FastAPI + TorchScript
- **Containerización:** Docker + Docker Compose
- **Orquestación:** Kubernetes (opcional, para alta disponibilidad)
- **Monitoring:** Prometheus + Grafana
- **Logging:** ELK Stack o Loki

**Métricas a Monitorear:**
- Latencia de predicción (P50, P95, P99)
- Throughput (predicciones/segundo)
- Accuracy de pronósticos vs. realidad
- PnL en tiempo real
- Drawdown actual vs. histórico
- Tasa de errores (fallos de API, timeouts)

---

## ⭐ Versión 2.0 (Media Prioridad)

### 5. Multi-Agent Benchmarking Framework

**Status:** 📋 Backlog  
**Categoría:** Experimentación  
**Esfuerzo Estimado:** 2-3 semanas

**Descripción:**
Framework para comparar SAC contra otros algoritmos de RL en mismas condiciones.

**Agentes a Benchmarking:**
1. **SAC (Baseline):** Nuestro agente principal
2. **TD3 (Twin Delayed DDPG):** Más estable, menos exploración
3. **PPO (Proximal Policy Optimization):** On-policy, más sample-efficient
4. **DQN:** Baseline discreto (si discretizamos acciones)
5. **A2C/A3C:** Actor-Critic simple

**Por Qué Cada Uno:**

**TD3:**
- ✅ Pros: Muy estable, menos hiperparámetros sensibles
- ❌ Contras: Política determinista (menos exploración)
- 🎯 Uso: Comparar si la exploración estocástica de SAC ayuda

**PPO:**
- ✅ Pros: On-policy (mejor convergencia), más robusto
- ❌ Contras: Menos sample-efficient, más lento
- 🎯 Uso: Ver si on-policy es mejor para markets (no-estacionarios)

**Razón del Framework:**
- Validar que SAC es la mejor elección
- Detectar si hay regímenes de mercado donde otros agentes funcionan mejor
- Publicar resultados en paper/blog (credibilidad científica)

**Implementación:**
```python
# experiments/benchmark_agents.py
agents = {
    'SAC': SACAgent(config),
    'TD3': TD3Agent(config),
    'PPO': PPOAgent(config),
}

results = {}
for name, agent in agents.items():
    print(f"Training {name}...")
    agent.train(env, n_episodes=10000)
    
    print(f"Evaluating {name}...")
    metrics = evaluate(agent, test_env, n_episodes=100)
    results[name] = metrics
    
# Comparar métricas
compare_agents(results)  # Genera tablas y gráficos
```

**Métricas de Comparación:**
- Sharpe Ratio (risk-adjusted returns)
- Convergence speed (episodes to threshold)
- Sample efficiency (returns per sample)
- Stability (variance across seeds)
- Robustness (performance en diferentes periodos)

**Entregables:**
1. Script de benchmark automatizado
2. Resultados en formato tabla (LaTeX para papers)
3. Gráficos de comparación (learning curves, violin plots)
4. Documento técnico con análisis

---

### 6. Hierarchical RL (Meta-Policy)

**Status:** 🔬 Investigación  
**Categoría:** Algoritmos  
**Esfuerzo Estimado:** 4-6 semanas

**Descripción:**
Implementar arquitectura jerárquica donde un meta-agente selecciona estrategias y sub-agentes las ejecutan.

**Motivación:**
- Markets tienen regímenes (trending, ranging, high volatility)
- Un solo agente puede no ser óptimo en todos los regímenes
- Meta-learner puede aprender a cambiar de estrategia

**Arquitectura:**
```
┌──────────────────────────────────────┐
│         Meta-Policy (PPO)            │
│  Input: Market regime features       │
│  Output: Strategy selection (0-N)    │
└──────────────┬───────────────────────┘
               │
       ┌───────┼───────┬───────┐
       │               │       │
┌──────▼─────┐ ┌──────▼─────┐ ┌──────▼─────┐
│ Strategy 1 │ │ Strategy 2 │ │ Strategy 3 │
│ (Trend)    │ │ (Mean Rev) │ │ (Breakout) │
│ SAC Agent  │ │ SAC Agent  │ │ SAC Agent  │
└────────────┘ └────────────┘ └────────────┘
```

**Implementación:**
1. Pre-entrenar sub-agentes en regímenes específicos
2. Entrenar meta-policy para seleccionar sub-agente
3. Opcional: Fine-tune end-to-end

**Métricas:**
- ¿Mejora sobre single-agent SAC?
- ¿Correctness de regime detection?
- ¿Switching cost (transaction costs al cambiar estrategia)?

**Referencias:**
- Bacon et al. (2017) - The Option-Critic Architecture
- Vezhnevets et al. (2017) - FeUdal Networks

---

### 7. Offline RL (Batch RL)

**Status:** 🔬 Investigación  
**Categoría:** Algoritmos  
**Esfuerzo Estimado:** 3-4 semanas

**Descripción:**
Entrenar agente puramente en datos históricos sin interacción con environment.

**Motivación:**
- Más sample-efficient (usa todo el histórico disponible)
- No requiere simulación durante entrenamiento
- Evita exploration noise en training

**Algoritmos Candidatos:**
- **BCQ (Batch-Constrained Q-learning)**
- **CQL (Conservative Q-Learning)**
- **TD3+BC** (TD3 with Behavior Cloning)

**Workflow:**
1. Generar dataset de experiencias (offline dataset)
2. Entrenar agente offline
3. Evaluar en environment
4. Comparar vs. SAC online

**Ventajas:**
- Entrena más rápido (no espera simulación)
- Usa más data histórica
- Más estable (no exploration randomness)

**Desventajas:**
- Puede sufrir de distribution shift
- Requiere dataset de buena calidad
- Menos exploración

---

### 8. Explainability & Interpretability

**Status:** 📋 Backlog  
**Categoría:** Análisis  
**Esfuerzo Estimado:** 2-3 semanas

**Descripción:**
Agregar herramientas para entender por qué el agente toma decisiones.

**Componentes:**

**1. Attention Weights Visualization (TFT)**
```python
# Visualizar qué features atiende el TFT
def plot_attention_weights(tft_model, sample_input):
    attention_weights = tft_model.get_attention_weights(sample_input)
    plt.imshow(attention_weights, cmap='viridis')
    plt.ylabel('Time Step')
    plt.xlabel('Feature')
    plt.show()
```

**2. SHAP Values (Feature Importance)**
```python
import shap

# Explicar decisión del agente
explainer = shap.DeepExplainer(sac_agent.actor, background_data)
shap_values = explainer.shap_values(current_state)

shap.waterfall_plot(shap_values)
```

**3. Policy Rollouts Visualization**
- Mostrar estados visitados en latent space (VAE)
- Colorear por reward obtenido
- Identificar regiones de alta/baja performance

**4. Trade Analysis Dashboard**
- ¿Qué features tenían valores altos cuando se hizo buen trade?
- ¿Cuándo falló el agente? (post-mortem analysis)
- ¿Correlación entre forecast uncertainty y trading decisions?

**Razón:**
- Debugging (entender errores del modelo)
- Confianza (stakeholders quieren explicaciones)
- Mejora (insights para feature engineering)

---

### 9. Walk-Forward Optimization

**Status:** 📋 Backlog  
**Categoría:** Validación  
**Esfuerzo Estimado:** 2-3 semanas

**Descripción:**
Validación robusta mediante re-entrenamiento periódico en ventanas móviles.

**Metodología:**
```
Train: [2020-01-01 to 2020-12-31]
Test:  [2021-01-01 to 2021-03-31]

Train: [2020-04-01 to 2021-03-31]
Test:  [2021-04-01 to 2021-06-30]

Train: [2020-07-01 to 2021-06-30]
Test:  [2021-07-01 to 2021-09-30]

...
```

**Ventajas:**
- Simula re-entrenamiento en producción
- Detecta degradación de performance over time
- Evalúa robustness a cambios de mercado

**Métricas:**
- Sharpe ratio promedio en todos los folds
- Peor fold (worst-case scenario)
- Stability (std dev de Sharpe across folds)

**Implementación:**
```python
def walk_forward_validation(agent_class, data, train_window, test_window):
    results = []
    
    for start in range(0, len(data) - train_window - test_window, test_window):
        train_data = data[start:start+train_window]
        test_data = data[start+train_window:start+train_window+test_window]
        
        # Train agent
        agent = agent_class()
        agent.train(train_data)
        
        # Evaluate
        metrics = agent.evaluate(test_data)
        results.append(metrics)
    
    return pd.DataFrame(results)
```

---

### 10. Portfolio Optimization (Multi-Asset Trading)

**Status:** 📋 Backlog  
**Categoría:** Trading Strategy  
**Esfuerzo Estimado:** 3-4 semanas

**Descripción:**
Optimizar portafolio multi-asset en vez de tradear pares independientemente.

**Actualmente:**
- SAC decide posición para cada par de forma independiente
- No considera correlaciones entre pares

**Mejora:**
- Usar Markowitz Mean-Variance Optimization
- O agregar peso de diversificación en reward function
- O entrenar SAC con action space = portfolio weights

**Reward Function Actualizada:**
```python
def portfolio_reward(actions, returns, cov_matrix):
    # Expected return
    portfolio_return = actions @ returns
    
    # Portfolio variance (penalizar alta correlación)
    portfolio_variance = actions @ cov_matrix @ actions
    
    # Sharpe ratio (return per unit of risk)
    sharpe = portfolio_return / np.sqrt(portfolio_variance)
    
    return sharpe
```

**Ventajas:**
- Mejor diversificación
- Menor riesgo total
- Mayor Sharpe ratio

---

## 💡 Futuro (Baja Prioridad)

### 11. Alternative Data Integration

**Status:** 💡 Idea  
**Categoría:** Datos

**Fuentes de Datos Alternativas:**
- **Sentiment Analysis:** Twitter, Reddit, news headlines
- **Order Book Data:** Level 2+ (profundidad de mercado)
- **Economic Calendar:** Eventos macro (NFP, Fed meetings, etc.)
- **Options Flow:** Indicador de sentimiento institucional

**Razón:**
- Potencial edge sobre competencia
- Mejor captura de régimen de mercado

**Desafíos:**
- Requiere licencias caras (Bloomberg, Refinitiv)
- Noisy data (especialmente sentiment)
- Más complejidad en pipeline

---

### 12. Adversarial Training

**Status:** 🔬 Investigación  
**Categoría:** Robustness

**Descripción:**
Entrenar agente contra adversary que intenta explotar sus debilidades.

**Motivación:**
- Markets son adversariales (otros traders, HFTs)
- Agente debe ser robusto a manipulación

**Implementación:**
```python
# Adversarial environment
class AdversarialMarket(TradingEnv):
    def __init__(self, agent, adversary):
        self.agent = agent
        self.adversary = adversary
    
    def step(self, action):
        # Adversary intenta predecir acción del agente
        adversary_action = self.adversary.predict(self.state)
        
        # Modifica mercado para castigar al agente
        market_impact = self.apply_adversarial_impact(adversary_action)
        
        return self._execute_with_impact(action, market_impact)
```

**Referencias:**
- Pinto et al. (2017) - Robust Adversarial Reinforcement Learning
- Gleave et al. (2020) - Adversarial Policies

---

### 13. Meta-Learning (Learning to Learn)

**Status:** 🔬 Investigación  
**Categoría:** Algoritmos

**Descripción:**
Entrenar agente que se adapta rápidamente a nuevos regímenes de mercado.

**Algoritmos:**
- **MAML (Model-Agnostic Meta-Learning)**
- **Reptile**

**Motivación:**
- Markets cambian constantemente
- Agente que aprende rápido tiene ventaja

**Workflow:**
1. Meta-train en múltiples periodos históricos (tasks)
2. Fine-tune en nuevo periodo con pocos samples
3. Evaluar adaptación speed

---

### 14. Ensemble Methods

**Status:** 💡 Idea  
**Categoría:** Modelos

**Descripción:**
Combinar múltiples modelos SAC para mejorar robustness.

**Tipos de Ensemble:**

**1. Model Averaging:**
```python
actions = [agent.predict(state) for agent in ensemble]
final_action = np.mean(actions, axis=0)
```

**2. Voting:**
- Cada agente vota por acción
- Acción con más votos se ejecuta

**3. Stacking:**
- Meta-learner combina predicciones de múltiples agentes

**Ventajas:**
- Reduce variance
- Más robusto a overfitting

**Desventajas:**
- Más lento (N veces más inferencias)
- Más complejo de mantener

---

## ❌ Ideas Descartadas

### ~~Reinforcement Learning from Human Feedback (RLHF)~~

**Razón para Descartar:**
- No tenemos expertos disponibles para labeling
- Too expensive and time-consuming
- MVP debe ser fully automated

---

### ~~Multi-Task Learning (MTL)~~

**Razón para Descartar:**
- No tenemos tareas relacionadas obvias
- Forex trading es suficientemente complejo por sí solo
- MTL agrega complejidad sin clear benefit

---

## 🔄 Proceso de Gestión del Backlog

### Añadir Items

1. Escribir descripción clara
2. Asignar categoría y prioridad
3. Estimar esfuerzo
4. Agregar razón/motivación
5. Opcional: Referencias o diseño preliminar

### Priorización

Cada trimestre, revisar y re-priorizar basado en:
- Feedback de MVP en producción
- Nuevos papers/técnicas
- Recursos disponibles
- Impacto estimado en performance

### Completar Items

Al implementar:
1. Crear issue en GitHub
2. Implementar en feature branch
3. PR con tests y documentación
4. Actualizar backlog: ✅ Completado

---

## Próximos Pasos

### Inmediato (Post-MVP)
1. ✅ Completar MVP baseline (en progreso)
2. 🔄 Evaluar performance del MVP
3. 📋 Decidir cuáles items del backlog son prioritarios
4. 🚀 Comenzar implementación de Post-MVP features

### Revisión del Backlog
- **Frecuencia:** Mensual
- **Owner:** Tech Lead / Product Owner
- **Criterios:** Performance metrics, ROI, effort vs. impact

---

## Contribuir al Backlog

Si tienes ideas para agregar:
1. Crear issue en GitHub con label `backlog-idea`
2. Usar template:
   ```
   **Título:** [Nombre descriptivo]
   **Categoría:** [Datos/Algoritmos/MLOps/etc.]
   **Prioridad Sugerida:** [Alta/Media/Baja]
   **Descripción:** [¿Qué quieres implementar?]
   **Razón:** [¿Por qué es valioso?]
   **Esfuerzo Estimado:** [Semanas]
   ```

---

**Autor:** AtlasFX Team  
**Última Actualización:** Octubre 2025  
**Versión:** 1.0
