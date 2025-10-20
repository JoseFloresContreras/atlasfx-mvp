# AtlasFX Architecture Document

**Version:** 1.0 - MVP Phase  
**Last Updated:** October 17, 2025  
**Status:** Draft - Awaiting Implementation

---

## Overview

AtlasFX is a professional-grade algorithmic trading system for short-term (1-10 minute) forex prediction using deep learning. The architecture integrates three advanced components:

1. **VAE (Variational Autoencoder)** - State representation learning
2. **TFT (Temporal Fusion Transformer)** - Multi-horizon forecasting
3. **SAC (Soft Actor-Critic)** - Stochastic policy learning with entropy regularization

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                             │
│  Raw Tick Data → Clean → Aggregate → Featurize → Normalize     │
└────────────────────────┬────────────────────────────────────────┘
                         │ Feature Matrix (N × D)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VAE (State Encoder)                           │
│  Encoder: X → μ(z), σ(z)                                        │
│  Sampler: z ~ N(μ, σ²)                                          │
│  Decoder: z → X̂ (reconstruction)                                │
└────────────────────────┬────────────────────────────────────────┘
                         │ Latent State z (N × d_latent)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│            TFT (Temporal Fusion Transformer)                     │
│  Input: [z_t-k:t, temporal_covariates]                          │
│  Output: Forecasts ŷ_t+1, ŷ_t+5, ŷ_t+10 + Uncertainties        │
└────────────────────────┬────────────────────────────────────────┘
                         │ Forecasts + Latent State
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              SAC (Soft Actor-Critic Agent)                       │
│  Actor: π(a|s) = Stochastic policy with entropy                 │
│  Critics: Q1(s,a), Q2(s,a) = Twin Q-functions                   │
│  Output: Trading actions (position, size, direction)            │
└────────────────────────┬────────────────────────────────────────┘
                         │ Actions
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                Trading Environment                               │
│  Execute trades → Calculate rewards → Update state              │
└────────────────────────┬────────────────────────────────────────┘
                         │ Experience (s, a, r, s', done)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Replay Buffer                                 │
│  Store transitions → Sample batches → Update networks           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Data Pipeline

**Purpose:** Transform raw Level 1 tick data into clean, normalized features.

**Input:**
- Raw tick data from Dukascopy (CSV/Parquet)
- Timestamp, bid, ask, volume for 7 major pairs + 3 instruments

**Output:**
- Aggregated time-series features at 1-5 minute intervals
- Feature matrix shape: `(N_timesteps, N_features)`
- Train/validation/test splits (70/15/15%)

**Key Operations:**
1. **Merge:** Combine multiple CSV files per symbol
2. **Clean:** Detect and handle gaps, outliers, and errors
3. **Aggregate:** Compute OHLC, volume, VWAP, OFI, micro price
4. **Featurize:** Calculate technical indicators and microstructure features
5. **Normalize:** Z-score normalization with outlier clipping
6. **Split:** Temporal train/val/test split

**Critical Requirements:**
- ✅ No lookahead bias in features
- ✅ Reproducible with fixed random seeds
- ✅ Type-safe with Pydantic schemas
- ✅ Comprehensive unit tests

**Tech Stack:**
- pandas, numpy for data manipulation
- PyArrow for Parquet I/O
- YAML for configuration

---

### 2. VAE (Variational Autoencoder)

**Purpose:** Learn a compact, continuous latent representation of market state.

**Architecture:**

```python
class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 200,      # Feature dimension
        latent_dim: int = 64,      # Latent space dimension
        hidden_dims: List[int] = [128, 64],
        beta: float = 1.0,         # KL weight (β-VAE)
    ):
        # Encoder: X → μ(z), log_σ²(z)
        self.encoder = MLP(input_dim, hidden_dims, 2 * latent_dim)
        
        # Decoder: z → X̂
        self.decoder = MLP(latent_dim, hidden_dims[::-1], input_dim)
```

**Training Objective:**

```
L_VAE = E_q[log p(X|z)] - β * KL(q(z|X) || p(z))
      = Reconstruction Loss + β * KL Divergence
```

**Hyperparameters:**
- `input_dim`: Number of input features from pipeline
- `latent_dim`: 32-128 (tunable via validation performance)
- `beta`: 0.5-2.0 (controls regularization strength)
- `hidden_dims`: [128, 64] or deeper for complex patterns

**Training Strategy:**
1. Pre-train VAE on feature reconstruction
2. Freeze encoder during RL training
3. Optional: Fine-tune end-to-end with RL loss

**Validation Metrics:**
- Reconstruction error (MSE)
- KL divergence
- Latent space visualization (PCA, t-SNE)
- Disentanglement metrics (MIG, SAP)

**Why VAE?**
- Provides uncertainty estimates (useful for risk management)
- Regularized latent space prevents overfitting
- Reduces dimensionality for faster RL training
- Captures stochastic nature of markets

---

### 3. TFT (Temporal Fusion Transformer)

**Purpose:** Multi-horizon probabilistic forecasting of price movements.

**Architecture:**

```python
class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,           # Latent state dimension
        static_dim: int = 10,          # Static features (pair ID, etc.)
        temporal_dim: int = 20,        # Time-varying covariates
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        forecast_horizons: List[int] = [1, 5, 10],  # Minutes
        quantiles: List[float] = [0.1, 0.5, 0.9],   # Prediction intervals
    ):
        # Variable Selection Networks
        self.static_vsn = VariableSelectionNetwork(static_dim, hidden_dim)
        self.temporal_vsn = VariableSelectionNetwork(temporal_dim, hidden_dim)
        
        # LSTM Encoder
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # Quantile Regression Heads
        self.forecast_heads = nn.ModuleDict({
            f"h{h}": QuantileHead(hidden_dim, len(quantiles))
            for h in forecast_horizons
        })
```

**Input:**
- Latent states from VAE: `z_t-k:t` (history window)
- Static features: pair ID, session flags
- Temporal covariates: time of day, day of week, volatility regime

**Output:**
- Multi-step forecasts: `ŷ_t+1, ŷ_t+5, ŷ_t+10`
- Quantile predictions for uncertainty: Q(0.1), Q(0.5), Q(0.9)
- Attention weights (interpretability)

**Training Objective:**

```
L_TFT = Σ_h Σ_q QuantileLoss(ŷ_t+h^q, y_t+h, q)
```

**Why TFT?**
- Handles multi-horizon forecasts natively
- Attention mechanism for interpretability
- Variable selection (feature importance)
- Quantile regression for uncertainty
- State-of-the-art for time-series forecasting

**Integration with SAC:**
- Forecasts become part of SAC's state space
- Uncertainty guides exploration/exploitation
- Horizon-specific actions (short vs. long positions)

---

### 4. SAC (Soft Actor-Critic)

**Purpose:** Learn a stochastic trading policy that maximizes risk-adjusted returns while maintaining high entropy (exploration).

**Architecture:**

```python
class SAC:
    def __init__(
        self,
        state_dim: int,               # VAE latent + TFT forecasts + internal state
        action_dim: int,              # Position sizing + direction
        hidden_dim: int = 256,
        alpha: float = 0.2,           # Entropy coefficient
        gamma: float = 0.99,          # Discount factor
        tau: float = 0.005,           # Soft update rate
    ):
        # Stochastic Actor (Gaussian policy)
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim)
        
        # Twin Q-Critics
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim)
        
        # Target Critics
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Automatic Entropy Tuning
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha)))
```

**Policy (Actor):**

```python
class GaussianPolicy(nn.Module):
    def forward(self, state):
        x = self.mlp(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x)  # Squash to [-1, 1]
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob
```

**Training Objective:**

```
# Critic Loss
L_Q = E[(Q(s,a) - (r + γ * (min Q_target(s',a') - α * log π(a'|s'))))²]

# Actor Loss  
L_π = E[α * log π(a|s) - Q(s,a)]

# Entropy Loss (auto-tune α)
L_α = -α * E[log π(a|s) + H_target]
```

**Action Space:**
- **Continuous:** Position size in [-1, 1]
  - Positive: Long position (buy)
  - Negative: Short position (sell)
  - Magnitude: Size as fraction of max allowed
- **Multi-dimensional:** One action per currency pair (7 pairs)

**State Space:**
- VAE latent state: `z_t` (64 dims)
- TFT forecasts: `ŷ_t+1, ŷ_t+5, ŷ_t+10` + uncertainties (9 dims)
- Internal state: Current positions, balance, unrealized PnL (8 dims)
- Total: ~80-100 dimensions

**Reward Function:**

```python
def calculate_reward(self, portfolio_value, prev_value, actions, positions):
    # PnL-based reward
    pnl = portfolio_value - prev_value
    pnl_reward = pnl / prev_value  # Normalized by balance
    
    # Risk penalties
    drawdown_penalty = -max(0, self.peak_value - portfolio_value) / self.peak_value
    volatility_penalty = -self.portfolio_volatility * 0.1
    
    # Action penalties (transaction costs, large positions)
    transaction_cost = -abs(actions).sum() * self.fee_rate
    position_penalty = -abs(positions).pow(2).sum() * 0.01  # L2 penalty
    
    # Total reward
    reward = (
        pnl_reward 
        + drawdown_penalty * 0.5
        + volatility_penalty * 0.2
        + transaction_cost
        + position_penalty * 0.1
    )
    
    return reward
```

**Why SAC?**
- Stochastic policy → Better exploration than deterministic TD3
- Entropy regularization → Prevents premature convergence
- Off-policy → Sample efficient (important for limited data)
- Twin Q-critics → Reduces overestimation bias
- Automatic temperature tuning → Adapts exploration dynamically

**Hyperparameters:**
- `alpha`: 0.1-0.3 (entropy coefficient, auto-tuned)
- `gamma`: 0.95-0.99 (discount factor, lower for short horizons)
- `tau`: 0.001-0.01 (soft update rate)
- `batch_size`: 256-512
- `replay_buffer_size`: 1M-10M transitions
- `learning_rate`: 3e-4 (actor), 3e-4 (critic)

---

## Trading Environment

**Purpose:** Simulate realistic forex trading with CFDs (Contracts for Difference).

**State Representation:**

```python
@dataclass
class EnvironmentState:
    # Market features (from VAE)
    latent_state: np.ndarray  # Shape: (latent_dim,)
    
    # Forecasts (from TFT)
    forecasts: Dict[str, np.ndarray]  # {horizon: predictions + uncertainties}
    
    # Portfolio state
    balance: float
    positions: Dict[str, Position]  # {pair: Position(size, entry_price, is_long)}
    unrealized_pnl: float
    realized_pnl: float
    
    # Risk metrics
    portfolio_value: float
    equity_curve: List[float]
    peak_value: float
    drawdown: float
```

**Action Execution:**

```python
class TradingEnvironment(gym.Env):
    def step(self, action: np.ndarray) -> Tuple[State, Reward, Done, Info]:
        # 1. Decode action (position sizing per pair)
        target_positions = self._decode_action(action)
        
        # 2. Execute trades (handle CFD logic)
        for pair, target_size in target_positions.items():
            current_size = self.positions.get(pair, 0)
            trade_size = target_size - current_size
            
            if abs(trade_size) > self.min_trade_size:
                self._execute_trade(pair, trade_size)
        
        # 3. Update portfolio value (mark-to-market)
        self._update_portfolio_value()
        
        # 4. Calculate reward
        reward = self._calculate_reward()
        
        # 5. Move to next timestep
        self.current_step += 1
        next_state = self._get_observation()
        done = self.current_step >= self.episode_length
        
        return next_state, reward, done, {}
```

**Realism Considerations:**
- ✅ Transaction costs (spread + commission)
- ✅ Position limits (max leverage)
- ✅ Margin requirements
- ⚠️ Slippage model (to be added)
- ⚠️ Market impact (negligible for retail sizes)
- ⚠️ Overnight financing costs (swap rates)

---

## Training Strategy

### Phase 1: Pre-training

1. **VAE Pre-training**
   - Train on all historical data
   - Minimize reconstruction loss + KL divergence
   - Validate latent space quality
   - Save encoder checkpoint

2. **TFT Pre-training**
   - Train on VAE latent states
   - Supervised learning on actual price movements
   - Validate forecast accuracy (RMSE, MAE, quantile loss)
   - Save TFT checkpoint

### Phase 2: RL Training

3. **SAC Training**
   - Load pre-trained VAE and TFT
   - Initialize replay buffer with random exploration
   - Train SAC with experience replay
   - Periodically validate on held-out data
   - Save checkpoints based on Sharpe ratio

### Phase 3: Fine-tuning (Optional)

4. **End-to-End Fine-tuning**
   - Unfreeze VAE and TFT
   - Train all three components jointly
   - Use RL loss + auxiliary losses (reconstruction, forecast)
   - Careful learning rate scheduling

**Training Loop:**

```python
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Get action from policy
        action = sac.select_action(state, evaluate=False)
        
        # Execute in environment
        next_state, reward, done, _ = env.step(action)
        
        # Store transition
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Update SAC (if enough samples)
        if len(replay_buffer) > batch_size:
            sac.update(replay_buffer, batch_size)
        
        state = next_state
    
    # Evaluate periodically
    if episode % eval_freq == 0:
        metrics = evaluate_policy(sac, eval_env, n_episodes=10)
        log_metrics(metrics)
        save_checkpoint(sac, vae, tft, metrics)
```

---

## Experiment Tracking

**Tools:** MLflow or Weights & Biases

**Tracked Metrics:**

**Training:**
- Episode rewards (mean, std, min, max)
- Actor loss, critic loss, entropy
- Portfolio value, balance
- Win rate, profit factor
- Learning rate, gradient norms

**Validation:**
- Sharpe ratio, Sortino ratio
- Maximum drawdown, Calmar ratio
- Win rate, average profit/loss
- Trade frequency, position holding time
- Forecast accuracy (TFT)

**Hyperparameters:**
- All model architecture params
- Learning rates, batch sizes
- Reward function weights
- Environment settings

---

## Deployment Architecture (Future)

```
┌─────────────────┐
│  Data Ingestion │  ← Real-time tick feed (WebSocket)
└────────┬────────┘
         │
┌────────▼────────┐
│  Feature Eng.   │  ← Same pipeline as training
└────────┬────────┘
         │
┌────────▼────────┐
│  Model Serving  │  ← VAE → TFT → SAC (ONNX/TorchScript)
└────────┬────────┘
         │
┌────────▼────────┐
│  Risk Manager   │  ← Position limits, stop-loss, circuit breakers
└────────┬────────┘
         │
┌────────▼────────┐
│  Order Exec.    │  ← Broker API (OANDA, Interactive Brokers)
└─────────────────┘
```

---

## Technology Stack

**Core:**
- Python 3.10+
- PyTorch 2.0+ (deep learning)
- Gymnasium (RL environment)
- Stable-Baselines3 (reference implementations)

**Data:**
- pandas, numpy (data manipulation)
- PyArrow (Parquet I/O)
- DVC (data versioning)

**ML Ops:**
- MLflow or Weights & Biases (experiment tracking)
- Optuna (hyperparameter tuning)
- Ray Tune (distributed training)

**Quality:**
- pytest (testing)
- mypy (type checking)
- ruff (linting/formatting)
- pre-commit (hooks)

**Deployment (Future):**
- FastAPI (model serving)
- Docker (containerization)
- Kubernetes (orchestration)

---

## Success Metrics

**MVP Success Criteria:**

1. **Training Stability**
   - [ ] VAE converges with reconstruction error < 0.1
   - [ ] TFT achieves RMSE < baseline on validation set
   - [ ] SAC achieves positive Sharpe ratio on validation set

2. **Backtesting Performance**
   - [ ] Sharpe Ratio > 1.0 on test set (risk-adjusted)
   - [ ] Maximum Drawdown < 20%
   - [ ] Win Rate > 50%
   - [ ] Positive expected value per trade

3. **Code Quality**
   - [ ] Test coverage ≥ 80%
   - [ ] No mypy errors
   - [ ] All features validated for lookahead bias
   - [ ] Complete documentation

4. **Reproducibility**
   - [ ] Fixed random seeds across runs
   - [ ] Data versioned with DVC
   - [ ] Experiments tracked in MLflow
   - [ ] Results reproducible by third party

---

## Known Limitations & Future Work

**Current Limitations:**
1. No slippage modeling
2. No market impact modeling
3. Simplified transaction costs
4. Single-asset focus (no portfolio optimization)
5. No regime detection

**Future Enhancements:**
1. Multi-asset portfolio optimization
2. Hierarchical RL (meta-policy selection)
3. Adversarial training (robustness)
4. Offline RL (better sample efficiency)
5. Explainable AI (attention visualization, SHAP)

---

## References

**VAE:**
- Kingma & Welling (2014) - Auto-Encoding Variational Bayes
- Higgins et al. (2017) - β-VAE

**TFT:**
- Lim et al. (2021) - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting

**SAC:**
- Haarnoja et al. (2018) - Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL

**Trading:**
- Cartea et al. (2015) - Algorithmic and High-Frequency Trading
- Lopez de Prado (2018) - Advances in Financial Machine Learning

---

**Document Status:** Draft - Awaiting implementation and validation  
**Next Review:** After MVP Phase 1 completion
