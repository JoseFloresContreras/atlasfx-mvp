import numpy as np
import pandas as pd
import gym
from typing import Dict, List, Tuple, Optional, Any
import random


class ForexTradingEnv(gym.Env):
    """
    Gym environment for forex trading with continuous data sampling.
    
    The environment loads forex data and allows an agent to trade on continuous
    samples of the data. Each episode samples n continuous timesteps from the data.
    """
    
    def __init__(self, 
                 data_path: str = "../../data-pipeline/data/1H_forex_data_train.parquet",
                 episode_length: int = 100,
                 initial_balance: float = 10000.0,
                 transaction_fee: float = 0.0001,
                 max_position_size: float = 1.0,
                 add_noise: bool = True
                 ):
        """
        Initialize the forex trading environment.
        
        Args:
            data_path: Path to the parquet file containing forex data
            episode_length: Number of continuous timesteps per episode
            initial_balance: Starting balance for the agent
            transaction_fee: Fee per transaction (as a fraction)
            max_position_size: Maximum position size as a fraction of balance
        """
        super().__init__()
        
        # Load and prepare data
        self.data_path = data_path
        self.episode_length = episode_length
        self._max_episode_steps = 2 * episode_length 
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position_size = max_position_size
        self.add_noise = add_noise
        
        # Load data
        self._load_data()
        
        # Environment state
        self.current_step = 0
        # Internal state: array of size num_pairs + 1 (positions + balance)
        self.internal_state = np.zeros(len(self.pairs) + 1, dtype=np.float32)
        self.internal_state[-1] = initial_balance  # Last element is balance
        self.previous_portfolio_value = initial_balance
        # Track CFD positions: {pair: {'size': float, 'entry_price': float, 'is_long': bool}}
        self.cfd_positions = {}
        self.episode_start_idx = 0
        self.current_data_slice = None
        
        # Memory state: n-dimensional memory that persists across steps
        self.memory_dim = 0
        self.memory = np.zeros(self.memory_dim, dtype=np.float32)
        
        # Define action and observation spaces
        self._setup_spaces()
        
    def _load_data(self):
        """Load and prepare the forex data."""
        print(f"🔄 Loading data from {self.data_path}...")
        self.data = pd.read_parquet(self.data_path)
        
        # Extract feature columns (environment states)
        feature_columns = [col for col in self.data.columns if col.startswith('[Feature]')]
        self.feature_data = self.data[feature_columns].values.astype(np.float32)
        
        # Extract pair columns (trading pairs)
        pair_columns = [col for col in self.data.columns if '-pair ' in col and not col.startswith('[Feature]')]
        self.pairs = list(set([col.split('-')[0] for col in pair_columns]))
        pair_close_cols = [f"{x}-pair | close" for x in self.pairs]
        self.pair_close = self.data[pair_close_cols].values.astype(np.float32)
        
        print(f"✅ Loaded {len(self.data):,} timesteps")
        print(f"📊 Feature columns: {len(feature_columns)} (shape: {self.feature_data.shape})")
        print(f"💱 Trading pairs: {self.pairs} (shape: {self.pair_close.shape})")
        print(f"📈 Data time range: {self.data['start_time'].min()} to {self.data['start_time'].max()}")

    
    def _setup_spaces(self):
        """Setup the action and observation spaces."""
        # Observation space: features + position proportions (normalized internal state) + profits + memory
        feature_dim = self.feature_data.shape[1]
        state_dim = len(self.pairs) + 1  # positions + balance
        profits_dim = len(self.pairs)  # profit for each pair
        total_obs_dim = feature_dim + state_dim + profits_dim + self.memory_dim
        
        self.observation_space = gym.spaces.Box(
            low=-3, 
            high=3, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )
        
        # Action space: CFD trading actions + memory
        # Actions are in range [-1, 1] with num_pairs + 2 + memory_dim dimensions:
        # - First num_pairs actions: symbol selection (max determines which pair)
        # - Next action: position size (negative = no trade, positive = trade size)
        # - Next action: direction (positive = long, negative = short)
        # - Last memory_dim actions: memory values for next step
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.pairs) + 2 + self.memory_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial observation
        """
        super().reset(seed=seed)
        
        # Reset environment state
        self.internal_state = np.zeros(len(self.pairs) + 1, dtype=np.float32)
        self.internal_state[-1] = self.initial_balance  # Last element is balance
        self.cfd_positions = {}  # Reset CFD positions
        self.current_step = 0
        self.previous_portfolio_value = self.initial_balance
        
        # Reset memory to zeros
        self.memory = np.zeros(self.memory_dim, dtype=np.float32)
        
        # Sample a random starting point for this episode
        max_start_idx = len(self.data) - self.episode_length
        if max_start_idx <= 0:
            raise ValueError(f"Episode length {self.episode_length} is larger than available data {len(self.data)}")
        
        self.episode_start_idx = random.randint(0, max_start_idx)
        
        # Get the data slice for this episode
        end_idx = self.episode_start_idx + self.episode_length
        self.current_data_slice = self.feature_data[self.episode_start_idx:end_idx]
            
        if self.add_noise:
            # Add Gaussian noise to the data slice
            noise = np.random.normal(0, 0.2, self.current_data_slice.shape)
            self.current_data_slice = self.current_data_slice + noise
        
        # Return initial observation with normalized internal state, profits, and memory
        normalized_state = self.get_normalized_internal_state()
        profits = self.getProfits(scale=True)
        initial_obs = np.concatenate([self.current_data_slice[0], normalized_state, profits, self.memory])
        return initial_obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Array of actions for each pair in range [-1, 1]
            
        Returns:
            observation: Current state features
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if self.current_step >= self.episode_length - 1:
            raise ValueError("Episode has already ended. Call reset() to start a new episode.")
        
        # Extract memory from action
        new_memory = action[len(action) - self.memory_dim:]
        
        # Execute trades based on actions (excluding memory dimensions)
        trading_action = action[:len(action) - self.memory_dim]
        self._execute_trades(trading_action)

        # Update memory for next step
        self.memory = new_memory

        # Move to next timestep
        self.current_step += 1

        reward = self.get_reward()
        
        # Check if episode is done
        done = self.current_step == self.episode_length - 1
        
        # Get current observation with normalized internal state, profits, and memory
        feature_obs = self.current_data_slice[self.current_step]
        
        normalized_state = self.get_normalized_internal_state()
        profits = self.getProfits(scale=True)
        observation = np.concatenate([feature_obs, normalized_state, profits, self.memory])
        
        info = {}
        
        return observation, reward, done, info

    def get_reward(self) -> float:
        """Get reward for the current step."""
        current_portfolio_value = self.get_total_position_value()   
        reward = current_portfolio_value - self.previous_portfolio_value
        self.previous_portfolio_value = current_portfolio_value
        return 100000 * reward
    
    def _execute_trades(self, action: np.ndarray) -> float:
        """
        Execute CFD trades based on the agent's actions.
        
        Action space structure (num_pairs + 2 dimensions):
        - First num_pairs actions: symbol selection (max determines which pair)
        - Next action: position size (negative = no trade, positive = trade size)
        - Last action: direction (positive = long, negative = short)
        
        CFD trading logic:
        - Track entry price, position size, and direction (long/short)
        - Calculate P&L based on price difference and direction
        - Handle USD as base vs quote currency correctly
        
        Args:
            action: Array of actions with num_pairs + 2 dimensions
        """
        
        # Parse action components
        symbol_actions = action[:len(self.pairs)]
        position_size_action = action[len(self.pairs)]
        direction_action = action[len(self.pairs) + 1]
        balance = self.internal_state[-1]
        
        # Determine which pair to trade (max symbol action)
        max_symbol_idx = np.argmax(symbol_actions)
        
        # Check if we should trade
        if position_size_action > 0:
            pair = self.pairs[max_symbol_idx]
            
            # Get current price
            current_idx = self.episode_start_idx + self.current_step
            if current_idx >= len(self.pair_close):
                return  # Can't trade if no price data
            
            current_price = self.pair_close[current_idx, max_symbol_idx]
            
            # Determine direction
            is_long = direction_action > 0
            
            usd_is_quote = pair.endswith('usd') and not pair.startswith('usd')
            
            if pair not in self.cfd_positions and usd_is_quote:
                position_size_in_usd = position_size_action * self.max_position_size * balance
                position_size_in_base = position_size_in_usd / current_price
                self.cfd_positions[pair] = {
                    'size_in_base': position_size_in_base,
                    'entry_price': current_price,
                    'is_long': is_long,
                }
                self.internal_state[max_symbol_idx] = position_size_in_usd
                self.internal_state[-1] = balance - position_size_in_usd
            elif pair not in self.cfd_positions and not usd_is_quote:
                position_size_in_usd = position_size_action * self.max_position_size * balance
                position_size_in_base = position_size_in_usd
                self.cfd_positions[pair] = {
                    'size_in_base': position_size_in_base,
                    'entry_price': current_price,
                    'is_long': is_long,
                }
                self.internal_state[max_symbol_idx] = position_size_in_usd
                self.internal_state[-1] = balance - position_size_in_usd
            elif pair in self.cfd_positions:
                value = self.get_position_value(pair)
                self.internal_state[-1] += value
                del self.cfd_positions[pair]
                self.internal_state[max_symbol_idx] = 0
    
    def get_normalized_internal_state(self) -> np.ndarray:
        """Get normalized internal state (proportions of positions and balance)."""
        total_sum = np.sum(self.internal_state)
        if total_sum <= 0:
            return np.zeros(len(self.internal_state))
        
        return self.internal_state / total_sum

    def get_position_value(self, pair: str) -> float:
        """Get current position value of a pair."""
        if pair not in self.cfd_positions:
            return 0
        usd_is_quote = pair.endswith('usd') and not pair.startswith('usd')
        position = self.cfd_positions[pair]
        current_idx = self.episode_start_idx + self.current_step
        pair_idx = self.pairs.index(pair)
        current_price = self.pair_close[current_idx, pair_idx]
        pnl = 0
        if usd_is_quote:
            if position['is_long']:
                pnl = (current_price - position['entry_price']) * position['size_in_base']
            else:
                pnl = (position['entry_price'] - current_price) * position['size_in_base']
        else:
            if position['is_long']:
                pnl = (current_price - position['entry_price']) * position['size_in_base'] / current_price
            else:
                pnl = (position['entry_price'] - current_price) * position['size_in_base'] / current_price
        return pnl + self.internal_state[pair_idx]
    
    def get_total_position_value(self) -> float:
        """Get total position value of all pairs."""
        total_value = 0
        for pair in self.pairs:
            total_value += self.get_position_value(pair)
        return total_value + self.internal_state[-1]

    def getProfits(self, scale: bool) -> np.ndarray:
        """
        Calculate profit for each position as log(current_value / initial_usd_amount).
        
        Returns:
            Array of size len(pairs) with profit values for each position.
            Returns 0 for positions that don't exist or have size 0.
        """
        profits = np.zeros(len(self.pairs), dtype=np.float32)
        
        for i, pair in enumerate(self.pairs):
            if pair in self.cfd_positions:                
                # Get initial USD amount invested in this position
                initial_usd_amount = self.internal_state[i]
                
                # Skip if no initial investment
                if initial_usd_amount <= 0:
                    profits[i] = 0.0
                    continue
                
                # Get current position value
                current_value = self.get_position_value(pair)
                
                # Calculate profit as log(current_value / initial_usd_amount)
                if current_value > 0:
                    profits[i] = np.log(current_value / initial_usd_amount)
                else:
                    profits[i] = 0.0
            else:
                # No position exists for this pair
                profits[i] = 0.0
        if scale:
            profits *= 100
        return profits

    def close(self):
        """Clean up resources."""
        pass


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = ForexTradingEnv(
        data_path="../../data-pipeline/data/1H_forex_data_train.parquet",
        episode_length=50,
        initial_balance=1,
        transaction_fee=0.0001,
        max_position_size=0.1
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of pairs: {len(env.pairs)}")
    print(f"Feature dimension: {env.feature_data.shape[1]}")
    print(f"State dimension: {len(env.pairs) + 1}")
    print(f"Profits dimension: {len(env.pairs)}")
    print(f"Total observation dimension: {env.observation_space.shape[0]}")
    
    # Test a few episodes
    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")
        obs = env.reset()
        total_reward = 0
        
        for step in range(1000):  # Just test first 10 steps
            # Random actions
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if step == 0:  # Print first step info
                print(f"Initial observation shape: {obs.shape}")
                print(f"Action shape: {action.shape}")
                print(f"Reward: {reward:.4f}")
                print(f"Initial profits: {env.getProfits()}")
            
            if done:
                print(f"Final profits: {env.getProfits()}")
                break
        
        print(f"Episode {episode + 1} finished. Total reward: {total_reward:.4f}")
