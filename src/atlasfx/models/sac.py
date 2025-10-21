"""
Soft Actor-Critic (SAC) for stochastic trading policy learning.

This module implements the SAC algorithm from:
Haarnoja et al. (2018) - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"

Key features:
- Maximum entropy framework for robust exploration
- Twin Q-networks to reduce overestimation bias
- Automatic entropy temperature tuning
- Off-policy learning with replay buffer
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Stochastic policy network (actor) for SAC.

    Outputs mean and log-std for a Gaussian policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        """
        Initialize actor network.

        Args:
            state_dim: Dimension of state (latent + forecasts)
            action_dim: Dimension of action space (position, size, etc.)
            hidden_dims: List of hidden layer dimensions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # TODO: Implement actor architecture
        # - MLP with hidden_dims layers
        # - Output: mean and log_std for each action
        # - Tanh squashing for bounded actions
        raise NotImplementedError("Actor not implemented yet")

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Tuple of (action, log_prob, mean) where:
                action: Sampled action (batch_size, action_dim)
                log_prob: Log probability of action (batch_size, 1)
                mean: Mean of action distribution (batch_size, action_dim)
        """
        raise NotImplementedError("Actor forward not implemented yet")

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action from policy (for evaluation).

        Args:
            state: State tensor (batch_size, state_dim)
            deterministic: If True, return mean action; else sample

        Returns:
            Action tensor (batch_size, action_dim)
        """
        raise NotImplementedError("Actor get_action not implemented yet")


class Critic(nn.Module):
    """
    Q-network (critic) for SAC.

    Estimates Q(s, a) for state-action pairs.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
    ) -> None:
        """
        Initialize critic network.

        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # TODO: Implement critic architecture
        # - MLP with hidden_dims layers
        # - Input: concatenated [state, action]
        # - Output: Q-value (scalar)
        raise NotImplementedError("Critic not implemented yet")

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic.

        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)

        Returns:
            Q-value (batch_size, 1)
        """
        raise NotImplementedError("Critic forward not implemented yet")


class SAC(nn.Module):
    """
    Soft Actor-Critic agent.

    Combines:
    - Stochastic actor for policy
    - Twin critics for Q-function estimation
    - Target networks for stability
    - Automatic entropy temperature tuning
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
    ) -> None:
        """
        Initialize SAC agent.

        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action
            hidden_dims: Hidden dimensions for networks
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critics
            lr_alpha: Learning rate for entropy temperature
            gamma: Discount factor
            tau: Target network update rate
            alpha: Initial entropy temperature
            automatic_entropy_tuning: Whether to auto-tune alpha
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dims)

        # Twin critic networks
        self.critic1 = Critic(state_dim, action_dim, hidden_dims)
        self.critic2 = Critic(state_dim, action_dim, hidden_dims)

        # Target critic networks
        self.critic1_target = Critic(state_dim, action_dim, hidden_dims)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dims)

        # Copy weights to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Entropy temperature
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        # TODO: Setup optimizers
        raise NotImplementedError("SAC initialization not complete")

    def select_action(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """
        Select action from policy.

        Args:
            state: State tensor (batch_size, state_dim)
            evaluate: If True, use deterministic policy

        Returns:
            Action tensor (batch_size, action_dim)
        """
        raise NotImplementedError("SAC select_action not implemented yet")

    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> dict[str, float]:
        """
        Update SAC networks.

        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            reward: Reward tensor (batch_size, 1)
            next_state: Next state tensor (batch_size, state_dim)
            done: Done flag (batch_size, 1)

        Returns:
            Dictionary with losses:
                - 'critic1_loss'
                - 'critic2_loss'
                - 'actor_loss'
                - 'alpha_loss' (if auto-tuning)
        """
        raise NotImplementedError("SAC update not implemented yet")

    def update_targets(self) -> None:
        """
        Soft update of target networks.

        θ_target = τ * θ + (1 - τ) * θ_target
        """
        for param, target_param in zip(
            self.critic1.parameters(), self.critic1_target.parameters(), strict=True
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.critic2.parameters(), self.critic2_target.parameters(), strict=True
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic1_target": self.critic1_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "log_alpha": self.log_alpha if self.automatic_entropy_tuning else None,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        if self.automatic_entropy_tuning and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp()


# TODO: Implement replay buffer
# TODO: Implement training loop
# TODO: Implement evaluation metrics
# TODO: Implement tensorboard logging
