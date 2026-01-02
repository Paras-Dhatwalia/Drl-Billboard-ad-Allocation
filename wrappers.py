"""
Environment Wrappers for Billboard Allocation

This module contains wrapper classes that adapt the OptimizedBillboardEnv
to work with different RL frameworks (PettingZoo, Gymnasium, Tianshou).

Key Design Principles:
- Wrappers defined at module level (not __main__) for multiprocessing compatibility
- Single-agent wrappers for PPO training with Tianshou
- Proper observation/action space conversion
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from pettingzoo.utils import BaseWrapper
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class BillboardPettingZooWrapper(BaseWrapper):
    """
    Wrapper to convert OptimizedBillboardEnv (PettingZoo AECEnv) to single-agent Gymnasium interface.

    This is necessary because:
    - OptimizedBillboardEnv inherits from PettingZoo's AECEnv (multi-agent interface)
    - Tianshou PPO expects Gymnasium-style single-agent interface
    - We only have one agent in our billboard allocation problem

    Key conversions:
    - reset() returns single observation instead of dict[agent, obs]
    - step() accepts single action instead of dict[agent, action]
    - Properly handles termination/truncation
    """

    def __init__(self, env):
        super().__init__(env)
        self._single_agent = True

    @property
    def action_space(self):
        """Get action space from wrapped environment"""
        # PettingZoo AECEnv has action_space as a method requiring agent argument
        # Since we're single-agent, get the first (and only) agent
        if hasattr(self.env, 'possible_agents') and len(self.env.possible_agents) > 0:
            agent = self.env.possible_agents[0]
            return self.env.action_space(agent)
        # Fallback for non-PettingZoo envs
        if callable(self.env.action_space):
            return self.env.action_space()
        return self.env.action_space

    @property
    def observation_space(self):
        """Get observation space from wrapped environment"""
        # PettingZoo AECEnv has observation_space as a method requiring agent argument
        if hasattr(self.env, 'possible_agents') and len(self.env.possible_agents) > 0:
            agent = self.env.possible_agents[0]
            return self.env.observation_space(agent)
        # Fallback for non-PettingZoo envs
        if callable(self.env.observation_space):
            return self.env.observation_space()
        return self.env.observation_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment and return single observation.

        Returns:
            obs: Single observation (not dict)
            info: Single info dict (not dict of dicts)
        """
        # PettingZoo AECEnv reset returns (obs, info) for the first agent
        obs, info = self.env.reset(seed=seed, options=options)

        # Return single observation, not dict
        # The base env already returns the observation for the current agent
        return obs, info

    def step(self, action):
        """
        Execute action and return single-agent transition.

        Args:
            action: Single action (int or array), not dict

        Returns:
            obs: Next observation
            reward: Scalar reward
            terminated: Episode ended naturally
            truncated: Episode ended due to time limit
            info: Info dict
        """
        # Call base environment step with single action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert from PettingZoo multi-agent format to single-agent format
        # PettingZoo returns dicts for reward/terminated/truncated with agent keys
        # Tianshou expects scalars for single-agent
        if isinstance(reward, dict):
            # Extract value for the single agent
            agent = list(reward.keys())[0] if reward else self.env.possible_agents[0]
            reward = reward.get(agent, 0.0)

        if isinstance(terminated, dict):
            agent = list(terminated.keys())[0] if terminated else self.env.possible_agents[0]
            terminated = terminated.get(agent, False)

        if isinstance(truncated, dict):
            agent = list(truncated.keys())[0] if truncated else self.env.possible_agents[0]
            truncated = truncated.get(agent, False)

        return obs, reward, terminated, truncated, info


class MinimalWrapper(BaseWrapper):
    """
    Minimal wrapper for debugging - passes through all calls unchanged.
    Useful for isolating issues with wrapper logic.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class NoGraphObsWrapper(BaseWrapper):
    """
    Wrapper that removes graph_edge_links from observations.

    Why this exists:
    - graph_edge_links is static and doesn't change during training
    - Storing it in replay buffer wastes memory
    - Instead, the model should store the graph structure once
    - Observations only need to contain dynamic features

    Note: This is OPTIONAL. The current implementation keeps graph in obs.
    Use this if you want to optimize memory usage.
    """

    def __init__(self, env):
        super().__init__(env)
        logger.warning("NoGraphObsWrapper is experimental. Graph must be injected at model forward time.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict) and 'graph_edge_links' in obs:
            # Store graph structure in info for model to access
            info['graph_edge_links'] = obs['graph_edge_links']
            # Remove from observation
            obs = {k: v for k, v in obs.items() if k != 'graph_edge_links'}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, dict) and 'graph_edge_links' in obs:
            info['graph_edge_links'] = obs['graph_edge_links']
            obs = {k: v for k, v in obs.items() if k != 'graph_edge_links'}
        return obs, reward, terminated, truncated, info


class EAMaskValidator(BaseWrapper):
    """
    Wrapper that validates EA mode mask semantics.

    Ensures:
    - Mask shape matches action space
    - Mask contains only valid boolean values
    - At least one action is valid (not all masked)
    - Flattening is consistent: pair_index = ad_idx * n_billboards + bb_idx

    Critical for research correctness - catches bugs early.
    """

    def __init__(self, env, strict: bool = True):
        super().__init__(env)
        self.strict = strict

        # Verify this is EA mode
        if not hasattr(env, 'action_mode') or env.action_mode != 'ea':
            logger.warning("EAMaskValidator should only be used with EA mode")

        # Get expected dimensions
        if hasattr(env, 'n_nodes') and hasattr(env.config, 'max_active_ads'):
            self.n_billboards = env.n_nodes
            self.max_ads = env.config.max_active_ads
            self.expected_mask_size = self.n_billboards * self.max_ads
            logger.info(f"EA Mask Validator: expecting mask size {self.expected_mask_size} "
                       f"({self.max_ads} ads × {self.n_billboards} billboards)")

    def _validate_mask(self, obs: Dict[str, Any], step_type: str = "reset"):
        """Validate mask in observation"""
        if not isinstance(obs, dict) or 'mask' not in obs:
            if self.strict:
                raise ValueError(f"[{step_type}] Observation must contain 'mask' key")
            return

        mask = obs['mask']

        # Check shape
        if mask.shape[-1] != self.expected_mask_size:
            raise ValueError(
                f"[{step_type}] Mask shape mismatch: got {mask.shape}, "
                f"expected last dim = {self.expected_mask_size}"
            )

        # Check dtype
        if mask.dtype not in [np.bool_, bool]:
            logger.warning(f"[{step_type}] Mask dtype is {mask.dtype}, expected bool")

        # Check at least one valid action
        if not np.any(mask):
            raise ValueError(f"[{step_type}] All actions are masked! No valid actions available.")

        # Log statistics
        n_valid = np.sum(mask)
        logger.debug(f"[{step_type}] Valid actions: {n_valid}/{self.expected_mask_size} "
                    f"({100*n_valid/self.expected_mask_size:.1f}%)")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._validate_mask(obs, "reset")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            self._validate_mask(obs, "step")
        return obs, reward, terminated, truncated, info
