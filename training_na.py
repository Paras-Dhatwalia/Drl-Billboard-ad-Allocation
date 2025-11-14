"""
Compact PPO Training for Billboard Allocation using Tianshou
Simplified and streamlined version matching the reference implementation style
"""

import os
import torch
import tianshou as ts
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import ExponentialLR
from pettingzoo.utils import BaseWrapper
from typing import Dict, Any
import platform
import gymnasium as gym  

# Import your environment and model
from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN, DEFAULT_CONFIGS

# Configuration - FIXED: Use raw strings or forward slashes for Windows paths
env_config = {
    "billboard_csv": r"C:\Coding Files\DynamicBillboard\env\BillBoard_NYC.csv",
    "advertiser_csv": r"C:\Coding Files\DynamicBillboard\env\Advertiser_NYC2.csv",
    "trajectory_csv": r"C:\Coding Files\DynamicBillboard\env\TJ_NYC.csv",
    "action_mode": "na",
    "max_events": 1000,
    "influence_radius": 500.0,
    "tardiness_cost": 50.0
}

train_config = {
    "hidden_dim": 128,
    "n_graph_layers": 3,
    "lr": 3e-4,
    "discount_factor": 0.99,
    "batch_size": 64,
    "nr_envs": 4,
    "max_epoch": 50,
    "step_per_collect": 2048,
    "step_per_epoch": 100000,
    "repeat_per_collect": 10,
    "save_path": "models/ppo_billboard_na.pt",
    "log_path": "logs/ppo_billboard_na"
}


# FIXED: Improved wrapper for PettingZoo compatibility
class BillboardPettingZooWrapper(BaseWrapper):
    """Wrapper to make Billboard env compatible with Tianshou"""

    def __init__(self, env):
        super().__init__(env)
        self._last_obs = None

    def reset(self, seed=None, options=None):
        """Reset environment and return proper format"""
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        # Return single agent observation
        return obs, info

    def step(self, action):
        """Execute step and return proper format"""
        obs, rewards, terminations, truncations, infos = self.env.step(action)
        self._last_obs = obs

        # Extract single agent values
        reward = rewards.get(self.env.agent_selection, 0.0)
        terminated = terminations.get(self.env.agent_selection, False)
        truncated = truncations.get(self.env.agent_selection, False)
        info = infos.get(self.env.agent_selection, {})

        return obs, reward, terminated, truncated, info

    def observe(self, agent=None):
        """Return last observation"""
        return self._last_obs if self._last_obs is not None else self.env.observe(self.env.agent_selection)


def get_env():
    """Create wrapped environment instance"""
    env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode=env_config["action_mode"],
        config=EnvConfig(
            max_events=env_config["max_events"],
            influence_radius_meters=env_config["influence_radius"],
            tardiness_cost=env_config["tardiness_cost"]
        )
    )
    return BillboardPettingZooWrapper(env)


def preprocess_observations(obs_dict, device):
    """Convert single observation dict to torch tensors"""
    if not isinstance(obs_dict, dict):
        return obs_dict

def main():
    """Main training function - compact version"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories for saving
    os.makedirs(os.path.dirname(train_config["save_path"]), exist_ok=True)
    os.makedirs(train_config["log_path"], exist_ok=True)

    # Create environments
    print("Creating environments...")

    # Windows compatibility fix
    if platform.system() == "Windows":
        train_envs = ts.env.DummyVectorEnv([get_env for _ in range(train_config["nr_envs"])])
    else:
        train_envs = ts.env.SubprocVectorEnv([get_env for _ in range(train_config["nr_envs"])])

    test_envs = ts.env.DummyVectorEnv([get_env for _ in range(2)])

    # Get environment info for model config
    sample_env = get_env()
    n_billboards = sample_env.env.n_nodes
    print(f"Environment has {n_billboards} billboards")

    # Create model using existing config
    model_config = DEFAULT_CONFIGS['na_billboard_nyc'].copy()
    model_config['n_billboards'] = n_billboards
    model_config['hidden_dim'] = train_config['hidden_dim']
    model_config['n_graph_layers'] = train_config['n_graph_layers']

    print(f"Creating model with config: {model_config}")

    # Initialize actor and critic networks
    actor = BillboardAllocatorGNN(**model_config).to(device)

    # Critic uses same architecture but separate parameters
    critic_config = model_config.copy()
    critic = BillboardAllocatorGNN(**critic_config).to(device)

    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=train_config["lr"]
    )
    lr_scheduler = ExponentialLR(optimizer, 0.95)
    
    # Get action space from sample environment
    sample_env_for_space = get_env()
    action_space = sample_env_for_space.action_space(sample_env_for_space.possible_agents[0])

    # Create PPO policy with correct parameters
  
    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        dist_fn=torch.distributions.Categorical,
        action_space=action_space,
        action_mask_key="mask", 
        discount_factor=train_config["discount_factor"],
        gae_lambda=train_config.get("gae_lambda", 0.95),
        vf_coef=train_config.get("vf_coef", 0.5),
        ent_coef=train_config.get("ent_coef", 0.01),
        max_grad_norm=train_config.get("max_grad_norm", 0.5),
        eps_clip=train_config.get("eps_clip", 0.2),
        value_clip=True,
        deterministic_eval=True
    )

   # Create collectors
    train_collector = ts.data.Collector(
        policy, train_envs,
        ts.data.VectorReplayBuffer(20000, train_config["nr_envs"]),
        exploration_noise=True
    )

    test_collector = ts.data.Collector(
        policy, test_envs,
        exploration_noise=False
    )
    # Setup logging
    writer = SummaryWriter(train_config["log_path"])
    logger = TensorboardLogger(writer)

    # Save function
    def save_best_fn(policy):
        print("Saving improved policy...")
        torch.save(policy.state_dict(), train_config["save_path"])

    # Train
    print("Starting training...")
    try:
        trainer = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=train_config["max_epoch"],
            step_per_epoch=train_config["step_per_epoch"],
            repeat_per_collect=train_config.get("repeat_per_collect", 10),
            episode_per_test=10,
            batch_size=train_config["batch_size"],
            step_per_collect=train_config["step_per_collect"],
            save_best_fn=save_best_fn,
            logger=logger
        )
        result = trainer.run()
        print(f'Training complete! Duration: {result["duration"]}')
        print(f'Best reward: {result["best_reward"]}')

        # Save final model
        torch.save(policy.state_dict(), f"{train_config['save_path']}.final")

        return result

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        train_envs.close()
        test_envs.close()
        writer.close()


if __name__ == "__main__":
    # FIXED: Proper argument parsing with error handling
    import argparse

    parser = argparse.ArgumentParser(description='Train PPO agent for billboard allocation')
    parser.add_argument('--billboards', type=str, help='Path to billboard CSV')
    parser.add_argument('--advertisers', type=str, help='Path to advertiser CSV')
    parser.add_argument('--trajectories', type=str, help='Path to trajectory CSV')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    # Update configs with command line args if provided
    if args.billboards:
        env_config['billboard_csv'] = args.billboards
    if args.advertisers:
        env_config['advertiser_csv'] = args.advertisers
    if args.trajectories:
        env_config['trajectory_csv'] = args.trajectories

    train_config['max_epoch'] = args.epochs
    train_config['lr'] = args.lr
    train_config['batch_size'] = args.batch_size

    # Run training
    result = main()

    if result is not None:
        print("\nâœ… Training completed successfully!")
    else:
        print("\nâŒ Training failed!")