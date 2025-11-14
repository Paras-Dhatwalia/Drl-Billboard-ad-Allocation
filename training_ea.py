"""
PPO Training for Billboard Allocation - EA (Edge Action) Mode
This mode allows the agent to directly select ad-billboard pairs
"""

import os
import torch
import tianshou as ts
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from pettingzoo.utils import BaseWrapper
from typing import Dict, Any
import logging

# Import your environment and model
from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN, DEFAULT_CONFIGS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for EA mode
env_config = {
    "billboard_csv": r"C:\Coding Files\DynamicBillboard\env\BillBoard_NYC.csv", 
    "advertiser_csv": r"C:\Coding Files\DynamicBillboard\env\Advertiser_NYC2.csv", 
    "trajectory_csv": r"C:\Coding Files\DynamicBillboard\env\TJ_NYC.csv", 
    "action_mode": "ea",
    "max_events": 1000,
    "influence_radius": 500.0,
    "tardiness_cost": 50.0
}

train_config = {
    "hidden_dim": 256,  # Larger for complex pair interactions
    "n_graph_layers": 4,
    "lr": 2e-4,  # Lower learning rate for stability
    "discount_factor": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ent_coef": 0.02,  # Higher entropy for exploration
    "max_grad_norm": 0.5,
    "eps_clip": 0.2,
    "batch_size": 128,  # Larger batch size
    "nr_envs": 6,
    "max_epoch": 150,
    "step_per_collect": 3072,
    "step_per_epoch": 100000,
    "repeat_per_collect": 15,
    "save_path": "models/ppo_billboard_ea.pt",
    "log_path": "logs/ppo_billboard_ea"
}


class BillboardPettingZooWrapper(BaseWrapper):
    """Wrapper for PettingZoo compatibility"""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        observations = {agent: obs for agent in self.agents}
        infos = {agent: info for agent in self.agents}
        return observations, infos

    def step(self, actions):
        action = actions.get(self.agent_selection, actions)
        return self.env.step(action)


def get_env():
    """Create wrapped environment for EA mode"""
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


def preprocess_observations(**kwargs):
    """Convert numpy observations to torch tensors for EA mode"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_obs(obs_dict):
        if isinstance(obs_dict, dict) and 'graph_nodes' in obs_dict:
            processed = {
                "graph_nodes": torch.from_numpy(obs_dict['graph_nodes']).float().to(device),
                "graph_edge_links": torch.from_numpy(obs_dict['graph_edge_links']).long().to(device),
                "ad_features": torch.from_numpy(obs_dict['ad_features']).float().to(device),
                "mask": torch.from_numpy(obs_dict['mask']).bool().to(device)
            }

            # Handle EA mode mask shape (should be flattened)
            if len(processed["mask"].shape) == 3:
                batch_size, max_ads, n_billboards = processed["mask"].shape
                processed["mask"] = processed["mask"].reshape(batch_size, -1)

            return processed
        return obs_dict

    if "obs" in kwargs:
        kwargs["obs"] = [process_obs(obs) for obs in kwargs["obs"]]
    if "obs_next" in kwargs:
        kwargs["obs_next"] = [process_obs(obs) for obs in kwargs["obs_next"]]

    return kwargs


def main():
    """Main training function for EA mode"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training EA mode on device: {device}")

    # Create environments
    logger.info("Creating environments...")
    train_envs = ts.env.SubprocVectorEnv([get_env for _ in range(train_config["nr_envs"])])
    test_envs = ts.env.DummyVectorEnv([get_env for _ in range(2)])

    # Get environment info
    sample_env = get_env()
    n_billboards = sample_env.env.n_nodes
    max_ads = sample_env.env.config.max_active_ads
    logger.info(f"Environment: {n_billboards} billboards, max {max_ads} ads")

    # Create model configuration for EA mode
    model_config = {
        'node_feat_dim': 10,
        'ad_feat_dim': 8,
        'hidden_dim': train_config['hidden_dim'],
        'n_graph_layers': train_config['n_graph_layers'],
        'mode': 'ea',  # Edge Action mode
        'n_billboards': n_billboards,
        'max_ads': max_ads,
        'use_attention': True,
        'conv_type': 'gat',  # GAT for better pair modeling
        'dropout': 0.15
    }

    # Initialize networks
    logger.info("Creating actor and critic networks...")
    actor = BillboardAllocatorGNN(**model_config).to(device)
    critic = BillboardAllocatorGNN(**model_config).to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in actor.parameters()) + \
                   sum(p.numel() for p in critic.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=train_config["lr"],
        eps=1e-5
    )

    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config["max_epoch"],
        eta_min=train_config["lr"] * 0.1
    )

    # Create PPO policy
    policy = ts.policy.PPOPolicy(
        actor, critic, optimizer,
        discount_factor=train_config["discount_factor"],
        gae_lambda=train_config["gae_lambda"],
        vf_coef=train_config["vf_coef"],
        ent_coef=train_config["ent_coef"],
        max_grad_norm=train_config["max_grad_norm"],
        eps_clip=train_config["eps_clip"],
        value_clip=True,
        dist_fn=torch.distributions.Categorical,
        deterministic_eval=True,
        lr_scheduler=lr_scheduler
    )

    # Create collectors
    buffer_size = max(30000, train_config["step_per_collect"] * 2)
    train_collector = ts.data.Collector(
        policy, train_envs,
        ts.data.VectorReplayBuffer(buffer_size, train_config["nr_envs"]),
        exploration_noise=True,
        preprocess_fn=preprocess_observations
    )

    test_collector = ts.data.Collector(
        policy, test_envs,
        exploration_noise=False,
        preprocess_fn=preprocess_observations
    )

    # Setup logging
    os.makedirs(os.path.dirname(train_config["log_path"]), exist_ok=True)
    writer = SummaryWriter(train_config["log_path"])
    logger_tb = TensorboardLogger(writer)

    # Save function
    os.makedirs(os.path.dirname(train_config["save_path"]), exist_ok=True)
    best_reward = -float('inf')

    def save_best_fn(policy):
        nonlocal best_reward
        test_result = test_collector.collect(n_episode=10)
        current_reward = test_result["rews"].mean()

        if current_reward > best_reward:
            best_reward = current_reward
            logger.info(f"New best reward: {best_reward:.2f}, saving model...")
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'best_reward': best_reward
            }, train_config["save_path"])

    # Train
    logger.info("Starting EA mode training...")
    logger.info(f"Epochs: {train_config['max_epoch']}")
    logger.info(f"Steps per epoch: {train_config['step_per_epoch']}")
    logger.info(f"Batch size: {train_config['batch_size']}")

    result = ts.trainer.onpolicy_trainer(
        policy, train_collector,
        test_collector=test_collector,
        max_epoch=train_config["max_epoch"],
        step_per_epoch=train_config["step_per_epoch"],
        step_per_collect=train_config["step_per_collect"],
        episode_per_test=10,
        batch_size=train_config["batch_size"],
        repeat_per_collect=train_config["repeat_per_collect"],
        save_best_fn=save_best_fn,
        logger=logger_tb,
        show_progress=True,
        test_in_train=True
    )

    # Save final model
    final_path = train_config["save_path"].replace('.pt', '_final.pt')
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'config': model_config,
        'training_config': train_config,
        'final_reward': result.get("best_reward", 0)
    }, final_path)

    logger.info(f'EA mode training complete!')
    logger.info(f'Duration: {result.get("duration", "N/A")}')
    logger.info(f'Best reward: {best_reward:.2f}')
    logger.info(f'Model saved to: {train_config["save_path"]}')

    # Clean up
    train_envs.close()
    test_envs.close()
    writer.close()

    return result


if __name__ == "__main__":
    main()