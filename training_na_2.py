"""
PPO Training for Billboard Allocation - NA Mode
FIXED: Correctly handles Tianshou Batch objects in Graph Injection
"""

import os
import torch
import tianshou as ts
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import ExponentialLR
from typing import Dict, Any, Optional, Tuple, Union
import platform
import gymnasium as gym
from gymnasium import spaces

from tianshou.trainer import OnpolicyTrainer

from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN


# =============================================================================
# WRAPPER: Returns observations WITHOUT graph_edge_links
# =============================================================================

class NoGraphObsWrapper(gym.Env):
    """
    Wrapper that completely removes graph_edge_links from observations.
    The graph is accessed separately via get_graph() method.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, env: OptimizedBillboardEnv):
        super().__init__()
        self.env = env
        self._agent = "Allocator_0"
        
        # Store the graph ONCE
        self._graph = env.edge_index.copy()
        
        # Create observation space WITHOUT graph
        orig_space = env.observation_spaces[self._agent]
        self._observation_space = spaces.Dict({
            k: v for k, v in orig_space.spaces.items() 
            if k != 'graph_edge_links'
        })
        self._action_space = env.action_spaces[self._agent]
    
    def get_graph(self) -> np.ndarray:
        """Get the graph structure (call once, store in model)."""
        return self._graph
    
    @property
    def action_space(self) -> spaces.Space:
        return self._action_space
    
    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space
    
    def _strip_obs(self, obs: Dict) -> Dict:
        """Remove graph_edge_links from observation."""
        return {k: v for k, v in obs.items() if k != 'graph_edge_links'}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._strip_obs(obs), info
    
    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, rewards, terminations, truncations, infos = self.env.step(action)
        
        reward = rewards.get(self._agent, 0.0) if isinstance(rewards, dict) else rewards
        terminated = terminations.get(self._agent, False) if isinstance(terminations, dict) else terminations
        truncated = truncations.get(self._agent, False) if isinstance(truncations, dict) else truncations
        info = infos.get(self._agent, {}) if isinstance(infos, dict) else infos
        
        return self._strip_obs(obs), float(reward), bool(terminated), bool(truncated), info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    # Expose env properties
    @property
    def n_nodes(self): return self.env.n_nodes
    @property
    def config(self): return self.env.config
    @property
    def edge_index(self): return self.env.edge_index


# =============================================================================
# MODEL WRAPPER: Stores graph internally, injects during forward
# =============================================================================

class GraphAwareActor(torch.nn.Module):
    """
    Actor wrapper that stores graph as a buffer and injects it during forward.
    """
    
    def __init__(self, model: BillboardAllocatorGNN, graph: np.ndarray):
        super().__init__()
        self.model = model
        # Register graph as a buffer (not a parameter, won't be trained)
        self.register_buffer('graph', torch.from_numpy(graph).long())
        self.n_nodes = model.n_billboards
    
    def forward(self, obs, state=None, info={}):
        """Add graph to observation and forward to model."""
        # FIX: Directly handle obs whether it is a Batch or a Dict
        obs_with_graph = self._add_graph_to_obs(obs)
        return self.model(obs_with_graph, state, info)
    
    def _add_graph_to_obs(self, obs: Union[Dict, ts.data.Batch]):
        """Add graph_edge_links to observation."""
        
        # 1. Helper to get values safely from Dict or Batch
        def get(obj, key):
            if isinstance(obj, dict): return obj.get(key)
            return getattr(obj, key, None)

        # 2. If graph already exists, return as is
        if get(obs, 'graph_edge_links') is not None:
            return obs
            
        # 3. Get nodes to determine batch size
        nodes = get(obs, 'graph_nodes')
        if nodes is None:
            return obs
            
        if isinstance(nodes, np.ndarray):
            nodes = torch.from_numpy(nodes).float()
            
        # Determine batch size
        # If nodes is (B, N, F), batch_size is B. If (N, F), batch_size is 1.
        batch_size = nodes.shape[0] if len(nodes.shape) == 3 else 1
        device = nodes.device
        
        # 4. Prepare the graph batch: (2, E) -> (B, 2, E)
        graph_batch = self.graph.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        
        # 5. Inject based on type
        if isinstance(obs, dict):
            new_obs = obs.copy()
            new_obs['graph_edge_links'] = graph_batch
            return new_obs
        else:
            # It is a Tianshou Batch
            # Create a shallow copy to prevent in-place modification issues
            new_obs = ts.data.Batch(obs)
            new_obs.graph_edge_links = graph_batch
            return new_obs


class GraphAwareCritic(torch.nn.Module):
    """
    Critic wrapper that stores graph as a buffer and injects it during forward.
    """
    
    def __init__(self, model: BillboardAllocatorGNN, graph: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer('graph', torch.from_numpy(graph).long())
    
    def forward(self, obs, state=None, info={}):
        """Add graph to observation and call critic_forward."""
        obs_with_graph = self._add_graph_to_obs(obs)
        return self.model.critic_forward(obs_with_graph)
    
    def _add_graph_to_obs(self, obs: Union[Dict, ts.data.Batch]):
        """Add graph_edge_links to observation."""
        
        def get(obj, key):
            if isinstance(obj, dict): return obj.get(key)
            return getattr(obj, key, None)

        if get(obs, 'graph_edge_links') is not None:
            return obs
            
        nodes = get(obs, 'graph_nodes')
        if nodes is None: return obs
            
        if isinstance(nodes, np.ndarray):
            nodes = torch.from_numpy(nodes).float()
            
        batch_size = nodes.shape[0] if len(nodes.shape) == 3 else 1
        device = nodes.device
        
        graph_batch = self.graph.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        
        if isinstance(obs, dict):
            new_obs = obs.copy()
            new_obs['graph_edge_links'] = graph_batch
            return new_obs
        else:
            new_obs = ts.data.Batch(obs)
            new_obs.graph_edge_links = graph_batch
            return new_obs


# =============================================================================
# CONFIGURATION
# =============================================================================

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
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "eps_clip": 0.2,
    "batch_size": 64,
    "nr_envs": 4,
    "max_epoch": 50,
    "step_per_collect": 2048,
    "step_per_epoch": 100000,
    "repeat_per_collect": 10,
    "save_path": "models/ppo_billboard_na.pt",
    "log_path": "logs/ppo_billboard_na",
    "buffer_size": 20000  # Can be large now - no graph in observations!
}

# Global graph storage for environment factory
GRAPH_NUMPY = None


def get_env():
    """Create wrapped environment."""
    global GRAPH_NUMPY
    
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
    wrapped = NoGraphObsWrapper(env)
    
    # Store graph globally for model initialization
    if GRAPH_NUMPY is None:
        GRAPH_NUMPY = wrapped.get_graph()
    
    return wrapped


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    global GRAPH_NUMPY
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Tianshou version: {ts.__version__}")

    os.makedirs(os.path.dirname(train_config["save_path"]), exist_ok=True)
    os.makedirs(train_config["log_path"], exist_ok=True)

    # Create sample environment first
    print("\n1. Creating sample environment...")
    sample_env = get_env()
    n_billboards = sample_env.n_nodes
    max_ads = sample_env.config.max_active_ads
    
    print(f"   Billboards: {n_billboards}")
    print(f"   Max ads: {max_ads}")
    print(f"   Graph edges: {GRAPH_NUMPY.shape[1]}")
    print(f"   Graph stored separately (NOT in observations)")
    
    # Verify observation space has no graph
    obs, _ = sample_env.reset()
    print(f"   Observation keys: {list(obs.keys())}")
    assert 'graph_edge_links' not in obs, "Graph should not be in observations!"
    
    action_space = sample_env.action_space
    print(f"   Action space: {action_space}")

    # Create vectorized environments
    print("\n2. Creating vectorized environments...")
    if platform.system() == "Windows":
        train_envs = ts.env.DummyVectorEnv([get_env for _ in range(train_config["nr_envs"])])
        print(f"   Using DummyVectorEnv (Windows)")
    else:
        train_envs = ts.env.SubprocVectorEnv([get_env for _ in range(train_config["nr_envs"])])
        print(f"   Using SubprocVectorEnv")
    
    test_envs = ts.env.DummyVectorEnv([get_env for _ in range(2)])
    print(f"   Train envs: {train_config['nr_envs']}, Test envs: 2")

    # Create models
    print("\n3. Creating models...")
    model_config = {
        'node_feat_dim': 10,
        'ad_feat_dim': 8,
        'hidden_dim': train_config['hidden_dim'],
        'n_graph_layers': train_config['n_graph_layers'],
        'mode': 'na',
        'n_billboards': n_billboards,
        'max_ads': max_ads,
        'use_attention': True,
        'conv_type': 'gin',
        'dropout': 0.1
    }

    # Create base models
    actor_base = BillboardAllocatorGNN(**model_config)
    critic_base = BillboardAllocatorGNN(**model_config)
    
    # Wrap with graph-aware wrappers
    actor = GraphAwareActor(actor_base, GRAPH_NUMPY).to(device)
    critic = GraphAwareCritic(critic_base, GRAPH_NUMPY).to(device)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"   Actor parameters: {actor_params:,}")
    print(f"   Critic parameters: {critic_params:,}")
    print(f"   Graph buffer size: {GRAPH_NUMPY.nbytes / 1024:.1f} KB (stored once)")

    # Optimizer
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=train_config["lr"],
        eps=1e-5
    )
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    # Create PPO policy
    print("\n4. Creating PPO policy...")
    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        dist_fn=torch.distributions.Categorical,
        action_space=action_space,
        action_scaling=False,
        discount_factor=train_config["discount_factor"],
        gae_lambda=train_config["gae_lambda"],
        vf_coef=train_config["vf_coef"],
        ent_coef=train_config["ent_coef"],
        max_grad_norm=train_config["max_grad_norm"],
        eps_clip=train_config["eps_clip"],
        value_clip=True,
        deterministic_eval=True,
        lr_scheduler=lr_scheduler
    )

    # Create collectors with standard buffers (no graph = no memory issue!)
    print("\n5. Creating collectors...")
    print(f"   Buffer size: {train_config['buffer_size']} (graph-free, memory efficient)")
    
    train_collector = ts.data.Collector(
        policy, 
        train_envs,
        ts.data.VectorReplayBuffer(train_config["buffer_size"], train_config["nr_envs"]),
        exploration_noise=True
    )

    test_collector = ts.data.Collector(
        policy, 
        test_envs,
        exploration_noise=False
    )

    # Logging
    writer = SummaryWriter(train_config["log_path"])
    tb_logger = TensorboardLogger(writer)

    # Save function
    best_reward = -float('inf')
    
    def save_best_fn(policy):
        nonlocal best_reward
        test_result = test_collector.collect(n_episode=10)
        current_reward = test_result.returns.mean()
        
        if current_reward > best_reward:
            best_reward = current_reward
            print(f"\n   New best reward: {best_reward:.2f}, saving...")
            torch.save({
                'actor_state_dict': actor_base.state_dict(),
                'critic_state_dict': critic_base.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'best_reward': best_reward,
                'graph': GRAPH_NUMPY  # Save graph for inference
            }, train_config["save_path"])

    # Training
    print("\n6. Starting training...")
    print(f"   Epochs: {train_config['max_epoch']}")
    print(f"   Steps per epoch: {train_config['step_per_epoch']}")
    print(f"   Batch size: {train_config['batch_size']}")
    print("-" * 50)
    
    try:
        trainer = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=train_config["max_epoch"],
            step_per_epoch=train_config["step_per_epoch"],
            repeat_per_collect=train_config["repeat_per_collect"],
            episode_per_test=10,
            batch_size=train_config["batch_size"],
            step_per_collect=train_config["step_per_collect"],
            save_best_fn=save_best_fn,
            logger=tb_logger,
            show_progress=True,
            test_in_train=True
        )
        
        result = trainer.run()
        
        print("\n" + "=" * 50)
        print(" Training complete!")
        print(f"   Best reward: {best_reward:.2f}")
        print("=" * 50)

        # Save final model
        torch.save({
            'actor_state_dict': actor_base.state_dict(),
            'critic_state_dict': critic_base.state_dict(),
            'config': model_config,
            'training_config': train_config,
            'final_reward': best_reward,
            'graph': GRAPH_NUMPY
        }, train_config["save_path"].replace('.pt', '_final.pt'))

        return result

    except Exception as e:
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        train_envs.close()
        test_envs.close()
        writer.close()
        sample_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train PPO for billboard allocation (NA mode)')
    parser.add_argument('--billboards', type=str, help='Path to billboard CSV')
    parser.add_argument('--advertisers', type=str, help='Path to advertiser CSV')  
    parser.add_argument('--trajectories', type=str, help='Path to trajectory CSV')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--buffer-size', type=int, default=20000)

    args = parser.parse_args()

    if args.billboards:
        env_config['billboard_csv'] = args.billboards
    if args.advertisers:
        env_config['advertiser_csv'] = args.advertisers
    if args.trajectories:
        env_config['trajectory_csv'] = args.trajectories

    train_config['max_epoch'] = args.epochs
    train_config['lr'] = args.lr
    train_config['batch_size'] = args.batch_size
    train_config['buffer_size'] = args.buffer_size

    result = main()
    
    if result is not None:
        print("\n Success!")
    else:
        print("\n Training had issues")