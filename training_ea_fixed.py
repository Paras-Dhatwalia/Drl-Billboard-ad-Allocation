"""
PPO Training for Billboard Allocation - EA (Edge Action) Mode

EA Mode Design:
- Combinatorial action space: select multiple (ad, billboard) pairs simultaneously
- Uses Independent Bernoulli distribution (NOT Categorical)
- Each pair decision is independent
- Action shape: (max_ads × n_billboards) binary vector

Research Rationale:
- EA allows batch allocation decisions
- More efficient than sequential NA (Node Action) mode
- Captures ad-billboard compatibility directly
- Critical for real-time billboard allocation

Key Differences from NA mode:
1. Action distribution: Independent Bernoulli vs Categorical
2. Action space: MultiBinary vs Discrete
3. Mask handling: element-wise Bernoulli masking vs categorical masking
4. Entropy: sum of Bernoulli entropies vs categorical entropy

"""

import os
import platform
import torch
import tianshou as ts
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any
import logging

# Import environment and model
from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN, DEFAULT_CONFIGS

# Import wrappers from separate module (CRITICAL for multiprocessing)
from wrappers import BillboardPettingZooWrapper, EAMaskValidator, NoGraphObsWrapper

# Import custom distribution
from distributions import IndependentBernoulli

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


#  CONFIGURATION 

def get_test_config():
    """
    Small configuration for correctness validation.

    This to:
    - Verify code runs without errors
    - Check gradient flow
    - Validate mask handling
    - Debug model architecture

    NOT for actual experiments!
    """
    return {
        # Environment config
        "env": {
            "billboard_csv": r"C:\Coding Files\DynamicBillboard\env\BillBoard_NYC.csv",
            "advertiser_csv": r"C:\Coding Files\DynamicBillboard\env\Advertiser_NYC2.csv",
            "trajectory_csv": r"C:\Coding Files\DynamicBillboard\env\TJ_NYC.csv",
            "action_mode": "ea",
            "max_events": 50,  # Very small for quick testing
            "max_active_ads": 3,  # Reduced action space
            "influence_radius": 500.0,
            "tardiness_cost": 50.0,
        },
        # Training config
        "train": {
            "hidden_dim": 64,  # Small model
            "n_graph_layers": 2,
            "lr": 1e-3,  # Higher LR for faster convergence in testing
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.01,  # EA: lower entropy coefficient
            "max_grad_norm": 0.5,
            "eps_clip": 0.2,
            "batch_size": 16,  # Very small
            "nr_envs": 1,  # Single env for debugging
            "max_epoch": 3,  # Just test a few epochs
            "step_per_collect": 64,  # Small
            "step_per_epoch": 200,  # Small
            "repeat_per_collect": 4,
            "save_path": "models/test_ppo_billboard_ea.pt",
            "log_path": "logs/test_ppo_billboard_ea",
            "use_validation": True,  # Enable EA mask validation
        }
    }


def get_full_config():
    """
    Full configuration for actual experiments.

    This for:
    - Production training runs
    - Publishable results
    - Hyperparameter tuning

    EA-specific tuning:
    - Lower entropy coefficient (EA has high dimensionality)
    - Smaller learning rate (Bernoulli PPO is noisier)
    - Moderate batch size (helps with sparse rewards)
    """
    return {
        # Environment config
        "env": {
            "billboard_csv": r"C:\Coding Files\DynamicBillboard\env\BillBoard_NYC.csv",
            "advertiser_csv": r"C:\Coding Files\DynamicBillboard\env\Advertiser_NYC2.csv",
            "trajectory_csv": r"C:\Coding Files\DynamicBillboard\env\TJ_NYC.csv",
            "action_mode": "ea",
            "max_events": 1000,
            "max_active_ads": 20,  # Full action space
            "influence_radius": 500.0,
            "tardiness_cost": 50.0,
        },
        # Training config
        "train": {
            "hidden_dim": 256,  # Large model for complex interactions
            "n_graph_layers": 4,
            "lr": 1e-4,  # EA: lower LR for stability
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.005,  # EA: much lower entropy (high-dim action space)
            "max_grad_norm": 0.5,
            "eps_clip": 0.2,
            "batch_size": 128,
            "nr_envs": 8,  # More parallel envs
            "max_epoch": 200,
            "step_per_collect": 4096,
            "step_per_epoch": 100000,
            "repeat_per_collect": 15,
            "save_path": "models/ppo_billboard_ea.pt",
            "log_path": "logs/ppo_billboard_ea",
            "use_validation": False,  # Disable validation in production for speed
        }
    }


#  ENVIRONMENT CREATION 

def create_single_env(env_config: Dict[str, Any], use_validation: bool = False):
    """
    Create a single billboard environment with proper wrapping.

    Args:
        env_config: Environment configuration dict
        use_validation: Whether to add EA mask validation wrapper

    Returns:
        Wrapped environment ready for Tianshou
    """
    # Create base environment
    base_env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode=env_config["action_mode"],
        config=EnvConfig(
            max_events=env_config["max_events"],
            max_active_ads=env_config.get("max_active_ads", 20),
            influence_radius_meters=env_config["influence_radius"],
            tardiness_cost=env_config["tardiness_cost"]
        )
    )

    # Wrap for PettingZoo -> Gymnasium conversion
    env = BillboardPettingZooWrapper(base_env)

    # Optionally add validation wrapper (for testing/debugging)
    if use_validation:
        env = EAMaskValidator(env, strict=True)
        logger.info("Added EA mask validation wrapper")

    return env


def create_env_factory(env_config: Dict[str, Any], use_validation: bool = False):
    """
    Create factory function for environment creation.

    This is needed for vectorized environments.

    Args:
        env_config: Environment configuration
        use_validation: Whether to use validation wrapper

    Returns:
        Callable that creates environment
    """
    def _make_env():
        return create_single_env(env_config, use_validation)

    return _make_env


def create_vectorized_envs(env_config: Dict[str, Any], n_envs: int, use_validation: bool = False):
    """
    Create vectorized environments with OS-appropriate backend.

    OS-conditional logic:
    - Windows: Use DummyVectorEnv (no multiprocessing, avoids pickling issues)
    - Linux: Use SubprocVectorEnv (multiprocessing, faster)

    Args:
        env_config: Environment configuration
        n_envs: Number of parallel environments
        use_validation: Whether to use validation wrapper

    Returns:
        Vectorized environment
    """
    env_factory = create_env_factory(env_config, use_validation)

    current_os = platform.system()
    logger.info(f"Creating {n_envs} vectorized environments on {current_os}")

    if current_os == "Windows":
        # Windows: use DummyVectorEnv to avoid multiprocessing/pickling issues
        logger.info("Using DummyVectorEnv (Windows - no multiprocessing)")
        venv = ts.env.DummyVectorEnv([env_factory for _ in range(n_envs)])
    else:
        # Linux/Mac: use SubprocVectorEnv for better performance
        logger.info("Using SubprocVectorEnv (Linux/Mac - multiprocessing enabled)")
        venv = ts.env.SubprocVectorEnv([env_factory for _ in range(n_envs)])

    return venv


#  OBSERVATION PREPROCESSING 

def preprocess_observations(**kwargs):
    """
    Convert numpy observations to torch tensors for EA mode.

    Handles:
    - Converting numpy arrays to torch tensors
    - Moving tensors to correct device
    - Reshaping masks for EA mode (flatten if needed)
    - Batch processing

    Note: Graph structure is kept in observations for now.
    Can be optimized later by storing graph once in model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_obs(obs_dict):
        """Process single observation dict"""
        if isinstance(obs_dict, dict) and 'graph_nodes' in obs_dict:
            processed = {
                "graph_nodes": torch.from_numpy(obs_dict['graph_nodes']).float().to(device),
                "graph_edge_links": torch.from_numpy(obs_dict['graph_edge_links']).long().to(device),
                "ad_features": torch.from_numpy(obs_dict['ad_features']).float().to(device),
                "mask": torch.from_numpy(obs_dict['mask']).bool().to(device)
            }

            # EA mode: mask should be flattened to (batch, max_ads * n_billboards)
            if len(processed["mask"].shape) == 3:
                batch_size, max_ads, n_billboards = processed["mask"].shape
                processed["mask"] = processed["mask"].reshape(batch_size, -1)

            return processed
        return obs_dict

    # Process observations and next observations
    if "obs" in kwargs:
        kwargs["obs"] = [process_obs(obs) for obs in kwargs["obs"]]
    if "obs_next" in kwargs:
        kwargs["obs_next"] = [process_obs(obs) for obs in kwargs["obs_next"]]

    return kwargs


#  EA-SPECIFIC METRICS 

class EAMetricsLogger:
    """
    Track EA-specific metrics for research validation.

    Metrics tracked:
    - Number of active ads per step
    - Number of valid (ad, billboard) pairs
    - Mean number of selected edges per step
    - Percentage of masked actions
    - Action distribution statistics

    These are critical for:
    - Debugging
    - Sanity checks
    - Reviewer-facing validation
    """

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.step_count = 0

    def log_step_metrics(self, batch, global_step: int):
        """Log metrics for a batch of transitions"""
        if not hasattr(batch, 'obs') or not isinstance(batch.obs, list):
            return

        # Aggregate metrics across batch
        n_active_ads_list = []
        n_valid_pairs_list = []
        mask_ratio_list = []

        for obs in batch.obs:
            if isinstance(obs, dict) and 'mask' in obs:
                mask = obs['mask']

                # Count valid actions
                n_valid = mask.sum().item() if hasattr(mask, 'sum') else np.sum(mask)
                total = mask.numel() if hasattr(mask, 'numel') else mask.size

                n_valid_pairs_list.append(n_valid)
                mask_ratio_list.append(n_valid / total if total > 0 else 0)

        # Log to tensorboard
        if n_valid_pairs_list:
            self.writer.add_scalar('ea/mean_valid_pairs', np.mean(n_valid_pairs_list), global_step)
            self.writer.add_scalar('ea/mean_mask_ratio', np.mean(mask_ratio_list), global_step)

        self.step_count += 1


#  MAIN TRAINING 

def main(use_test_config: bool = True):
    """
    Main training function for EA mode.

    Args:
        use_test_config: If True, use test config. If False, use full config.

    Returns:
        Training result dictionary
    """
    # Select configuration
    config = get_test_config() if use_test_config else get_full_config()
    env_config = config["env"]
    train_config = config["train"]

    # Log configuration
    logger.info("="*60)
    logger.info("EA MODE TRAINING - Billboard Allocation")
    logger.info("="*60)
    logger.info(f"Configuration: {'TEST (correctness validation)' if use_test_config else 'FULL (production)'}")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info(f"OS: {platform.system()}")
    logger.info(f"Action mode: {env_config['action_mode']}")
    logger.info(f"Max events: {env_config['max_events']}")
    logger.info(f"Max active ads: {env_config.get('max_active_ads', 20)}")
    logger.info("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample environment to get dimensions
    logger.info("Creating sample environment to infer dimensions...")
    sample_env = create_single_env(env_config, use_validation=False)
    n_billboards = sample_env.env.n_nodes
    max_ads = sample_env.env.config.max_active_ads
    action_space_size = n_billboards * max_ads

    logger.info(f"Environment dimensions:")
    logger.info(f"  - Billboards: {n_billboards}")
    logger.info(f"  - Max active ads: {max_ads}")
    logger.info(f"  - Action space size: {action_space_size} (combinatorial)")

    # Create vectorized environments
    logger.info(f"Creating {train_config['nr_envs']} training environments...")
    train_envs = create_vectorized_envs(
        env_config,
        n_envs=train_config['nr_envs'],
        use_validation=train_config.get('use_validation', False)
    )

    logger.info("Creating 2 test environments...")
    test_envs = create_vectorized_envs(env_config, n_envs=2, use_validation=False)

    # Create model configuration for EA mode
    model_config = {
        'node_feat_dim': 10,
        'ad_feat_dim': 8,
        'hidden_dim': train_config['hidden_dim'],
        'n_graph_layers': train_config['n_graph_layers'],
        'mode': 'ea',  # CRITICAL: Edge Action mode
        'n_billboards': n_billboards,
        'max_ads': max_ads,
        'use_attention': True,
        'conv_type': 'gat',  # GAT for better pair modeling
        'dropout': 0.15
    }

    # Initialize networks
    logger.info("Creating actor and critic networks...")

    # Create base models
    actor_base = BillboardAllocatorGNN(**model_config).to(device)
    critic_base = BillboardAllocatorGNN(**model_config).to(device)

    # Wrap models to match Tianshou's expected interface
    # Tianshou expects: actor(obs) -> (logits, state) tuple
    # Tianshou expects: critic(obs) -> values (single tensor, NOT tuple)
    class TianshouActorWrapper(torch.nn.Module):
        """Wrapper to make BillboardAllocatorGNN compatible with Tianshou for actor"""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, obs, state=None, info={}):
            """
            Forward pass compatible with Tianshou.

            Args:
                obs: Observations (dict or batch)
                state: Optional recurrent state
                info: Optional info dict (defaults to empty dict)

            Returns:
                logits: Action logits (single tensor)
                state: Updated state (for recurrent models)
            """
            # Handle info parameter - Tianshou might pass None or dict
            if info is None:
                info = {}

            # Call base model
            logits, new_state = self.model(obs, state, info)
            # Return tuple: (logits, state)
            # Tianshou's PPOPolicy will handle this correctly
            return logits, new_state

    class TianshouCriticWrapper(torch.nn.Module):
        """Wrapper to make BillboardAllocatorGNN compatible with Tianshou for critic
        
        CRITICAL: Critic must return a SINGLE tensor, not a tuple!
        Tianshou expects: critic(obs) -> values (tensor)
        """
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, obs, state=None, info={}):
            """
            Forward pass for critic - returns single tensor.

            Args:
                obs: Observations (dict, Batch, or list)
                state: Optional recurrent state (ignored for critic)
                info: Optional info dict (defaults to empty dict)

            Returns:
                values: State values (single tensor, NOT a tuple)
            """
            # Handle Tianshou Batch format - extract observations
            if isinstance(obs, ts.data.Batch):
                obs_dict = obs.obs if hasattr(obs, 'obs') else obs
            else:
                obs_dict = obs
            
            # Handle list of observations - batch them
            if isinstance(obs_dict, list):
                # Convert list of dicts to batched dict
                batched_dict = {}
                for key in obs_dict[0].keys():
                    values = [o[key] for o in obs_dict]
                    if isinstance(values[0], (torch.Tensor, np.ndarray)):
                        if isinstance(values[0], torch.Tensor):
                            batched_dict[key] = torch.stack(values, dim=0)
                        else:
                            batched_dict[key] = np.stack(values, axis=0)
                    else:
                        batched_dict[key] = values[0]
                obs_dict = batched_dict
            
            # Call critic_forward which returns a single tensor
            values = self.model.critic_forward(obs_dict)
            
            # CRITICAL: Return only the tensor, NOT a tuple
            # Ensure it's a tensor (not numpy array)
            if isinstance(values, np.ndarray):
                values = torch.from_numpy(values).float()
            
            return values

    actor = TianshouActorWrapper(actor_base)
    critic = TianshouCriticWrapper(critic_base)

    # Log model parameters
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    total_params = actor_params + critic_params
    logger.info(f"Model parameters:")
    logger.info(f"  - Actor: {actor_params:,}")
    logger.info(f"  - Critic: {critic_params:,}")
    logger.info(f"  - Total: {total_params:,}")

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

    # CRITICAL: Create PPO policy with Independent Bernoulli distribution
    logger.info("Creating PPO policy with Independent Bernoulli distribution...")

    # Get action space from sample environment
    # Note: For MultiBinary action space, action_scaling must be False
    action_space = sample_env.action_space
    logger.info(f"Action space: {action_space}")

    # WORKAROUND: The model currently outputs probabilities (after softmax), not logits
    # For EA mode with IndependentBernoulli, we need logits
    # Create a wrapper that converts probabilities back to logits
    def dist_fn_wrapper(logits_or_probs):
        """
        Wrapper for IndependentBernoulli that handles the model's output.

        The model applies softmax internally (line 751 in models.py).
        For Bernoulli, we need logits, so we convert probs back to logits.

        TODO: Fix model to output logits directly for EA mode
        """
        # If model outputs probabilities (0-1 range), convert to logits
        if torch.all((logits_or_probs >= 0) & (logits_or_probs <= 1)):
            # Clamp to avoid log(0) or log(1)
            probs = torch.clamp(logits_or_probs, 1e-8, 1 - 1e-8)
            logits = torch.log(probs / (1 - probs))  # inverse sigmoid (logit function)
        else:
            logits = logits_or_probs

        # Note: mask handling will be done inside the model
        # For now, create distribution without explicit mask
        return IndependentBernoulli(logits=logits, mask=None)

    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        action_space=action_space,
        dist_fn=dist_fn_wrapper,  # CRITICAL: Use wrapped IndependentBernoulli
        discount_factor=train_config["discount_factor"],
        gae_lambda=train_config["gae_lambda"],
        vf_coef=train_config["vf_coef"],
        ent_coef=train_config["ent_coef"],
        max_grad_norm=train_config["max_grad_norm"],
        eps_clip=train_config["eps_clip"],
        value_clip=True,
        deterministic_eval=True,  # Use deterministic actions during evaluation
        action_scaling=False,  # Must be False for MultiBinary action space
        lr_scheduler=lr_scheduler
    )

    logger.info(f"PPO configuration:")
    logger.info(f"  - Distribution: IndependentBernoulli (EA mode)")
    logger.info(f"  - Entropy coefficient: {train_config['ent_coef']}")
    logger.info(f"  - Learning rate: {train_config['lr']}")
    logger.info(f"  - Batch size: {train_config['batch_size']}")

    # Create collectors
    # Note: preprocess_fn is deprecated in newer Tianshou versions
    # Preprocessing should happen in the model's forward pass
    # For EA mode, observations include large graph structure
    # Use smaller buffer to avoid memory issues
    buffer_size = max(500, train_config["step_per_collect"] * 2)  # Reduced for memory
    logger.info(f"Buffer size: {buffer_size}")

    train_collector = ts.data.Collector(
        policy, train_envs,
        ts.data.VectorReplayBuffer(buffer_size, train_config["nr_envs"]),
        exploration_noise=True
    )

    # Test collector doesn't need a buffer (just for evaluation)
    # But Tianshou creates a huge default buffer, causing memory issues
    # Use a minimal buffer for test
    test_collector = ts.data.Collector(
        policy, test_envs,
        buffer=ts.data.VectorReplayBuffer(100, 2),  # Minimal buffer
        exploration_noise=False
    )

    # Setup logging
    os.makedirs(os.path.dirname(train_config["log_path"]), exist_ok=True)
    writer = SummaryWriter(train_config["log_path"])
    logger_tb = TensorboardLogger(writer)

    # EA-specific metrics logger
    ea_metrics = EAMetricsLogger(writer)

    # Save function
    os.makedirs(os.path.dirname(train_config["save_path"]), exist_ok=True)
    best_reward = -float('inf')

    def save_best_fn(policy):
        nonlocal best_reward
        test_result = test_collector.collect(n_episode=10)

        # New Tianshou API: test_result is CollectStats object
        # Access mean reward from returns_stat
        if test_result.returns_stat is not None:
            current_reward = test_result.returns_stat.mean
        elif len(test_result.returns) > 0:
            current_reward = float(test_result.returns.mean())
        else:
            current_reward = -float('inf')

        if current_reward > best_reward:
            best_reward = current_reward
            logger.info(f"New best reward: {best_reward:.2f}, saving model...")
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'train_config': train_config,
                'env_config': env_config,
                'best_reward': best_reward,
                'mode': 'ea',
                'distribution': 'IndependentBernoulli'
            }, train_config["save_path"])

    # Train
    logger.info("="*60)
    logger.info("Starting EA mode training...")
    logger.info(f"Epochs: {train_config['max_epoch']}")
    logger.info(f"Steps per epoch: {train_config['step_per_epoch']}")
    logger.info(f"Steps per collect: {train_config['step_per_collect']}")
    logger.info(f"Batch size: {train_config['batch_size']}")
    logger.info("="*60)

    # Debug: Test policy before training
    logger.info("Testing policy forward pass...")
    try:
        # Get a test observation
        test_obs_raw, test_info_raw = sample_env.reset()

        # Create a batch (Tianshou format) - must include info
        test_batch = ts.data.Batch(obs=[test_obs_raw], info=[test_info_raw])

        # Call policy
        logger.info(f"Test batch type: {type(test_batch)}")
        logger.info(f"Test obs type: {type(test_obs_raw)}")
        logger.info(f"Test info type: {type(test_info_raw)}")

        result_batch = policy(test_batch)
        logger.info(f"Policy result type: {type(result_batch)}")
        logger.info(f"Result keys: {result_batch.keys() if hasattr(result_batch, 'keys') else 'No keys'}")

        if hasattr(result_batch, 'act'):
            logger.info(f"Act type: {type(result_batch.act)}")
            logger.info(f"Act value: {result_batch.act}")
            logger.info(f"Is callable: {callable(result_batch.act)}")
            if hasattr(result_batch.act, 'shape'):
                logger.info(f"Act shape: {result_batch.act.shape}")

        logger.info("✓ Policy test passed")
    except Exception as e:
        logger.error(f"✗ Policy test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Create trainer (new API)
    trainer = ts.trainer.OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
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

    # Run training
    result = trainer.run()

    # Save final model
    final_path = train_config["save_path"].replace('.pt', '_final.pt')
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'model_config': model_config,
        'train_config': train_config,
        'env_config': env_config,
        'final_reward': result.best_reward,
        'mode': 'ea',
        'distribution': 'IndependentBernoulli'
    }, final_path)

    # Print summary
    logger.info("="*60)
    logger.info('EA mode training complete!')
    logger.info(f'Duration: {result.timing.duration:.2f}s')
    logger.info(f'Best reward: {result.best_reward:.2f} ± {result.best_reward_std:.2f}')
    logger.info(f'Best score: {result.best_score:.2f}')
    logger.info(f'Total gradient steps: {result.gradient_step}')
    logger.info(f'Best model saved to: {train_config["save_path"]}')
    logger.info(f'Final model saved to: {final_path}')
    logger.info("="*60)

    # Clean up
    train_envs.close()
    test_envs.close()
    writer.close()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train EA mode PPO agent')
    parser.add_argument('--full', action='store_true',
                       help='Use full config instead of test config')
    args = parser.parse_args()

    # Run training
    use_test = not args.full
    result = main(use_test_config=use_test)
