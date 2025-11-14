# OPTIMIZED DYNABILLBOARD ENVIRONMENT
from __future__ import annotations
import math
import random
import logging
import time
from functools import wraps
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
import pandas as pd
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
@dataclass
class EnvConfig:
    """Environment configuration parameters"""
    influence_radius_meters: float = 500.0
    slot_duration_range: Tuple[int, int] = (1, 5)
    new_ads_per_step_range: Tuple[int, int] = (1, 5)
    tardiness_cost: float = 50.0
    max_events: int = 1000
    max_active_ads: int = 20
    graph_connection_distance: float = 5000.0
    cache_ttl: int = 1  # Cache TTL in steps
    enable_profiling: bool = False
    debug: bool = False

# ==================== PERFORMANCE MONITORING ====================
class PerformanceMonitor:
    """Track performance metrics and timing"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.step_times = []
        self.influence_calc_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.step_count = 0
    
    def time_function(self, category='general'):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                
                if category == 'step':
                    self.step_times.append(elapsed)
                elif category == 'influence':
                    self.influence_calc_times.append(elapsed)
                
                if self.step_count % 100 == 0 and self.step_count > 0:
                    self.print_stats()
                    
                return result
            return wrapper
        return decorator
    
    def record_cache_hit(self):
        self.cache_hits += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
    
    def print_stats(self):
        """Print performance statistics"""
        if self.step_times:
            avg_step = np.mean(self.step_times)
            logger.info(f"Avg step time: {avg_step:.2f}ms")
        
        if self.influence_calc_times:
            avg_influence = np.mean(self.influence_calc_times)
            logger.info(f"Avg influence calc time: {avg_influence:.2f}ms")
        
        if self.cache_hits + self.cache_misses > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
            logger.info(f"Cache hit rate: {hit_rate:.2%}")

# ==================== HELPER FUNCTIONS ====================
def time_str_to_minutes(v: Any) -> int:
    """Convert time string to minutes since midnight."""
    if isinstance(v, str) and ":" in v:
        try:
            hh, mm = v.split(":")[:2]
            return int(hh) * 60 + int(mm)
        except Exception as e:
            logger.warning(f"Could not parse time string {v}: {e}")
            return 0
    try:
        return int(v)
    except Exception:
        return 0

def haversine_distance_vectorized(lat1: np.ndarray, lon1: np.ndarray, 
                                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance calculation between points in meters."""
    R = 6371000.0  # Earth radius in meters
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def validate_csv(df: pd.DataFrame, required_columns: List[str], csv_name: str):
    """Validate that CSV has required columns"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_name} missing required columns: {missing}")

# ==================== DATA CLASSES ====================
class Ad:
    """Represents an advertisement with demand and payment attributes."""
    
    def __init__(self, aid: int, demand: float, payment: float, 
                 payment_demand_ratio: float, ttl: int = 15):
        self.aid = aid
        self.demand = float(demand)
        self.payment = float(payment)
        self.payment_demand_ratio = float(payment_demand_ratio)
        self.ttl = ttl
        self.original_ttl = ttl
        self.state = 0  # 0: ongoing, 1: finished, 2: tardy/expired
        self.assigned_billboards: Set[int] = set()  # Use set for O(1) operations
        self.time_active = 0
        self.cumulative_influence = 0.0
        self.spawn_step: Optional[int] = None
        self._cached_influence: Optional[float] = None
        self._cache_step: Optional[int] = None

    def step_time(self):
        """Tick TTL and mark tardy if TTL expires while still ongoing."""
        if self.state == 0:
            self.time_active += 1
            self.ttl -= 1
            if self.ttl <= 0:
                self.state = 2  # tardy / failed

    def assign_billboard(self, b_id: int):
        """Assign a billboard to this ad."""
        self.assigned_billboards.add(b_id)
        self._cached_influence = None  # Invalidate cache

    def release_billboard(self, b_id: int):
        """Release a billboard from this ad."""
        self.assigned_billboards.discard(b_id)
        self._cached_influence = None  # Invalidate cache

    def norm_payment_ratio(self) -> float:
        """Normalized payment ratio using sigmoid function."""
        return 1.0 / (1.0 + math.exp(-(self.payment_demand_ratio - 1.0)))

    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for this ad."""
        return np.array([
            self.demand,
            self.payment,
            self.payment_demand_ratio,
            self.norm_payment_ratio(),
            self.ttl / max(1, self.original_ttl),
            self.cumulative_influence,
            len(self.assigned_billboards),
            1.0 if self.state == 0 else 0.0,
        ], dtype=np.float32)


class Billboard:
    """Represents a billboard with location and properties."""
    
    def __init__(self, b_id: int, lat: float, lon: float, tags: str, 
                 b_size: float, b_cost: float, influence: float):
        self.b_id = b_id
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.tags = tags if pd.notna(tags) else ""
        self.b_size = float(b_size)
        self.b_cost = float(b_cost)
        self.influence = float(influence)
        self.occupied_until = 0
        self.current_ad: Optional[int] = None
        self.p_size = 0.0  # normalized size
        self.total_usage = 0
        self.revenue_generated = 0.0

    def is_free(self) -> bool:
        """Check if billboard is available."""
        return self.occupied_until <= 0

    def assign(self, ad_id: int, duration: int):
        """Assign an ad to this billboard for a duration."""
        self.current_ad = ad_id
        self.occupied_until = max(1, int(duration))
        self.total_usage += 1

    def release(self) -> Optional[int]:
        """Release current ad from billboard."""
        ad_id = self.current_ad
        self.current_ad = None
        self.occupied_until = 0
        return ad_id

    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for this billboard."""
        return np.array([
            1.0,  # node type (billboard)
            0.0 if self.is_free() else 1.0,  # is_occupied
            self.b_cost,
            self.b_size,
            self.influence,
            self.p_size,
            self.occupied_until,
            self.total_usage,
            self.latitude / 90.0,  # normalized latitude
            self.longitude / 180.0,  # normalized longitude
        ], dtype=np.float32)


# ==================== OPTIMIZED ENVIRONMENT ====================
class OptimizedBillboardEnv(AECEnv):
    """
    Optimized Dynamic Billboard Allocation Environment with vectorized operations.
    
    Key optimizations:
    - Vectorized influence calculations using NumPy broadcasting
    - Cached per-minute billboard probabilities
    - Precomputed billboard size ratios
    - Efficient trajectory storage as NumPy arrays
    - Performance monitoring and profiling
    """
    
    metadata = {"render_modes": ["human"], "name": "optimized_billboard_env"}
    
    def __init__(self, billboard_csv: str, advertiser_csv: str, trajectory_csv: str,
                 action_mode: str = "na", config: Optional[EnvConfig] = None,
                 start_time_min: Optional[int] = None, seed: Optional[int] = None):
        
        super().__init__()
        
        # Use provided config or default
        self.config = config or EnvConfig()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.action_mode = action_mode.lower()
        
        if self.action_mode not in ['na', 'ea', 'mh']:
            raise ValueError(f"Unsupported action_mode: {action_mode}. Use 'na', 'ea', or 'mh'")
        
        logger.info(f"Initializing OptimizedBillboardEnv with action_mode={self.action_mode}")
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor() if self.config.enable_profiling else None
        
        self.influence_cache: Dict[Tuple[int, frozenset], Tuple[float, int]] = {}

        # Load and process data
        self._load_data(billboard_csv, advertiser_csv, trajectory_csv, start_time_min)
        
        # Precompute billboard properties
        self._precompute_billboard_properties()
        
        # Create graph structure
        self.edge_index = self._create_billboard_graph()
        logger.info(f"Created graph with {self.edge_index.shape[1]} edges")
        
        # PettingZoo setup
        self.possible_agents = ["Allocator_0"]
        self._setup_action_observation_spaces()
        
        # Runtime state
        self._initialize_state()
        
        # Cache for influence calculations
        self.influence_cache: Dict[Tuple[int, frozenset], Tuple[float, int]] = {}
    
    def _load_data(self, billboard_csv: str, advertiser_csv: str, 
                   trajectory_csv: str, start_time_min: Optional[int]):
        """Load and preprocess all data files with validation."""
        
        # Load and validate billboard data
        bb_df = pd.read_csv(billboard_csv)
        validate_csv(bb_df, ['B_id', 'Latitude', 'Longitude', 'B_Size', 'B_Cost', 'Influence'], 
                    "Billboard CSV")
        logger.info(f"Loaded {len(bb_df)} billboard entries")
        
        # Get unique billboards
        uniq_df = bb_df.drop_duplicates(subset=['B_id'], keep='first')
        logger.info(f"Found {len(uniq_df)} unique billboards")
        
        # Create billboard objects
        self.billboards: List[Billboard] = []
        for _, r in uniq_df.iterrows():
            self.billboards.append(Billboard(
                int(r['B_id']), float(r['Latitude']), float(r['Longitude']),
                r.get('Tags', ''), float(r['B_Size']), float(r['B_Cost']),
                float(r['Influence'])
            ))
        
        self.n_nodes = len(self.billboards)
        self.billboard_map = {b.b_id: b for b in self.billboards}
        self.billboard_id_to_node_idx = {b.b_id: i for i, b in enumerate(self.billboards)}
        
        # Load and validate advertiser data
        adv_df = pd.read_csv(advertiser_csv)
        adv_df.columns = adv_df.columns.str.strip().str.replace('\ufeff', '')
        validate_csv(adv_df, ['Id', 'Demand', 'Payment', 'Payment_Demand_Ratio'],
                    "Advertiser CSV")
        logger.info(f"Loaded {len(adv_df)} advertiser templates")
        
        self.ads_db: List[Ad] = []
        for aid, demand, payment, ratio in zip(
            adv_df['Id'].values,
            adv_df['Demand'].values,
            adv_df['Payment'].values,
            adv_df['Payment_Demand_Ratio'].values):
            
            self.ads_db.append(
                Ad(aid=int(aid), demand=float(demand), payment=float(payment),
                   payment_demand_ratio=float(ratio), ttl=15)
            )
        
        # Load and validate trajectory data
        traj_df = pd.read_csv(trajectory_csv)
        validate_csv(traj_df, ['Time', 'Latitude', 'Longitude'], "Trajectory CSV")
        
        traj_df['t_min'] = traj_df['Time'].apply(time_str_to_minutes)
        self.start_time_min = int(start_time_min if start_time_min is not None 
                                 else traj_df['t_min'].min())
        
        # Preprocess trajectories as NumPy arrays for efficient operations
        self.trajectory_map = self._preprocess_trajectories_optimized(traj_df)
        logger.info(f"Processed trajectories for {len(self.trajectory_map)} time points")
    
    def _preprocess_trajectories_optimized(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Preprocess trajectory data as NumPy arrays for vectorized operations."""
        traj_map: Dict[int, np.ndarray] = {}
        
        for t_min, grp in df.groupby('t_min'):
            # Store as float32 NumPy array for efficiency
            coords = np.column_stack([
                grp['Latitude'].values.astype(np.float32),
                grp['Longitude'].values.astype(np.float32)
            ])
            traj_map[int(t_min)] = coords
        
        return traj_map
    
    def _precompute_billboard_properties(self):
        """Precompute billboard properties for efficiency."""
        # Find max billboard size
        self.max_billboard_size = max((b.b_size for b in self.billboards), default=1.0)
        
        # Precompute normalized sizes
        for b in self.billboards:
            b.p_size = (b.b_size / self.max_billboard_size) if self.max_billboard_size > 0 else 0.0
        
        # Store billboard coordinates as NumPy arrays for vectorized distance calculations
        self.billboard_coords = np.array([
            [b.latitude, b.longitude] for b in self.billboards
        ], dtype=np.float32)
        
        # Precompute size ratios
        self.billboard_size_ratios = np.array([
            b.b_size / self.max_billboard_size for b in self.billboards
        ], dtype=np.float32)
    
    def _create_billboard_graph(self) -> np.ndarray:
        """Create adjacency matrix for billboards using vectorized distance calculation."""
        n = len(self.billboards)
        
        if n == 0:
            return np.array([[0], [0]])
        
        # Vectorized distance calculation
        coords = self.billboard_coords
        lat1 = coords[:, 0:1]  # Shape (n, 1)
        lon1 = coords[:, 1:2]  # Shape (n, 1)
        lat2 = coords[:, 0].reshape(1, -1)  # Shape (1, n)
        lon2 = coords[:, 1].reshape(1, -1)  # Shape (1, n)
        
        # Calculate all pairwise distances at once
        distances = haversine_distance_vectorized(lat1, lon1, lat2, lon2)
        
        # Find edges within threshold
        valid_pairs = np.where((distances <= self.config.graph_connection_distance) & 
                              (distances > 0))
        
        if len(valid_pairs[0]) > 0:
            edges = np.column_stack(valid_pairs)
            # Add reverse edges for bidirectional graph
            edges_reverse = edges[:, [1, 0]]
            all_edges = np.vstack([edges, edges_reverse])
            return all_edges.T
        else:
            # If no edges, create self-loops
            edges = np.array([[i, i] for i in range(n)])
            return edges.T
    
    def _setup_action_observation_spaces(self):
        """Setup action and observation spaces based on action mode."""
        # Node features: billboard properties
        self.n_node_features = 10
        # Ad features: advertisement properties
        self.n_ad_features = 8
        
        if self.action_mode == 'na':
            # Node Action: select billboard (ad chosen by environment)
            self.action_spaces = {'Allocator_0': spaces.Discrete(self.n_nodes)}
            self.observation_spaces = {'Allocator_0': spaces.Dict({
                'graph_nodes': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.n_nodes, self.n_node_features), 
                                         dtype=np.float32),
                'graph_edge_links': spaces.Box(low=0, high=self.n_nodes-1,
                                              shape=(2, self.edge_index.shape[1]), 
                                              dtype=np.int64),
                'mask': spaces.MultiBinary(self.n_nodes),
                'current_ad': spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(self.n_ad_features,), dtype=np.float32)
            })}
        
        elif self.action_mode == 'ea':
            # Edge Action: select ad-billboard pairs
            max_pairs = self.config.max_active_ads * self.n_nodes
            self.action_spaces = {'Allocator_0': spaces.MultiBinary(max_pairs)}
            self.observation_spaces = {'Allocator_0': spaces.Dict({
                'graph_nodes': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.n_nodes, self.n_node_features), 
                                         dtype=np.float32),
                'graph_edge_links': spaces.Box(low=0, high=self.n_nodes-1,
                                              shape=(2, self.edge_index.shape[1]), 
                                              dtype=np.int64),
                'ad_features': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.config.max_active_ads, self.n_ad_features), 
                                         dtype=np.float32),
                'mask': spaces.MultiBinary(max_pairs)
            })}
        
        elif self.action_mode == 'mh':
            # Multi-Head: select ad, then billboard
            self.action_spaces = {'Allocator_0': spaces.MultiBinary(
                [self.config.max_active_ads, self.n_nodes])}
            self.observation_spaces = {'Allocator_0': spaces.Dict({
                'graph_nodes': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.n_nodes, self.n_node_features), 
                                         dtype=np.float32),
                'graph_edge_links': spaces.Box(low=0, high=self.n_nodes-1,
                                              shape=(2, self.edge_index.shape[1]), 
                                              dtype=np.int64),
                'ad_features': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.config.max_active_ads, self.n_ad_features), 
                                         dtype=np.float32),
                'mask': spaces.MultiBinary([self.config.max_active_ads, self.n_nodes])
            })}
    
    def _initialize_state(self):
        """Initialize runtime state variables."""
        self.current_step = 0
        self.ads: List[Ad] = []
        self.agents: List[str] = []
        self.rewards: Dict[str, float] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}
        self._agent_selector = agent_selector([])
        self.agent_selection: str = ""
        self.placement_history: List[Dict[str, Any]] = []
        self.current_ad_for_na_mode: Optional[Ad] = None
        self.performance_metrics = {
            'total_ads_processed': 0,
            'total_ads_completed': 0,
            'total_ads_tardy': 0,
            'total_revenue': 0.0,
            'billboard_utilization': 0.0
        }
        
        # Clear cache
        self.influence_cache.clear()
        
        if self.perf_monitor:
            self.perf_monitor.reset()
    
    def distance_factor(self, dist_meters: np.ndarray) -> np.ndarray:
        """Vectorized distance effect on billboard influence with 10m cap at 0.9."""
        factor = np.ones_like(dist_meters) * 0.9  # Default for distances <= 10m
        mask = dist_meters > 10.0
        factor[mask] = np.maximum(0.1, 1.0 - (dist_meters[mask] / self.config.influence_radius_meters))
        return factor
    
    def get_mask(self) -> np.ndarray:
        """Get action mask based on current action mode with validation."""
        if self.action_mode == 'na':
            # Mask free billboards
            mask = np.array([1 if b.is_free() else 0 for b in self.billboards], dtype=np.int8)
            if mask.sum() == 0:
                logger.warning("No free billboards available for 'na' mode")
            return mask
        
        elif self.action_mode == 'ea':
            # Mask valid ad-billboard pairs (flattened)
            mask = np.zeros(self.config.max_active_ads * self.n_nodes, dtype=np.int8)
            active_ads = [ad for ad in self.ads if ad.state == 0]
            
            for ad_idx in range(min(len(active_ads), self.config.max_active_ads)):
                for bb_idx, billboard in enumerate(self.billboards):
                    if billboard.is_free():
                        pair_idx = ad_idx * self.n_nodes + bb_idx
                        mask[pair_idx] = 1
            
            if mask.sum() == 0:
                logger.warning("No valid ad-billboard pairs for 'ea' mode")
            return mask
        
        elif self.action_mode == 'mh':
            # 2D mask over (ad_idx, bb_idx) pairs
            active_ads = [ad for ad in self.ads if ad.state == 0]
            pair_mask = np.zeros((self.config.max_active_ads, self.n_nodes), dtype=np.int8)
            
            free_cols = np.array([1 if b.is_free() else 0 for b in self.billboards], dtype=np.int8)
            for i in range(min(len(active_ads), self.config.max_active_ads)):
                pair_mask[i, :] = free_cols
            
            if pair_mask.sum() == 0:
                logger.warning("No valid ad-billboard pairs for 'mh' mode")
            return pair_mask
        
        return np.array([1], dtype=np.int8)
    
    def _get_obs(self) -> Dict[str, Any]:
        """Get current observation."""
        # Node features (billboards)
        nodes = np.zeros((self.n_nodes, self.n_node_features), dtype=np.float32)
        for i, b in enumerate(self.billboards):
            nodes[i] = b.get_feature_vector()
        
        obs = {
            'graph_nodes': nodes,
            'graph_edge_links': self.edge_index.copy(),
            'mask': self.get_mask()
        }
        
        # Add ad features for modes that need them
        if self.action_mode in ['ea', 'mh']:
            ad_features = np.zeros((self.config.max_active_ads, self.n_ad_features), 
                                  dtype=np.float32)
            active_ads = [ad for ad in self.ads if ad.state == 0]
            
            for i, ad in enumerate(active_ads[:self.config.max_active_ads]):
                ad_features[i] = ad.get_feature_vector()
            
            obs['ad_features'] = ad_features
        
        # Add current ad for NA mode
        elif self.action_mode == 'na':
            active_ads = [ad for ad in self.ads if ad.state == 0]
            if active_ads:
                self.current_ad_for_na_mode = random.choice(active_ads)
                obs['current_ad'] = self.current_ad_for_na_mode.get_feature_vector()
            else:
                self.current_ad_for_na_mode = None
                obs['current_ad'] = np.zeros(self.n_ad_features, dtype=np.float32)
        
        return obs
    
    def _calculate_influence_for_ad_vectorized(self, ad: Ad) -> float:
        """
        Vectorized influence calculation using NumPy broadcasting.
        
        Implements: I(S) = sum_{u_i in U} [1 - prod_{b_j in S} (1 - Pr(b_j, u_i))]
        """
        # Check cache first
        cache_key = (self.current_step, frozenset(ad.assigned_billboards))
        if cache_key in self.influence_cache:
            cached_value, cached_step = self.influence_cache[cache_key]
            if self.current_step - cached_step <= self.config.cache_ttl:
                if self.perf_monitor:
                    self.perf_monitor.record_cache_hit()
                return cached_value
        
        if self.perf_monitor:
            self.perf_monitor.record_cache_miss()
        
        # S: Set of billboards assigned to this ad
        if not ad.assigned_billboards:
            return 0.0
        
        bb_indices = [self.billboard_id_to_node_idx[b_id] 
                     for b_id in ad.assigned_billboards 
                     if b_id in self.billboard_id_to_node_idx]
        
        if not bb_indices:
            return 0.0
        
        # Get user locations for current time (set U)
        minute_key = (self.start_time_min + self.current_step) % 1440
        user_locs = self.trajectory_map.get(minute_key, np.array([]))
        
        if len(user_locs) == 0:
            return 0.0
        
        # Get billboard coordinates and size ratios for assigned billboards
        bb_coords = self.billboard_coords[bb_indices]  # Shape: (n_billboards, 2)
        bb_size_ratios = self.billboard_size_ratios[bb_indices]  # Shape: (n_billboards,)
        
        # Vectorized distance calculation
        # user_locs shape: (n_users, 2)
        # bb_coords shape: (n_billboards, 2)
        # We need distances between all users and all billboards
        
        user_lats = user_locs[:, 0:1]  # Shape: (n_users, 1)
        user_lons = user_locs[:, 1:2]  # Shape: (n_users, 1)
        bb_lats = bb_coords[:, 0].reshape(1, -1)  # Shape: (1, n_billboards)
        bb_lons = bb_coords[:, 1].reshape(1, -1)  # Shape: (1, n_billboards)
        
        # Calculate distances: shape (n_users, n_billboards)
        distances = haversine_distance_vectorized(user_lats, user_lons, bb_lats, bb_lons)
        
        # Apply radius mask
        within_radius = distances <= self.config.influence_radius_meters
        
        # Calculate probabilities where within radius
        probabilities = np.zeros_like(distances)
        mask = within_radius
        
        if np.any(mask):
            # Base probability from size normalization
            # Broadcasting: (n_users, n_billboards) * (1, n_billboards)
            probabilities[mask] = bb_size_ratios[None, :].repeat(len(user_locs), axis=0)[mask]
            
            # Apply distance decay
            probabilities[mask] *= self.distance_factor(distances[mask])
            
            # Numerical safety clamp
            probabilities = np.clip(probabilities, 0.0, 0.999999)
        
        # Calculate influence for each user
        # prod_{b_j in S} (1 - Pr(b_j, u_i)) for each user
        prob_no_influence = np.prod(1.0 - probabilities, axis=1)  # Shape: (n_users,)
        
        # Total influence: sum of (1 - prob_no_influence) for all users
        total_influence = np.sum(1.0 - prob_no_influence)
        
        # Cache the result
        self.influence_cache[cache_key] = (total_influence, self.current_step)
        
        # Clean old cache entries periodically
        if len(self.influence_cache) > 1000:
            current_step = self.current_step
            self.influence_cache = {
                k: v for k, v in self.influence_cache.items() 
                if current_step - v[1] <= self.config.cache_ttl
            }
        
        return total_influence
    
    def _compute_reward(self) -> float:
        """Compute reward using simplified cost formulation: r_t = -(C_hold + C_tardy + C_move)"""
        # C_hold: Billboard holding costs
        C_hold = sum(b.b_cost for b in self.billboards if not b.is_free())
        
        # C_tardy: Tardiness penalty for failed/expired ads
        C_tardy = sum(1 for ad in self.ads if ad.state == 2) * self.config.tardiness_cost
        
        # C_move: Movement/assignment cost (minimal)
        C_move = 0.0
        
        # Simple negative cost formulation
        return -(C_hold + C_tardy + C_move)
    
    def _apply_influence_for_current_minute(self):
        """Apply influence for current minute using vectorized calculations."""
        if self.config.debug:
            minute = (self.start_time_min + self.current_step) % 1440
            logger.debug(f"Applying influence at step {self.current_step} (minute {minute})")
        
        # Use vectorized influence calculation
        if self.config.enable_profiling and self.perf_monitor:
            @self.perf_monitor.time_function('influence')
            def calculate_influence(ad):
                return self._calculate_influence_for_ad_vectorized(ad)
        else:
            calculate_influence = self._calculate_influence_for_ad_vectorized
        
        for ad in list(self.ads):
            if ad.state != 0:
                continue
            
            # Check if we need to recalculate (lazy evaluation)
            if ad._cached_influence is not None and ad._cache_step == self.current_step:
                delta = ad._cached_influence
            else:
                delta = calculate_influence(ad)
                ad._cached_influence = delta
                ad._cache_step = self.current_step
            
            ad.cumulative_influence += delta
            
            if self.config.debug and delta > 0:
                logger.debug(f"Ad {ad.aid} gained {delta:.4f} influence")
            
            # Complete ad if demand is satisfied
            if ad.cumulative_influence >= ad.demand:
                ad.state = 1  # completed
                self.performance_metrics['total_ads_completed'] += 1
                
                # Release billboards and generate revenue
                for b_id in list(ad.assigned_billboards):
                    if b_id in self.billboard_map:
                        billboard = self.billboard_map[b_id]
                        billboard.revenue_generated += ad.payment / max(1, len(ad.assigned_billboards))
                        billboard.release()
                    ad.release_billboard(b_id)
                
                if self.config.debug:
                    logger.debug(f"Ad {ad.aid} completed with {ad.cumulative_influence:.2f}/{ad.demand} demand")
    
    def _tick_and_release_boards(self):
        """Tick billboard timers and release expired ones."""
        for b in self.billboards:
            if not b.is_free():
                b.occupied_until -= 1
                
                if b.occupied_until <= 0:
                    ad_id = b.release()
                    if ad_id is not None:
                        ad = next((a for a in self.ads if a.aid == ad_id), None)
                        if ad:
                            ad.release_billboard(b.b_id)
                            
                            # Update placement history
                            for rec in self.placement_history:
                                if (rec['ad_id'] == ad.aid and
                                    rec['billboard_id'] == b.b_id and
                                    'fulfilled_by_end' not in rec):
                                    rec['fulfilled_by_end'] = ad.cumulative_influence
                                    break
    
    def _spawn_ads(self):
        """Spawn new ads based on configuration."""
        # Remove completed/tardy ads
        self.ads = [ad for ad in self.ads if ad.state == 0]
        
        # Spawn new ads
        n_spawn = random.randint(*self.config.new_ads_per_step_range)
        current_ad_ids = {ad.aid for ad in self.ads}
        available_templates = [a for a in self.ads_db if a.aid not in current_ad_ids]
        
        spawn_count = min(
            self.config.max_active_ads - len(self.ads),
            n_spawn,
            len(available_templates)
        )
        
        if spawn_count > 0:
            selected_templates = random.sample(available_templates, spawn_count)
            for template in selected_templates:
                new_ad = Ad(
                    template.aid, template.demand, template.payment,
                    template.payment_demand_ratio, template.ttl
                )
                new_ad.spawn_step = self.current_step
                self.ads.append(new_ad)
                self.performance_metrics['total_ads_processed'] += 1
            
            if self.config.debug:
                logger.debug(f"Spawned {spawn_count} new ads")
    
    def _execute_action(self, action):
        """Execute the selected action with validation."""
        try:
            if self.action_mode == 'na':
                # Node Action mode - now accepts single integer
                ad_to_assign = self.current_ad_for_na_mode
                if ad_to_assign and isinstance(action, (int, np.integer)):
                    bb_idx = int(action)
                    
                    # Check if action is valid and billboard is free
                    if 0 <= bb_idx < self.n_nodes and self.billboards[bb_idx].is_free():
                        duration = random.randint(*self.config.slot_duration_range)
                        billboard = self.billboards[bb_idx]
                        billboard.assign(ad_to_assign.aid, duration)
                        ad_to_assign.assign_billboard(billboard.b_id)
                        
                        self.placement_history.append({
                            'spawn_step': ad_to_assign.spawn_step,
                            'allocated_step': self.current_step,
                            'ad_id': ad_to_assign.aid,
                            'billboard_id': billboard.b_id,
                            'duration': duration,
                            'demand': ad_to_assign.demand
                        })
                        
                        if self.config.debug:
                            logger.debug(f"Assigned ad {ad_to_assign.aid} to billboard {billboard.b_id}")
                    
                    elif self.config.debug:
                        if not (0 <= bb_idx < self.n_nodes):
                            logger.warning(f"Invalid action (billboard index) {bb_idx}")
                        elif not self.billboards[bb_idx].is_free():
                            logger.warning(f"Action failed: Billboard {bb_idx} is not free")
                
                elif self.config.debug:
                    if not ad_to_assign:
                        logger.warning("NA action skipped: No ad to assign")
                    else:
                        logger.warning(f"Invalid action type for NA mode: {type(action)}")
            
            elif self.action_mode == 'ea':
                # Edge Action mode
                if isinstance(action, (list, np.ndarray)):
                    action = np.asarray(action).flatten()
                    expected_shape = self.config.max_active_ads * self.n_nodes
                    if action.shape[0] != expected_shape:
                        logger.warning(f"Invalid action shape for 'ea' mode: {action.shape}")
                        return
                    
                    active_ads = [ad for ad in self.ads if ad.state == 0]
                    
                    for pair_idx, chosen in enumerate(action):
                        if chosen == 1:
                            ad_idx = pair_idx // self.n_nodes
                            bb_idx = pair_idx % self.n_nodes
                            
                            if (ad_idx < min(len(active_ads), self.config.max_active_ads) and
                                self.billboards[bb_idx].is_free()):
                                
                                ad_to_assign = active_ads[ad_idx]
                                duration = random.randint(*self.config.slot_duration_range)
                                
                                billboard = self.billboards[bb_idx]
                                billboard.assign(ad_to_assign.aid, duration)
                                ad_to_assign.assign_billboard(billboard.b_id)
                                
                                self.placement_history.append({
                                    'spawn_step': ad_to_assign.spawn_step,
                                    'allocated_step': self.current_step,
                                    'ad_id': ad_to_assign.aid,
                                    'billboard_id': billboard.b_id,
                                    'duration': duration,
                                    'demand': ad_to_assign.demand
                                })
            
            elif self.action_mode == 'mh':
                # Multi-Head mode
                if isinstance(action, (list, np.ndarray)):
                    action = np.asarray(action)
                    if action.shape != (self.config.max_active_ads, self.n_nodes):
                        logger.warning(f"Invalid action shape for 'mh' mode: {action.shape}")
                        return
                    
                    active_ads = [ad for ad in self.ads if ad.state == 0]
                    
                    for ad_idx in range(min(len(active_ads), self.config.max_active_ads)):
                        for bb_idx in range(self.n_nodes):
                            if action[ad_idx, bb_idx] == 1 and self.billboards[bb_idx].is_free():
                                ad_to_assign = active_ads[ad_idx]
                                duration = random.randint(*self.config.slot_duration_range)
                                
                                billboard = self.billboards[bb_idx]
                                billboard.assign(ad_to_assign.aid, duration)
                                ad_to_assign.assign_billboard(billboard.b_id)
                                
                                self.placement_history.append({
                                    'spawn_step': ad_to_assign.spawn_step,
                                    'allocated_step': self.current_step,
                                    'ad_id': ad_to_assign.aid,
                                    'billboard_id': billboard.b_id,
                                    'duration': duration,
                                    'demand': ad_to_assign.demand
                                })
        
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
    
    # --- PettingZoo required methods ---
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.current_step = 0
        self.ads.clear()
        
        # Reset billboards
        for b in self.billboards:
            b.release()
            b.total_usage = 0
            b.revenue_generated = 0.0
        
        # Reset agents
        self.agents = self.possible_agents.copy()
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Reset tracking
        self.placement_history.clear()
        self.performance_metrics = {
            'total_ads_processed': 0,
            'total_ads_completed': 0,
            'total_ads_tardy': 0,
            'total_revenue': 0.0,
            'billboard_utilization': 0.0
        }
        
        # Clear cache
        self.influence_cache.clear()
        
        if self.perf_monitor:
            self.perf_monitor.reset()
        
        # Spawn initial ads
        self._spawn_ads()
        
        logger.info(f"Environment reset with {len(self.ads)} initial ads")
        
        return self._get_obs(), {}
    
    def observe(self, agent: str) -> Dict[str, Any]:
        """Get observation for specified agent."""
        return self._get_obs()
    
    def action_space(self, agent: str):
        """Get action space for specified agent."""
        return self.action_spaces[agent]
    
    def observation_space(self, agent: str):
        """Get observation space for specified agent."""
        return self.observation_spaces[agent]
    
    def step(self, action):
        """Execute one environment step with performance monitoring."""
        if self.config.enable_profiling and self.perf_monitor:
            self.perf_monitor.step_count += 1
            return self._step_with_profiling(action)
        else:
            return self._step_internal(action)
    
    def step(self, action):  # Execute one environment step with performance monitoring
        if self.config.enable_profiling and self.perf_monitor:
            # Use the performance monitor decorator
            @self.perf_monitor.time_function('step')
            def profiled_step():
                return self._step_internal(action)
            return profiled_step()
        else:
            return self._step_internal(action)
    
    
    def _step_internal(self, action):
        """Internal step implementation."""
        agent = self.agent_selection
        
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            return self._get_obs(), self.rewards, self.terminations, self.truncations, self.infos
        
        # 1. Apply influence for current minute
        self._apply_influence_for_current_minute()
        
        # 2. Tick and release expired billboards
        self._tick_and_release_boards()
        
        # 3. Tick ad TTLs
        for ad in self.ads:
            prev_state = ad.state
            ad.step_time()
            if ad.state == 2 and prev_state != 2:  # became tardy
                self.performance_metrics['total_ads_tardy'] += 1
        
        # 4. Execute agent action
        self._execute_action(action)
        
        # 5. Compute rewards
        reward = self._compute_reward()
        for a in self.agents:
            self.rewards[a] = reward
            self._cumulative_rewards[a] += reward
        
        # 6. Update performance metrics
        self.performance_metrics['total_revenue'] = sum(b.revenue_generated for b in self.billboards)
        occupied_count = sum(1 for b in self.billboards if not b.is_free())
        self.performance_metrics['billboard_utilization'] = occupied_count / max(1, self.n_nodes) * 100
        
        # 7. Spawn new ads
        self._spawn_ads()
        
        # 8. Update termination conditions
        self.current_step += 1
        
        # Update infos with comprehensive metrics
        for a in self.agents:
            self.terminations[a] = (self.current_step >= self.config.max_events)
            self.infos[a] = {
                'total_revenue': self.performance_metrics['total_revenue'],
                'utilization': self.performance_metrics['billboard_utilization'],
                'ads_completed': self.performance_metrics['total_ads_completed'],
                'ads_processed': self.performance_metrics['total_ads_processed'],
                'ads_tardy': self.performance_metrics['total_ads_tardy'],
                'current_minute': (self.start_time_min + self.current_step) % 1440
            }
        
        # 9. Advance agent selection
        if not self._agent_selector.is_last():
            self.agent_selection = self._agent_selector.next()
        
        return self._get_obs(), self.rewards, self.terminations, self.truncations, self.infos
    
    def render(self, mode="human"):
        """Render current environment state."""
        minute = (self.start_time_min + self.current_step) % 1440
        print(f"\n--- Step {self.current_step} | Time: {minute//60:02d}:{minute%60:02d} "
              f"| Reward: {self.rewards.get('Allocator_0', 0.0):.2f} ---")
        
        # Show occupied billboards
        occupied = [b for b in self.billboards if not b.is_free()]
        print(f"\nOccupied Billboards ({len(occupied)}/{self.n_nodes}):")
        
        if not occupied:
            print("  None")
        else:
            for b in occupied[:10]:  # Show first 10
                idx = self.billboard_id_to_node_idx[b.b_id]
                print(f"  Node {idx} (ID: {b.b_id}): Ad {b.current_ad}, "
                      f"Time Left: {b.occupied_until}, Cost: {b.b_cost:.2f}")
            if len(occupied) > 10:
                print(f"  ... and {len(occupied) - 10} more")
        
        # Show active ads
        active_with_assignments = [ad for ad in self.ads if ad.assigned_billboards]
        print(f"\nActive Ads with Assignments ({len(active_with_assignments)}):")
        
        if not active_with_assignments:
            print("  None")
        else:
            for ad in active_with_assignments[:10]:
                state_str = ('Ongoing', 'Finished', 'Tardy')[ad.state]
                progress = f"{ad.cumulative_influence:.2f}/{ad.demand:.2f}"
                print(f"  Ad {ad.aid}: Progress={progress}, TTL={ad.ttl}, "
                      f"State={state_str}, Billboards={len(ad.assigned_billboards)}")
            if len(active_with_assignments) > 10:
                print(f"  ... and {len(active_with_assignments) - 10} more")
        
        # Show performance metrics
        metrics = self.performance_metrics
        print(f"\nPerformance Metrics:")
        print(f"  Processed: {metrics['total_ads_processed']}")
        print(f"  Completed: {metrics['total_ads_completed']}")
        print(f"  Tardy: {metrics['total_ads_tardy']}")
        print(f"  Revenue: ${metrics['total_revenue']:.2f}")
        print(f"  Utilization: {metrics['billboard_utilization']:.1f}%")
        if self.current_step >= self.config.max_events:
            self.render_summary()
    
    def render_summary(self):
        """Render final performance summary."""
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE - Final Results")
        print(f"{'='*60}")
        
        metrics = self.performance_metrics
        
        print(f"Total Ads Processed: {metrics['total_ads_processed']}")
        print(f"Successfully Completed: {metrics['total_ads_completed']}")
        print(f"Failed (Tardy): {metrics['total_ads_tardy']}")
        success_rate = (metrics['total_ads_completed'] / max(1, metrics['total_ads_processed'])) * 100.0
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Revenue Generated: ${metrics['total_revenue']:.2f}")
        print(f"Average Billboard Utilization: {metrics['billboard_utilization']:.1f}%")
        print(f"Total Placements: {len(self.placement_history)}")
        
        # Performance stats if profiling enabled
        if self.perf_monitor:
            self.perf_monitor.print_stats()
    
    def close(self):
        """Clean up environment."""
        pass