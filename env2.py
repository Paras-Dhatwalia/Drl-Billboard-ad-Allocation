# IMPROVED DYNABILLBOARD ENVIRONMENT - FIXED VERSION
from __future__ import annotations
import math
import random
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper functions ---
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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def create_billboard_graph(billboards: List['Billboard'], max_distance: float = 5000.0) -> np.ndarray:
    """Create adjacency matrix for billboards based on distance."""
    n = len(billboards)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(
                billboards[i].latitude, billboards[i].longitude,
                billboards[j].latitude, billboards[j].longitude
            )
            if dist <= max_distance:
                edges.extend([[i, j], [j, i]])  # Bidirectional edges

    if not edges:
        # If no edges, create self-loops
        edges = [[i, i] for i in range(n)]

    return np.array(edges).T if edges else np.array([[0], [0]])


# --- Data classes ---
class Ad:
    """Represents an advertisement with demand and payment attributes."""

    def __init__(self, aid: int, demand: float, payment: float, payment_demand_ratio: float, ttl: int = 15):
        self.aid = aid
        self.demand = float(demand)
        self.payment = float(payment)
        self.payment_demand_ratio = float(payment_demand_ratio)
        self.ttl = ttl
        self.original_ttl = ttl
        self.state = 0  # 0: ongoing, 1: finished, 2: tardy/expired
        self.assigned_billboards: List[int] = []
        self.time_active = 0
        self.cumulative_influence = 0.0
        self.spawn_step: Optional[int] = None

    def step_time(self):
        """Tick TTL and mark tardy if TTL expires while still ongoing."""
        if self.state == 0:
            self.time_active += 1
            self.ttl -= 1
            if self.ttl <= 0:
                self.state = 2  # tardy / failed

    def assign_billboard(self, b_id: int):
        """Assign a billboard to this ad."""
        if b_id not in self.assigned_billboards:
            self.assigned_billboards.append(b_id)

    def release_billboard(self, b_id: int):
        """Release a billboard from this ad."""
        if b_id in self.assigned_billboards:
            self.assigned_billboards.remove(b_id)

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
            self.ttl / max(1, self.original_ttl),  # normalized TTL
            self.cumulative_influence,
            len(self.assigned_billboards),
            1.0 if self.state == 0 else 0.0,  # is_active
        ], dtype=np.float32)


class Billboard:
    """Represents a billboard with location and properties."""

    def __init__(self, b_id: int, lat: float, lon: float, tags: str, b_size: float,
                 b_cost: float, influence: float):
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


# --- Environment ---
class BillboardEnv(AECEnv):
    """
    Dynamic Billboard Allocation Environment using PettingZoo framework.
    """

    metadata = {"render_modes": ["human"], "name": "newenv_fixed"}

    def __init__(self, billboard_csv: str, advertiser_csv: str, trajectory_csv: str,
                 action_mode: str = "na", max_active_ads: int = 20,
                 new_ads_per_step_range: Tuple[int, int] = (1, 5),
                 slot_duration_range: Tuple[int, int] = (1, 5),
                 influence_radius_meters: float = 500.0,
                 tardiness_cost: float = 50.0, max_events: int = 1000,
                 start_time_min: Optional[int] = None,
                 graph_connection_distance: float = 5000.0,
                 seed: Optional[int] = None, debug: bool = False):

        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.debug = debug
        self.action_mode = action_mode.lower()

        if self.action_mode not in ['na', 'ea', 'mh']:
            raise ValueError(f"Unsupported action_mode: {action_mode}. Use 'na', 'ea', or 'mh'")

        logger.info(f"Initializing BillboardEnv with action_mode={self.action_mode}")

        self.max_active_ads = max_active_ads
        self._load_data(billboard_csv, advertiser_csv, trajectory_csv, start_time_min)
        self.max_billboard_size = max((b.b_size for b in self.billboards), default=1.0)
        self.new_ads_per_step_range = new_ads_per_step_range
        self.slot_duration_range = slot_duration_range
        self.influence_radius_meters = float(influence_radius_meters)
        self.tardiness_cost = tardiness_cost
        self.max_events = int(max_events)
        self.graph_connection_distance = graph_connection_distance
        self.edge_index = create_billboard_graph(self.billboards, graph_connection_distance)
        logger.info(f"Created graph with {self.edge_index.shape[1]} edges")
        self.possible_agents = ["Allocator_0"]
        self._setup_action_observation_spaces()
        self._initialize_state()

    def _load_data(self, billboard_csv: str, advertiser_csv: str, trajectory_csv: str, start_time_min: Optional[int]):
        """Load and preprocess all data files."""
        bb_df = pd.read_csv(billboard_csv)
        uniq_df = bb_df.drop_duplicates(subset=['B_id'], keep='first')
        self.billboards: List[Billboard] = [
            Billboard(
                int(r['B_id']), float(r['Latitude']), float(r['Longitude']),
                r.get('Tags', ''), float(r['B_Size']), float(r['B_Cost']),
                float(r['Influence'])
            ) for _, r in uniq_df.iterrows()
        ]
        max_size = max((b.b_size for b in self.billboards), default=1.0)
        for b in self.billboards:
            b.p_size = (b.b_size / max_size) if max_size > 0 else 0.0
        self.n_nodes = len(self.billboards)
        self.billboard_map = {b.b_id: b for b in self.billboards}
        self.billboard_id_to_node_idx = {b.b_id: i for i, b in enumerate(self.billboards)}

        adv_df = pd.read_csv(advertiser_csv)
        self.ads_db: List[Ad] = [
            Ad(int(r['Id']), float(r['Demand']), float(r['Payment']), float(r['Payment_Demand_Ratio']), ttl=15)
            for _, r in adv_df.iterrows()
        ]
        traj_df = pd.read_csv(trajectory_csv)
        if 'Time' not in traj_df.columns:
            raise ValueError("Trajectory CSV missing 'Time' column")
        traj_df['t_min'] = traj_df['Time'].apply(time_str_to_minutes)
        self.start_time_min = int(start_time_min if start_time_min is not None else traj_df['t_min'].min())
        self.trajectory_map = self._preprocess_trajectories(traj_df)
        logger.info(f"Loaded {self.n_nodes} billboards, {len(self.ads_db)} ad templates, and {len(self.trajectory_map)} time points")

    def _preprocess_trajectories(self, df: pd.DataFrame) -> Dict[int, List[Tuple[float, float]]]:
        """Preprocess trajectory data for efficient lookup."""
        return {
            int(t_min): list(zip(grp['Latitude'].astype(float), grp['Longitude'].astype(float)))
            for t_min, grp in df.groupby('t_min')
        }

    def _setup_action_observation_spaces(self):
        """Setup action and observation spaces based on action mode."""
        self.n_node_features = 10
        self.n_ad_features = 8
        
        obs_space_common = {
            'graph_nodes': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_nodes, self.n_node_features), dtype=np.float32),
            'graph_edge_links': spaces.Box(low=0, high=self.n_nodes - 1, shape=(2, self.edge_index.shape[1]), dtype=np.int64),
        }

        if self.action_mode == 'na':
            self.action_spaces = {'Allocator_0': spaces.MultiBinary(self.n_nodes)}
            self.observation_spaces = {'Allocator_0': spaces.Dict({
                **obs_space_common,
                'mask': spaces.MultiBinary(self.n_nodes),
                'current_ad': spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_ad_features,), dtype=np.float32)
            })}
        # FIX: Both 'ea' and 'mh' modes represent selecting ad-billboard pairs. 
        # Using a consistent 2D action space for both simplifies logic for agents.
        elif self.action_mode in ['ea', 'mh']:
            action_shape = (self.max_active_ads, self.n_nodes)
            self.action_spaces = {'Allocator_0': spaces.MultiBinary(action_shape)}
            self.observation_spaces = {'Allocator_0': spaces.Dict({
                **obs_space_common,
                'ad_features': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_active_ads, self.n_ad_features), dtype=np.float32),
                'mask': spaces.MultiBinary(action_shape)
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
        self.performance_metrics = {
            'total_ads_processed': 0, 'total_ads_completed': 0, 'total_ads_tardy': 0,
            'total_revenue': 0.0, 'billboard_utilization': 0.0
        }
        # FIX: Store the ad selected for the 'na' mode observation to ensure action is applied consistently.
        self.current_ad_for_na_mode: Optional[Ad] = None

    def distance_factor(self, dist_meters: float) -> float:
        """Distance effect on billboard influence with 10m cap at 0.9."""
        if dist_meters <= 10.0: return 0.9
        return max(0.1, 1.0 - (dist_meters / self.influence_radius_meters))

    def get_mask(self) -> np.ndarray:
        """Get action mask based on current action mode."""
        if self.action_mode == 'na':
            return np.array([1 if b.is_free() else 0 for b in self.billboards], dtype=np.int8)

        elif self.action_mode in ['ea', 'mh']:
            active_ads = [ad for ad in self.ads if ad.state == 0]
            mask = np.zeros((self.max_active_ads, self.n_nodes), dtype=np.int8)
            free_billboard_mask = np.array([1 if b.is_free() else 0 for b in self.billboards], dtype=np.int8)
            num_active = min(len(active_ads), self.max_active_ads)
            if num_active > 0:
                mask[:num_active, :] = free_billboard_mask
            return mask
        
        return np.array([1], dtype=np.int8)

    def _get_obs(self) -> Dict[str, Any]:
        """Get current observation."""
        nodes = np.array([b.get_feature_vector() for b in self.billboards], dtype=np.float32)
        obs = {'graph_nodes': nodes, 'graph_edge_links': self.edge_index.copy(), 'mask': self.get_mask()}

        if self.action_mode in ['ea', 'mh']:
            ad_features = np.zeros((self.max_active_ads, self.n_ad_features), dtype=np.float32)
            active_ads = [ad for ad in self.ads if ad.state == 0]
            for i, ad in enumerate(active_ads[:self.max_active_ads]):
                ad_features[i] = ad.get_feature_vector()
            obs['ad_features'] = ad_features

        elif self.action_mode == 'na':
            active_ads = [ad for ad in self.ads if ad.state == 0]
            if active_ads:
                # FIX: Consistently select and store the ad for the current step.
                self.current_ad_for_na_mode = random.choice(active_ads)
                obs['current_ad'] = self.current_ad_for_na_mode.get_feature_vector()
            else:
                self.current_ad_for_na_mode = None
                obs['current_ad'] = np.zeros(self.n_ad_features, dtype=np.float32)
        return obs

    def _calculate_influence_for_ad_set(self, ad: Ad) -> float:
        """Calculate influence using exact problem formulation."""
        bbs = [self.billboard_map[b_id] for b_id in ad.assigned_billboards if b_id in self.billboard_map]
        if not bbs: return 0.0
        minute_key = (self.start_time_min + self.current_step) % 1440
        user_locs = self.trajectory_map.get(minute_key, [])
        if not user_locs: return 0.0
        total_influence = 0.0
        for u_lat, u_lon in user_locs:
            prob_no_influence = 1.0
            for b in bbs:
                dist_meters = haversine_distance(b.latitude, b.longitude, u_lat, u_lon)
                if dist_meters <= self.influence_radius_meters:
                    pr_b_j_u_i = (b.b_size / max(1e-9, self.max_billboard_size)) * self.distance_factor(dist_meters)
                    pr_b_j_u_i = max(0.0, min(0.999999, pr_b_j_u_i))
                    prob_no_influence *= (1.0 - pr_b_j_u_i)
            total_influence += (1.0 - prob_no_influence)
        return total_influence

    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        total_cost = sum(b.b_cost for b in self.billboards if not b.is_free())
        tardiness_penalty = sum(1 for ad in self.ads if ad.state == 2) * self.tardiness_cost
        ongoing_revenue = 0.0
        for ad in self.ads:
            if ad.state == 0 and ad.assigned_billboards:
                progress_ratio = min(1.0, ad.cumulative_influence / max(ad.demand, 1e-6))
                ongoing_revenue += ad.payment * progress_ratio * 0.1
        completion_bonus = sum(ad.payment for ad in self.ads if ad.state == 1) * 0.5
        utilization_bonus = (sum(1 for b in self.billboards if not b.is_free()) / max(1, self.n_nodes)) * 10.0
        return ongoing_revenue + completion_bonus + utilization_bonus - total_cost - tardiness_penalty

    def _apply_influence_for_current_minute(self):
        """Apply influence and complete ads if demand is met."""
        for ad in list(self.ads):
            if ad.state == 0:
                delta = self._calculate_influence_for_ad_set(ad)
                ad.cumulative_influence += delta
                if ad.cumulative_influence >= ad.demand:
                    ad.state = 1
                    self.performance_metrics['total_ads_completed'] += 1
                    for b_id in ad.assigned_billboards[:]:
                        if b_id in self.billboard_map:
                            billboard = self.billboard_map[b_id]
                            billboard.revenue_generated += ad.payment / max(1, len(ad.assigned_billboards))
                            billboard.release()
                        ad.release_billboard(b_id)

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

    def _spawn_ads(self):
        """Spawn new ads based on configuration."""
        self.ads = [ad for ad in self.ads if ad.state == 0]
        n_spawn = random.randint(*self.new_ads_per_step_range)
        current_ad_ids = {ad.aid for ad in self.ads}
        available_templates = [a for a in self.ads_db if a.aid not in current_ad_ids]
        spawn_count = min(self.max_active_ads - len(self.ads), n_spawn, len(available_templates))
        if spawn_count > 0:
            selected_templates = random.sample(available_templates, spawn_count)
            for template in selected_templates:
                new_ad = Ad(template.aid, template.demand, template.payment, template.payment_demand_ratio, template.ttl)
                new_ad.spawn_step = self.current_step
                self.ads.append(new_ad)
                self.performance_metrics['total_ads_processed'] += 1

    def _execute_action(self, action):
        """Execute the selected action based on action mode."""
        if action is None: return

        if self.action_mode == 'na':
            # FIX: Use the ad that was selected when the observation was created.
            ad_to_assign = self.current_ad_for_na_mode
            if ad_to_assign and isinstance(action, (list, np.ndarray)):
                for bb_idx, chosen in enumerate(action):
                    if chosen == 1 and self.billboards[bb_idx].is_free():
                        duration = random.randint(*self.slot_duration_range)
                        billboard = self.billboards[bb_idx]
                        billboard.assign(ad_to_assign.aid, duration)
                        ad_to_assign.assign_billboard(billboard.b_id)
                        self.placement_history.append({
                            'spawn_step': ad_to_assign.spawn_step, 'allocated_step': self.current_step,
                            'ad_id': ad_to_assign.aid, 'billboard_id': billboard.b_id,
                            'duration': duration, 'demand': ad_to_assign.demand
                        })

        # FIX: Unified logic for 'ea' and 'mh' as they now share the same 2D action space.
        elif self.action_mode in ['ea', 'mh']:
            if isinstance(action, (list, np.ndarray)):
                action = np.array(action)
                active_ads = [ad for ad in self.ads if ad.state == 0]
                for ad_idx in range(min(len(active_ads), self.max_active_ads)):
                    for bb_idx in range(self.n_nodes):
                        if action.shape == (self.max_active_ads, self.n_nodes) and action[ad_idx, bb_idx] == 1 and self.billboards[bb_idx].is_free():
                            ad_to_assign = active_ads[ad_idx]
                            duration = random.randint(*self.slot_duration_range)
                            billboard = self.billboards[bb_idx]
                            billboard.assign(ad_to_assign.aid, duration)
                            ad_to_assign.assign_billboard(billboard.b_id)
                            self.placement_history.append({
                                'spawn_step': ad_to_assign.spawn_step, 'allocated_step': self.current_step,
                                'ad_id': ad_to_assign.aid, 'billboard_id': billboard.b_id,
                                'duration': duration, 'demand': ad_to_assign.demand
                            })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.current_step = 0
        self.ads.clear()
        for b in self.billboards:
            b.release()
            b.total_usage = 0
            b.revenue_generated = 0.0
        self.agents = self.possible_agents.copy()
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.placement_history.clear()
        self.performance_metrics = {
            'total_ads_processed': 0, 'total_ads_completed': 0, 'total_ads_tardy': 0,
            'total_revenue': 0.0, 'billboard_utilization': 0.0
        }
        self._spawn_ads()
        return self.observe(self.agent_selection), {}

    def observe(self, agent: str) -> Dict[str, Any]:
        """Get observation for specified agent."""
        return self._get_obs()

    def step(self, action):
        """Execute one environment step for the current agent."""
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            return
        
        # --- This block executes only once per environment step ---
        if self._agent_selector.is_first():
            self._apply_influence_for_current_minute()
            self._tick_and_release_boards()
            for ad in self.ads:
                prev_state = ad.state
                ad.step_time()
                if ad.state == 2 and prev_state != 2:
                    self.performance_metrics['total_ads_tardy'] += 1
            
            self.current_step += 1
            is_terminal = (self.current_step >= self.max_events)
            for ag in self.agents:
                self.terminations[ag] = is_terminal
        # --- End of once-per-step block ---

        self._execute_action(action)

        if self._agent_selector.is_last():
            reward = self._compute_reward()
            for a in self.agents:
                self.rewards[a] = reward
                self._cumulative_rewards[a] += reward

            self.performance_metrics['total_revenue'] = sum(b.revenue_generated for b in self.billboards)
            occupied_count = sum(1 for b in self.billboards if not b.is_free())
            self.performance_metrics['billboard_utilization'] = occupied_count / max(1, self.n_nodes)
            
            self._spawn_ads()

            for a in self.agents:
                self.infos[a] = {
                    'performance_metrics': self.performance_metrics.copy(),
                    'current_minute': (self.start_time_min + self.current_step) % 1440
                }
        
        self.agent_selection = self._agent_selector.next()

    def render(self, mode="human"):
        """Render current environment state."""
        minute = (self.start_time_min + self.current_step) % 1440
        print(f"\n--- Step {self.current_step} | Time: {minute//60:02d}:{minute%60:02d} | Reward: {self.rewards.get('Allocator_0', 0.0):.2f} ---")
        occupied = [b for b in self.billboards if not b.is_free()]
        print(f"\nOccupied Billboards ({len(occupied)}/{self.n_nodes}):")
        if not occupied: print("  None")
        else:
            for b in occupied[:10]:
                print(f"  BB_ID {b.b_id}: Ad {b.current_ad}, Time Left: {b.occupied_until}, Cost: {b.b_cost:.2f}")

    def close(self):
        """Clean up environment."""
        pass