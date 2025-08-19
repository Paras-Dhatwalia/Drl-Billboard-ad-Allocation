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

    Supports multiple action modes:
    - 'na': Node Action (environment selects ad, agent selects billboard)
    - 'ea': Edge Action (agent selects ad-billboard pairs)
    - 'mh': Multi-Head (agent selects ad, then billboard)
    """

    metadata = {"render_modes": ["human"], "name": "newenv_fixed"}

    def __init__(self, billboard_csv: str, advertiser_csv: str, trajectory_csv: str,
                 action_mode: str = "na", max_active_ads: int = 20,
                 new_ads_per_step_range: Tuple[int, int] = (1, 5),
                 slot_duration_range: Tuple[int, int] = (1, 5),
                 influence_radius_meters: float = 500.0,  # Increased radius
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

        # --- Load and process data ---
        self._load_data(billboard_csv, advertiser_csv, trajectory_csv, start_time_min)

        # Cache max billboard size
        self.max_billboard_size = max((b.b_size for b in self.billboards), default=1.0)

        # Environment parameters
        self.new_ads_per_step_range = new_ads_per_step_range
        self.slot_duration_range = slot_duration_range
        self.influence_radius_meters = float(influence_radius_meters)
        self.tardiness_cost = tardiness_cost
        self.max_events = int(max_events)
        self.graph_connection_distance = graph_connection_distance

        # Create graph structure
        self.edge_index = create_billboard_graph(self.billboards, graph_connection_distance)
        logger.info(f"Created graph with {self.edge_index.shape[1]} edges")

        # PettingZoo setup
        self.possible_agents = ["Allocator_0"]
        self._setup_action_observation_spaces()

        # Runtime state
        self._initialize_state()

    def _load_data(self, billboard_csv: str, advertiser_csv: str, trajectory_csv: str, start_time_min: Optional[int]):
        """Load and preprocess all data files."""

        # Load billboard data
        bb_df = pd.read_csv(billboard_csv)
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

        # Normalize billboard sizes
        max_size = max((b.b_size for b in self.billboards), default=1.0)
        for b in self.billboards:
            b.p_size = (b.b_size / max_size) if max_size > 0 else 0.0

        self.n_nodes = len(self.billboards)
        self.billboard_map = {b.b_id: b for b in self.billboards}
        self.billboard_id_to_node_idx = {b.b_id: i for i, b in enumerate(self.billboards)}

        # Load advertiser data
        adv_df = pd.read_csv(advertiser_csv)
        logger.info(f"Loaded {len(adv_df)} advertiser templates")

        self.ads_db: List[Ad] = []
        for _, r in adv_df.iterrows():
            self.ads_db.append(Ad(
                int(r['Id']), float(r['Demand']), float(r['Payment']),
                float(r['Payment_Demand_Ratio']), ttl=15  # Increased TTL
            ))

        # Load trajectory data
        traj_df = pd.read_csv(trajectory_csv)
        if 'Time' not in traj_df.columns:
            raise ValueError("Trajectory CSV missing 'Time' column")

        traj_df['t_min'] = traj_df['Time'].apply(time_str_to_minutes)
        self.start_time_min = int(start_time_min if start_time_min is not None else traj_df['t_min'].min())

        # Preprocess trajectories
        self.trajectory_map = self._preprocess_trajectories(traj_df)
        logger.info(f"Processed trajectories for {len(self.trajectory_map)} time points")

    def _preprocess_trajectories(self, df: pd.DataFrame) -> Dict[int, List[Tuple[float, float]]]:
        """Preprocess trajectory data for efficient lookup."""
        traj_map: Dict[int, List[Tuple[float, float]]] = {}

        for t_min, grp in df.groupby('t_min'):
            coords = list(zip(grp['Latitude'].astype(float), grp['Longitude'].astype(float)))
            traj_map[int(t_min)] = coords

        return traj_map

    def _setup_action_observation_spaces(self):
        """Setup action and observation spaces based on action mode."""

        # Node features: billboard properties
        self.n_node_features = 10
        # Ad features: advertisement properties
        self.n_ad_features = 8

        if self.action_mode == 'na':
            # Node Action: select billboard (ad chosen by environment)
            self.action_spaces = {'Allocator_0': spaces.MultiBinary(self.n_nodes)}
            self.observation_spaces = {'Allocator_0': spaces.Dict({
                'graph_nodes': spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(self.n_nodes, self.n_node_features), dtype=np.float32),
                'graph_edge_links': spaces.Box(low=0, high=self.n_nodes-1,
                                               shape=(2, self.edge_index.shape[1]), dtype=np.int64),
                'mask': spaces.MultiBinary(self.n_nodes),
                'current_ad': spaces.Box(low=-np.inf, high=np.inf,
                                         shape=(self.n_ad_features,), dtype=np.float32)
            })}

        elif self.action_mode == 'ea':
            # Edge Action: select ad-billboard pairs
            max_pairs = self.max_active_ads * self.n_nodes
            self.action_spaces = {'Allocator_0': spaces.MultiBinary(max_pairs)}
            self.observation_spaces = {'Allocator_0': spaces.Dict({
                'graph_nodes': spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(self.n_nodes, self.n_node_features), dtype=np.float32),
                'graph_edge_links': spaces.Box(low=0, high=self.n_nodes-1,
                                               shape=(2, self.edge_index.shape[1]), dtype=np.int64),
                'ad_features': spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(self.max_active_ads, self.n_ad_features), dtype=np.float32),
                'mask': spaces.MultiBinary(max_pairs)
            })}

        elif self.action_mode == 'mh':
            # Multi-Head: select ad, then billboard
            self.action_spaces = {'Allocator_0': spaces.MultiBinary([self.max_active_ads, self.n_nodes])}
            self.observation_spaces = {'Allocator_0': spaces.Dict({
                'graph_nodes': spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(self.n_nodes, self.n_node_features), dtype=np.float32),
                'graph_edge_links': spaces.Box(low=0, high=self.n_nodes-1,
                                               shape=(2, self.edge_index.shape[1]), dtype=np.int64),
                'ad_features': spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(self.max_active_ads, self.n_ad_features), dtype=np.float32),
                # Keep the structure/name the same, but the mask will represent valid (ad, billboard) pairs
                'mask': spaces.MultiBinary([self.max_active_ads, self.n_nodes])
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

    # ---------------- Distance Factor ----------------
    def distance_factor(self, dist_meters: float) -> float:
        """Distance effect on billboard influence with 10m cap at 0.9."""
        if dist_meters <= 10.0:
            return 0.9
        return max(0.1, 1.0 - (dist_meters / self.influence_radius_meters))

    def get_mask(self) -> np.ndarray:
        """Get action mask based on current action mode."""
        if self.action_mode == 'na':
            # Mask free billboards
            return np.array([1 if b.is_free() else 0 for b in self.billboards], dtype=np.int8)

        elif self.action_mode == 'ea':
            # Mask valid ad-billboard pairs (flattened)
            mask = np.zeros(self.max_active_ads * self.n_nodes, dtype=np.int8)
            active_ads = [ad for ad in self.ads if ad.state == 0]

            for ad_idx, _ in enumerate(active_ads[:self.max_active_ads]):
                for bb_idx, billboard in enumerate(self.billboards):
                    if billboard.is_free():
                        pair_idx = ad_idx * self.n_nodes + bb_idx
                        mask[pair_idx] = 1
            return mask

        elif self.action_mode == 'mh':
            # FIXED: Provide a 2D mask over (ad_idx, bb_idx) pairs to match MultiBinary([A, B])
            active_ads = [ad for ad in self.ads if ad.state == 0]
            pair_mask = np.zeros((self.max_active_ads, self.n_nodes), dtype=np.int8)

            free_cols = np.array([1 if b.is_free() else 0 for b in self.billboards], dtype=np.int8)
            for i in range(min(len(active_ads), self.max_active_ads)):
                pair_mask[i, :] = free_cols  # valid billboard choices for this active ad

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
            ad_features = np.zeros((self.max_active_ads, self.n_ad_features), dtype=np.float32)
            active_ads = [ad for ad in self.ads if ad.state == 0]

            for i, ad in enumerate(active_ads[:self.max_active_ads]):
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
    
    def _calculate_influence_for_ad_set(self, ad: Ad) -> float:
        """Calculate influence using exact problem formulation.

        Implements: I(S) = sum_{u_i in U} [1 - prod_{b_j in S} (1 - Pr(b_j, u_i))]
        Where:
         - S is the set of billboards assigned to this ad
         - U is the set of all users (trajectories) at current time
         - Pr(b_j, u_i) is probability that billboard b_j influences user u_i
         - Pr(b_j, u_i) = Size(b_j) / max_billboard_size * distance_factor
        """
        # S: Set of billboards assigned to this ad
        bbs = [self.billboard_map[b_id] for b_id in ad.assigned_billboards
               if b_id in self.billboard_map]

        if not bbs:
            return 0.0

        # Get user locations for current time (set U)
        minute_key = (self.start_time_min + self.current_step) % 1440
        user_locs = self.trajectory_map.get(minute_key, [])

        if not user_locs:
            return 0.0

        total_influence = 0.0

        for u_lat, u_lon in user_locs:  # For each user u_i in U
            prob_no_influence = 1.0  # prod_{b_j in S} (1 - Pr(b_j, u_i))

            for b in bbs:  # For each billboard b_j in S
                # Distance gate within radius
                dist_meters = haversine_distance(b.latitude, b.longitude, u_lat, u_lon)

                if dist_meters <= self.influence_radius_meters:
                    # Base prob from size normalization
                    pr_b_j_u_i = (b.b_size / max(1e-9, self.max_billboard_size))

                    # Apply distance decay with 10m cap (0.9) and floor 0.1
                    pr_b_j_u_i *= self.distance_factor(dist_meters)

                    # Numerical safety clamp
                    pr_b_j_u_i = max(0.0, min(0.999999, pr_b_j_u_i))

                    # Update product: prod_{b_j in S} (1 - Pr(b_j, u_i))
                    prob_no_influence *= (1.0 - pr_b_j_u_i)

            # Add user's influence: 1 - prod_{b_j in S} (1 - Pr(b_j, u_i))
            user_influence = 1.0 - prob_no_influence
            total_influence += user_influence

        return total_influence

    def _compute_reward(self) -> float:
        """Compute reward for current state with proper balance."""
        # Cost component (negative)
        total_cost = sum(b.b_cost for b in self.billboards if not b.is_free())

        # Penalty component (negative)
        tardiness_penalty = sum(1 for ad in self.ads if ad.state == 2) * self.tardiness_cost

        # Revenue from ongoing ads (proportional to progress)
        ongoing_revenue = 0.0
        for ad in self.ads:
            if ad.state == 0 and ad.assigned_billboards:  # Ongoing with assignments
                progress_ratio = min(1.0, ad.cumulative_influence / max(ad.demand, 1e-6))
                ongoing_revenue += ad.payment * progress_ratio * 0.1  # 10% per step of completion

        # Completion bonus (positive)
        completion_bonus = sum(ad.payment for ad in self.ads if ad.state == 1) * 0.5  # 50% bonus

        # Efficiency bonus for utilization
        utilization_bonus = (sum(1 for b in self.billboards if not b.is_free()) / max(1, self.n_nodes)) * 10.0

        return ongoing_revenue + completion_bonus + utilization_bonus - total_cost - tardiness_penalty

    def _apply_influence_for_current_minute(self):
        """Apply influence for current minute and complete ads if demand is met."""
        if self.debug:
            minute = (self.start_time_min + self.current_step) % 1440
            logger.debug(f"Applying influence at step {self.current_step} (minute {minute})")

        for ad in list(self.ads):
            if ad.state != 0:
                continue

            delta = self._calculate_influence_for_ad_set(ad)
            ad.cumulative_influence += delta

            if self.debug and delta > 0:
                logger.debug(f"Ad {ad.aid} gained {delta:.4f} influence")

            # Complete ad if demand is satisfied
            if ad.cumulative_influence >= ad.demand:
                ad.state = 1  # completed
                self.performance_metrics['total_ads_completed'] += 1

                # Release billboards and generate revenue
                for b_id in ad.assigned_billboards[:]:
                    if b_id in self.billboard_map:
                        billboard = self.billboard_map[b_id]
                        # Generate proper revenue split
                        billboard.revenue_generated += ad.payment / max(1, len(ad.assigned_billboards))
                        billboard.release()
                    ad.release_billboard(b_id)

                if self.debug:
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
        n_spawn = random.randint(*self.new_ads_per_step_range)
        current_ad_ids = {ad.aid for ad in self.ads}
        available_templates = [a for a in self.ads_db if a.aid not in current_ad_ids]

        spawn_count = min(
            self.max_active_ads - len(self.ads),
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

            if self.debug:
                logger.debug(f"Spawned {spawn_count} new ads")

    def _execute_action(self, action):
        """Execute the selected action based on action mode."""

        if self.action_mode == 'na':
            # Node Action: environment-selected ad to agent-selected billboard(s)
            active_ads = [ad for ad in self.ads if ad.state == 0]

            # Action is now expected to be a MultiBinary vector of shape (n_nodes,)                                       
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

                        if self.debug:
                            logger.debug(f"Assigned ad {ad_to_assign.aid} to billboard {billboard.b_id} for {duration} steps")

        elif self.action_mode == 'ea':
            # Action is now a 2D MultiBinary matrix [max_active_ads, n_nodes]
            if isinstance(action, (list, np.ndarray)):
                action = np.array(action)
                active_ads = [ad for ad in self.ads if ad.state == 0]

                for ad_idx in range(min(len(active_ads), self.max_active_ads)):
                    for bb_idx in range(self.n_nodes):
                        if action[ad_idx, bb_idx] == 1 and self.billboards[bb_idx].is_free():
                            ad_to_assign = active_ads[ad_idx]
                            duration = random.randint(*self.slot_duration_range)

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
            # Action is now also a 2D MultiBinary matrix [max_active_ads, n_nodes]
            if isinstance(action, (list, np.ndarray)):
                action = np.array(action)
                active_ads = [ad for ad in self.ads if ad.state == 0]

                for ad_idx in range(min(len(active_ads), self.max_active_ads)):
                    for bb_idx in range(self.n_nodes):
                        if action[ad_idx, bb_idx] == 1 and self.billboards[bb_idx].is_free():
                            ad_to_assign = active_ads[ad_idx]
                            duration = random.randint(*self.slot_duration_range)

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

        # Spawn initial ads
        self._spawn_ads()

        logger.info(f"Environment reset with {len(self.ads)} initial ads")

        return self._get_obs(), {}

    def observe(self, agent: str) -> Dict[str, Any]:
        """Get observation for specified agent."""
        return self._get_obs()

    def step(self, action):
        """Execute one environment step."""
        agent = self.agent_selection

        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            return

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
        self.performance_metrics['billboard_utilization'] = occupied_count / max(1, self.n_nodes)

        # 7. Spawn new ads
        self._spawn_ads()

        # 8. Update termination conditions
        self.current_step += 1

        for a in self.agents:
            self.terminations[a] = (self.current_step >= self.max_events)
            self.infos[a] = {
                'performance_metrics': self.performance_metrics.copy(),
                'current_minute': (self.start_time_min + self.current_step) % 1440
            }

        # 9. Advance agent selection
        if not self._agent_selector.is_last():
            self.agent_selection = self._agent_selector.next()

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
            for ad in active_with_assignments[:10]:  # Show first 10
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
        print(f"  Utilization: {metrics['billboard_utilization']:.1%}")

        if self.current_step >= self.max_events:
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
        print(f"Average Billboard Utilization: {metrics['billboard_utilization']*100:.1f}%")
        print(f"Total Placements: {len(self.placement_history)}")

        # Ensure fulfilled_by_end is populated for all placements
        for rec in self.placement_history:
            if 'fulfilled_by_end' not in rec:
                ad = next((a for a in self.ads if a.aid == rec['ad_id']), None)
                rec['fulfilled_by_end'] = ad.cumulative_influence if ad else 0.0

        if self.placement_history:
            ratios = []
            for rec in self.placement_history:
                d = rec.get('demand', 0.0)
                f = rec.get('fulfilled_by_end', 0.0)
                if d > 0:
                    ratios.append(min(1.0, f / d))
            if ratios:
                avg_fulfillment = float(np.mean(ratios))
                print(f"Average Demand Fulfillment: {avg_fulfillment*100:.1f}%")
            else:
                print("Average Demand Fulfillment: N/A")

    def close(self):
        """Clean up environment."""
        pass
