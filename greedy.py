import numpy as np
import random
import logging
from typing import Dict, Any, List

# Import the corrected environment class
# Make sure 'corrected_environment.py' is in the same directory
from env2 import BillboardEnv, Ad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# DEMAND-AWARE MULTI-ALLOCATION GREEDY POLICY
# ------------------------------------------------------------
def smart_greedy_policy(env: BillboardEnv, epsilon: float = 0.1):
    """
    Greedy policy that allocates MULTIPLE billboards to the best-scored ad.
    Correctly interacts with the PettingZoo AECEnv API.
    """
    env.reset()
    total_reward = 0.0
    
    # --- Helper for calculating marginal influence without modifying env state ---
    def get_marginal_influence(ad_obj: Ad, bb_idx_to_add: int, current_assigned_ids: List[int]) -> float:
        # Create temporary Ad object to avoid state modification
        temp_ad = Ad(ad_obj.aid, ad_obj.demand, ad_obj.payment, ad_obj.payment_demand_ratio)
        temp_ad.assigned_billboards = list(current_assigned_ids)
        
        base_influence = env._calculate_influence_for_ad_set(temp_ad)
        
        b_id_to_add = env.billboards[bb_idx_to_add].b_id
        if b_id_to_add not in temp_ad.assigned_billboards:
            temp_ad.assigned_billboards.append(b_id_to_add)
        
        new_influence = env._calculate_influence_for_ad_set(temp_ad)
        return max(0.0, new_influence - base_influence)

    # Correct PettingZoo interaction loop
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        total_reward += reward
        if termination or truncation:
            action = None # Agent is done, pass no action
        else:
            action = None # Default action
            active_ads = [ad for ad in env.ads if ad.state == 0]

            if env.action_mode == 'na':
                mask = obs["mask"]
                act = np.zeros(env.n_nodes, dtype=np.int8)
                valid_indices = np.where(mask == 1)[0]
                if len(valid_indices) > 0:
                    chosen_bb_idx = random.choice(valid_indices)
                    act[chosen_bb_idx] = 1
                action = act

            elif env.action_mode in ['ea', 'mh']:
                mask = obs["mask"]
                ad_features = obs["ad_features"]
                action_shape = (env.max_active_ads, env.n_nodes)
                act = np.zeros(action_shape, dtype=np.int8)

                best_ad_idx, best_ad_score = -1, -float("inf")
                for ad_idx, ad_obj in enumerate(active_ads[:env.max_active_ads]):
                    need = ad_obj.demand - ad_obj.cumulative_influence
                    if need <= 0: continue
                    payment = ad_obj.payment
                    urgency = 1.0 - ad_features[ad_idx][4] # Normalized TTL
                    score = (payment / max(need, 1e-6)) + 0.3 * urgency * payment
                    if score > best_ad_score:
                        best_ad_score = score
                        best_ad_idx = ad_idx
                
                if best_ad_idx != -1:
                    ad_obj = active_ads[best_ad_idx]
                    need = ad_obj.demand - ad_obj.cumulative_influence
                    
                    scores = []
                    for bb_idx in range(env.n_nodes):
                        if mask[best_ad_idx, bb_idx] == 1:
                            delta = get_marginal_influence(ad_obj, bb_idx, ad_obj.assigned_billboards)
                            if delta <= 0: continue
                            bb_cost = obs['graph_nodes'][bb_idx][2]
                            score = (min(delta, need) / max(need, 1e-6)) * ad_obj.payment - 0.01 * bb_cost
                            scores.append((score, bb_idx))
                    
                    scores.sort(reverse=True)
                    
                    current_influence_gain = 0
                    temp_assigned_ids_for_calc = list(ad_obj.assigned_billboards)
                    for _, bb_idx in scores:
                        if (need - current_influence_gain) / ad_obj.demand <= epsilon:
                            break
                        
                        act[best_ad_idx, bb_idx] = 1
                        
                        b_id = env.billboards[bb_idx].b_id
                        temp_assigned_ids_for_calc.append(b_id)
                        
                        # FIX: The Ad constructor does not take 'assigned_billboards' as a kwarg.
                        # 1. Create the temporary Ad object.
                        # 2. Set its assigned_billboards attribute separately.
                        temp_ad = Ad(ad_obj.aid, ad_obj.demand, ad_obj.payment, ad_obj.payment_demand_ratio)
                        temp_ad.assigned_billboards = temp_assigned_ids_for_calc
                        
                        # 3. Use the correctly configured temp_ad for calculation.
                        cumulative_potential_influence = env._calculate_influence_for_ad_set(temp_ad)
                        current_influence_gain = cumulative_potential_influence - ad_obj.cumulative_influence

                action = act

        env.step(action)
    
    # After the loop, the final metrics are in the last 'info' object
    final_info = env.infos.get(env.possible_agents[0], {})
    final_metrics = final_info.get('performance_metrics', {})
    return total_reward, final_metrics


# ------------------------------------------------------------
# RANDOM POLICY
# ------------------------------------------------------------
def random_policy(env: BillboardEnv):
    """A policy that takes random valid actions."""
    env.reset()
    total_reward = 0.0

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        total_reward += reward
        if termination or truncation:
            action = None
        else:
            mask = obs["mask"]
            valid_indices = np.argwhere(mask == 1)
            action = None

            if len(valid_indices) > 0:
                choice = random.choice(valid_indices)
                if env.action_mode == 'na':
                    action = np.zeros(env.n_nodes, dtype=np.int8)
                    action[choice[0]] = 1
                elif env.action_mode in ['ea', 'mh']:
                    action = np.zeros((env.max_active_ads, env.n_nodes), dtype=np.int8)
                    action[choice[0], choice[1]] = 1
        
        env.step(action)

    final_info = env.infos.get(env.possible_agents[0], {})
    final_metrics = final_info.get('performance_metrics', {})
    return total_reward, final_metrics

# ------------------------------------------------------------
# EVALUATION LOOP
# ------------------------------------------------------------
def evaluate_agent(env_config: Dict[str, Any], policy_fn, runs: int, label: str):
    all_metrics = []
    total_rewards = []
    for i in range(runs):
        env = BillboardEnv(**env_config, seed=42+i)
        total_reward, metrics = policy_fn(env)
        total_rewards.append(total_reward)
        all_metrics.append(metrics)
        logger.info(
            f"[{label} Run {i+1}/{runs}] "
            f"Reward={total_reward:.2f} | "
            f"Revenue=${metrics.get('total_revenue', 0):.2f} | "
            f"Completed={metrics.get('total_ads_completed', 0)}/"
            f"{metrics.get('total_ads_processed', 0)}"
        )
    return total_rewards, all_metrics

# ------------------------------------------------------------
# RESULTS ANALYSIS
# ------------------------------------------------------------
def analyze_results(rewards: List[float], metrics_list: List[dict], label: str):
    avg_reward = np.mean(rewards)
    avg_revenue = np.mean([m.get("total_revenue", 0) for m in metrics_list])
    avg_completed = np.mean([m.get("total_ads_completed", 0) for m in metrics_list])
    avg_processed = np.mean([m.get("total_ads_processed", 0) for m in metrics_list])
    completion_rate = (avg_completed / max(1, avg_processed)) * 100
    
    print("\n" + "="*40)
    logger.info(f"Analysis for: {label} ({len(rewards)} runs)")
    logger.info(f"  Average Reward: {avg_reward:.2f}")
    logger.info(f"  Average Revenue: ${avg_revenue:.2f}")
    logger.info(f"  Average Completion Rate: {completion_rate:.2f}% ({avg_completed:.1f}/{avg_processed:.1f})")
    print("="*40 + "\n")

# ------------------------------------------------------------
# MAIN with placeholder file paths
# ------------------------------------------------------------
def main():
    # FIX: Using placeholder paths. Replace with your actual file paths.
    env_config = {
        "billboard_csv": "/home/hp/Documents/DynaBillboard/BillBoard_NYC.csv",
        "advertiser_csv": "/home/hp/Documents/DynaBillboard/Advertiser_NYC2.csv",
        "trajectory_csv": "/home/hp/Documents/DynaBillboard/TJ_NYC.csv",
        "action_mode": "mh",
        "max_active_ads": 20,
        "new_ads_per_step_range": (1, 5),
        "slot_duration_range": (1, 5),
        "influence_radius_meters": 100.0,
        "tardiness_cost": 50.0,
        "max_events": 100,
        "graph_connection_distance": 5000.0,
        "debug": False,
    }
    
    runs = 3
    
    greedy_rewards, greedy_metrics = evaluate_agent(env_config, smart_greedy_policy, runs, "Smart Greedy")
    analyze_results(greedy_rewards, greedy_metrics, "Smart Greedy")
    
    random_rewards, random_metrics = evaluate_agent(env_config, random_policy, runs, "Random")
    analyze_results(random_rewards, random_metrics, "Random")

if __name__ == "__main__":
    # Note: You will need to replace the placeholder file paths in main() to run this code.
    print("Running main function. Ensure file paths in `env_config` are correct.")
    # To run, you would call main() after updating the paths.
    main()