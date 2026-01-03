import gymnasium as gym
import torch
import numpy as np
import os
import pickle
import csv
import sys
from contextlib import contextmanager

from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

# --- Configuration & Paper Parameters (Table 2) ---
TASK_CONFIGS = {
    "CartPole-v1":    {"iters": 300, "samples": 5000},
    "MountainCar-v0": {"iters": 300, "samples": 5000},
    "Acrobot-v1":     {"iters": 300, "samples": 5000},
}

target_counts = [4, 7, 10]
base_data_dir = "expert_data"    
base_results_dir = "results"     
csv_file_path = os.path.join(base_results_dir, "fem_stats.csv")

os.makedirs(base_results_dir, exist_ok=True)
with open(csv_file_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["env_name", "nb_trajectories", "mean_reward", "std_reward"])

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try: yield
        finally: sys.stdout, sys.stderr = old_stdout, old_stderr

# ==========================================
# FEM EXECUTION LOOP
# ==========================================
for env_name, config in TASK_CONFIGS.items():
    print(f"\n>>> Processing FEM on: {env_name}")
    
    # Paper uses TRPO; vector normalization is key for stable training [cite: 467]
    venv = make_vec_env(env_name, n_envs=1, rng=np.random.default_rng(42))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    for count in target_counts:
        dataset_path = os.path.join(base_data_dir, env_name, f"traj_{count}.pkl")
        if not os.path.exists(dataset_path): continue
            
        try:
            with open(dataset_path, "rb") as f:
                demonstrations = pickle.load(f)

            # Dynamic batching for small expert datasets
            total_transitions = len(demonstrations) * 50
            safe_batch_size = min(64, total_transitions // 2)

            print(f"--- Training FEM | {count} Trajs | Batch Size: {safe_batch_size} ---")

            # Generator: TRPO [cite: 144, 207]
            learner = TRPO(
                "MlpPolicy", venv, verbose=0, 
                n_steps=config["samples"], 
                gamma=0.995
            )

            # FEM: Linear cost function approximation [cite: 121, 213]
            reward_net = BasicRewardNet(
                venv.observation_space, venv.action_space,
                normalize_input_layer=RunningNorm,
                hid_sizes=[] # <--- Constrains the cost class to be Linear [cite: 121]
            )

            trainer = GAIL(
                demonstrations=demonstrations,
                demo_batch_size=safe_batch_size,
                gen_algo=learner,
                reward_net=reward_net,
                venv=venv,                   
                allow_variable_horizon=True 
            )

            # Total steps: Iterations * Samples per iteration 
            total_steps = config["iters"] * config["samples"]
            with suppress_output():
                trainer.train(total_timesteps=total_steps)
            
            mean_r, std_r = evaluate_policy(learner, venv, n_eval_episodes=10)
            print(f"Result: {mean_r:.2f} +/- {std_r:.2f}")

            with open(csv_file_path, mode='a', newline='') as f:
                csv.writer(f).writerow([env_name, count, mean_r, std_r])

        except Exception as e:
            print(f"Error in {env_name} ({count}): {e}")

print("\nFEM experiments completed.")
