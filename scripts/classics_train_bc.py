import gymnasium as gym
import torch
import numpy as np
import os
import pickle
import csv
import sys
from contextlib import contextmanager

from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# --- Configuration & Paper Specifications ---
classics_envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
target_counts = [1, 4, 7, 10]  # Standard trajectory counts for evaluation
base_data_dir = "expert_data"
base_results_dir = "results"
csv_file_path = os.path.join(base_results_dir, "bc_stats.csv")

os.makedirs(base_results_dir, exist_ok=True)

# Initialize CSV
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
# MAIN LOOP (Behavioral Cloning)
# ==========================================
for env_name in classics_envs:
    print(f"\n>>> Processing BC on: {env_name}")
    
    # Paper uses neural network policies with two hidden layers of 100 units
    # Observation normalization is critical for stable learning
    venv = make_vec_env(env_name, n_envs=1, rng=np.random.default_rng(42))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    for count in target_counts:
        dataset_path = os.path.join(base_data_dir, env_name, f"traj_{count}.pkl")
        if not os.path.exists(dataset_path): continue
            
        print(f"--- Training BC with {count} trajectories ---")
        
        try:
            with open(dataset_path, "rb") as f:
                demonstrations = pickle.load(f)

            # Paper uses Adam optimizer and batch size 128
            # Architecture: 2 hidden layers, 100 units each, tanh non-linearities
            custom_net_arch = [100, 100] 
            
            bc_trainer = bc.BC(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                demonstrations=demonstrations,
                batch_size=128,  # Spec from paper
                rng=np.random.default_rng(42)
            )

            # Training: Paper splits data 70/30 and trains until validation error stops
            print("Training started...")
            with suppress_output():
                # We approximate 'training until validation error stops' with epochs
                bc_trainer.train(n_epochs=50) 
            
            mean_r, std_r = evaluate_policy(bc_trainer.policy, venv, n_eval_episodes=10)
            print(f"Result: {mean_r:.2f} +/- {std_r:.2f}")

            with open(csv_file_path, mode='a', newline='') as f:
                csv.writer(f).writerow([env_name, count, mean_r, std_r])

        except Exception as e:
            print(f"Error in {env_name} ({count}): {e}")

print("\nBC experiments completed.")
