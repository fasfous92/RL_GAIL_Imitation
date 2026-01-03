import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from imitation.rewards.reward_nets import RewardNet
from imitation.util.util import make_vec_env

# ==========================================
# 1. CUSTOM LSTM REWARD NET (TEMPORAL DISCRIMINATOR)
# ==========================================
class LSTM_RewardNet(RewardNet):
    def __init__(self, observation_space, action_space, hidden_size=64, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        state_dim = observation_space.shape[0]
        
        if isinstance(action_space, gym.spaces.Discrete):
            self.is_discrete = True
            action_dim = action_space.n 
        else:
            self.is_discrete = False
            action_dim = action_space.shape[0]

        input_dim = state_dim + action_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, state, action, next_state, done):
        if state.ndim > 2:
            state = state.flatten(start_dim=1)
        batch_size = state.shape[0]

        if self.is_discrete:
            if not (action.ndim == 2 and action.shape[1] == self.action_space.n):
                action = action.reshape(batch_size)
                action = F.one_hot(action.long(), num_classes=self.action_space.n).float()

        inputs = torch.cat([state, action], dim=1).unsqueeze(1) 
        out, _ = self.lstm(inputs)
        return self.head(out[:, -1, :]).squeeze(-1)

# ==========================================
# 2. CONFIGURATION (BASED ON TABLE 2)
# ==========================================
# Task configs mapped from the paper for classic control [cite: 469]
TASK_CONFIGS = {
    "CartPole-v1":    {"iters": 300, "samples": 5000},
    "MountainCar-v0": {"iters": 300, "samples": 5000},
    "Acrobot-v1":     {"iters": 300, "samples": 5000},
}

target_counts = [4, 7, 10]
base_data_dir = "expert_data"    
base_results_dir = "results"     
csv_file_path = os.path.join(base_results_dir, "gatil_stats.csv")

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
# 3. MAIN LOOP (GATIL)
# ==========================================
for env_name, config in TASK_CONFIGS.items():
    print(f"\n>>> Processing GATIL (Temporal GAIL) on: {env_name}")
    
    # Paper uses normalized observations for stable training [cite: 467]
    venv = make_vec_env(env_name, n_envs=1, rng=np.random.default_rng(42))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    for count in target_counts:
        dataset_path = os.path.join(base_data_dir, env_name, f"traj_{count}.pkl")
        if not os.path.exists(dataset_path): continue
            
        try:
            with open(dataset_path, "rb") as f:
                demonstrations = pickle.load(f)

            # Dynamic batching: safe for small datasets (e.g., 4 trajectories)
            total_transitions = len(demonstrations) * 50
            safe_batch_size = min(64, total_transitions // 2)

            print(f"--- Training GATIL | {count} Trajs | Batch Size: {safe_batch_size} ---")

            # Generator: TRPO to match the model-free approach of the paper [cite: 178, 180]
            learner = TRPO(
                "MlpPolicy", venv, verbose=0, 
                n_steps=config["samples"], # 5,000 steps per iteration [cite: 469]
                gamma=0.995                # Standard infinite horizon discount [cite: 467]
            )

            reward_net = LSTM_RewardNet(venv.observation_space, venv.action_space)

            trainer = GAIL(
                demonstrations=demonstrations,
                demo_batch_size=safe_batch_size,
                gen_algo=learner,
                reward_net=reward_net,
                venv=venv,                   
                allow_variable_horizon=True 
            )

            # Total timesteps: 1.5M for classic tasks [cite: 469]
            total_steps = config["iters"] * config["samples"]
            with suppress_output():
                trainer.train(total_timesteps=total_steps)
            
            mean_r, std_r = evaluate_policy(learner, venv, n_eval_episodes=10)
            print(f"Result: {mean_r:.2f} +/- {std_r:.2f}")

            with open(csv_file_path, mode='a', newline='') as f:
                csv.writer(f).writerow([env_name, count, mean_r, std_r])

        except Exception as e:
            print(f"Error in {env_name} ({count}): {e}")

print("\nGATIL experiments completed.")
