import gymnasium as gym
import numpy as np
import os
import pickle
import csv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # <--- CRITICAL IMPORT
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.util.util import make_vec_env

# --- Configuration ---
# You can run this list one by one or all together
classics_envs = ['MountainCar-v0','Acrobot-v1','CartPole-v1'] #Add other envs here
base_data_dir = "expert_data"
base_results_dir = "results"
csv_file_path = os.path.join(base_results_dir, "expert_stats.csv")
STEPS_PER_ITERATION = 5000
TOTAL_ITERATIONS = 300
TOTAL_TIMESTEPS = STEPS_PER_ITERATION * TOTAL_ITERATIONS # 1,500,000
os.makedirs(base_data_dir, exist_ok=True)
os.makedirs(base_results_dir, exist_ok=True)

# (Optional: Re-write header if file doesn't exist, else append)
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["env_name", "nb_trajectories", "mean_reward", "std_reward"])

for env_name in classics_envs:
    print(f"\nProcessing Environment: {env_name}")
    
    env_save_path = os.path.join(base_data_dir, env_name)
    os.makedirs(env_save_path, exist_ok=True)
    
    # 1. SETUP ENVIRONMENT
    # TRPO usually needs normalization to solve MountainCar efficiently
    raw_env = make_vec_env(env_name, n_envs=1, rng=np.random.default_rng(0))
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.) # <--- THE FIX

    # 2. TRAIN EXPERT
    # We increase steps for MountainCar because the first success takes time to find
    total_steps = 50_000 if "CartPole-v1" in env_name else TOTAL_TIMESTEPS
    
    print(f"--- Training TRPO Expert (Steps: {total_steps}) ---")
    expert = TRPO(
        "MlpPolicy", 
        env, 
        verbose=0, # Turn on to see if explained_variance increases
        learning_rate=1e-3,
        n_steps=1024,
        batch_size=64,
        gae_lambda=0.98, # Slightly higher lambda helps credit assignment in long horizons
        gamma=0.99,
    )

    expert.learn(total_timesteps=total_steps)

    # 3. EVALUATE (CRITICAL STEP)
    # We must turn off training-time reward normalization to see the REAL score (-200 vs -110)
    # But we keep observation normalization because the agent expects it.
    env.training = False 
    env.norm_reward = False
    
    global_reward, global_std = evaluate_policy(expert, env, n_eval_episodes=10)
    print(f"TRPO Expert Global Mean Reward: {global_reward:.2f} (Target: > -110)")

    # STOP IF FAILED
    if global_reward <= -199:
        print("!!! ERROR: Expert failed to solve MountainCar. Do not generate data from this.")
        continue 

    # ==========================================
    # 4. GENERATE TRAJECTORIES
    # ==========================================
    print("--- Generating Demonstrations ---")
    rng = np.random.default_rng()
    
    # We use the normalized env to collect data so the observations match what the agent saw
    raw_rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_episodes=20), 
        rng=rng,
        unwrap=False 
    )
 
    # UNWRAP/UN-NORMALIZE DATA?
    # Usually, we want to save the "real" observations for the imitator to learn from scratch.
    # However, if your imitator also uses VecNormalize (recommended), keeping them normalized is okay.
    # For a pure reproduction, it's safer to use the original observations. 
    # The 'rollout' function gets observations from the env. 
    # Since 'env' is VecNormalize, these obs are normalized.
    
    # Let's truncate and save
    processed_rollouts = []
    for traj in raw_rollouts:
        if len(traj.obs) > 50:
            truncated_traj = TrajectoryWithRew(
                obs=traj.obs[:51],       
                acts=traj.acts[:50],
                infos=traj.infos[:50] if traj.infos is not None else None,
                terminal=False, 
                rews=traj.rews[:50] # Real rewards
            )
            processed_rollouts.append(truncated_traj)

    # 5. SAVE SUBSETS
    target_counts = [4, 7, 10]
    for count in target_counts:
        if len(processed_rollouts) >= count:
            subset = processed_rollouts[:count]
            filename = f"traj_{count}.pkl"
            full_path = os.path.join(env_save_path, filename)
            
            with open(full_path, "wb") as f:
                pickle.dump(subset, f)
            
            with open(csv_file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                # Log the GLOBAL expert performance, not the truncated one
                writer.writerow([env_name, count, f"{global_reward:.2f}", f"{global_std:.2f}"])
            
            print(f" -> Saved {count} trajectories. Expert Perf: {global_reward:.2f}")

    # Reset env training mode for next iteration just in case
    env.training = True
    env.norm_reward = True
