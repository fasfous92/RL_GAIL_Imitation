import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import math

# --- Configuration ---
results_dir = "results"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Define display names and colors for each algorithm
algo_config = {
    'expert_stats': ('Expert', 'black', None),         
    'bc_stats':     ('Behavioral Cloning', '#d55e00', 's'), 
    'fem_stats':    ('FEM', '#cc79a7', 'X'),           
    'gatil_stats':  ('GATIL ', '#009e73', 'D'),   
    'gail_stats':   ('GAIL', '#0072b2', 'o')          
}

# 1. Load and Merge Data
print("--- Loading Results ---")
all_data = []

csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

for filepath in csv_files:
    filename = os.path.basename(filepath).replace(".csv", "")
    
    if filename not in algo_config:
        print(f"Skipping unknown file: {filename}")
        continue
        
    df = pd.read_csv(filepath)
    df['algorithm'] = filename
    all_data.append(df)

if not all_data:
    print("No data found in results folder!")
    exit()

full_df = pd.concat(all_data, ignore_index=True)
environments = sorted(full_df['env_name'].unique())
n_envs = len(environments)

print(f"Found {n_envs} environments: {environments}")

# 2. Setup Subplots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# Calculate grid size
n_cols = 3 if n_envs >= 3 else n_envs
n_rows = math.ceil(n_envs / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)

if n_envs > 1:
    axes_flat = axes.flatten()
else:
    axes_flat = [axes]

# 3. Plotting Loop
handles, labels = [], []

for i, env_name in enumerate(environments):
    print(f"Plotting normalized results for {env_name}...")
    ax = axes_flat[i]
    
    env_data = full_df[full_df['env_name'] == env_name]
    
    # --- FIND EXPERT BASELINE FOR NORMALIZATION ---
    expert_df = env_data[env_data['algorithm'] == 'expert_stats']
    if not expert_df.empty:
        # We assume expert performance is the target (1.0)
        # We take the max in case there is variance, or mean if preferred
        expert_val = expert_df['mean_reward'].max()
    else:
        expert_val = 1.0 # Avoid division by zero if missing, or handle error
        print(f"Warning: No expert data for {env_name}, skipping normalization.")

    # Prevent division by zero if expert gets 0 reward (unlikely but possible)
    if expert_val == 0: 
        expert_val = 1.0 

    # A. Plot Expert Line at 1.0 (Normalized)
    ax.axhline(y=1.0, color='black', linestyle='--', label='Expert', linewidth=2, alpha=0.7)
    
    # B. Plot Random Baseline at 0.0 (Normalized)
    ax.axhline(y=0, color='gray', linestyle='--', label='Random', linewidth=2, alpha=0.5)

    # C. Plot Algorithms (Normalized)
    for algo_key, (label, color, marker) in algo_config.items():
        if algo_key == 'expert_stats': 
            continue 
            
        subset = env_data[env_data['algorithm'] == algo_key].sort_values('nb_trajectories')
        
        if subset.empty:
            continue

        x = pd.to_numeric(subset['nb_trajectories'])
        
        # --- NORMALIZE REWARDS ---
        # Formula: Normalized = Raw_Reward / Expert_Reward
        # (Assuming Random ~ 0. If Random is not 0, use (x - rand)/(expert - rand))
        y = pd.to_numeric(subset['mean_reward']) / expert_val
        std = pd.to_numeric(subset['std_reward']) / expert_val
        
        # Plot Line
        ax.plot(x, y, label=label, color=color, marker=marker, markersize=8, linewidth=2.5)
        # Plot Error
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.2)

    # D. Formatting per Subplot
    ax.set_title(f"{env_name}", fontsize=18, weight='bold')
    ax.set_xlabel("Number of Trajectories", fontsize=14)
    ax.set_ylabel("Normalized Score (Expert=1.0)", fontsize=14)
    ax.set_ylim(bottom=-0.1, top=1.2) # Lock Y-axis to show 0 to 1 clearly
    
    valid_ticks = sorted(full_df['nb_trajectories'].unique())
    ax.set_xticks(valid_ticks)
    
    # Legend collection
    if i == 0:
        h, l = ax.get_legend_handles_labels()
        by_label = dict(zip(l, h))
        handles = list(by_label.values())
        labels = list(by_label.keys())

# 4. Cleanup and Save
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis('off')

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
           ncol=len(labels), frameon=True, fontsize=14)

save_path = os.path.join(output_dir, "normalized_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"\nSaved normalized plot to {save_path}")
