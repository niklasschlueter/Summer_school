import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
plots_dir = project_dir / "plotting" / "plots"

# Load CSV
df = pd.read_csv(data_dir / "trajectory_feedback_log.csv")

# Parse list-like strings into Python lists
df['joint_names'] = df['joint_names'].apply(ast.literal_eval)
df['desired_positions'] = df['desired_positions'].apply(ast.literal_eval)
df['actual_positions'] = df['actual_positions'].apply(ast.literal_eval)
df['desired_velocities'] = df['desired_velocities'].apply(ast.literal_eval)
df['actual_velocities'] = df['actual_velocities'].apply(ast.literal_eval)

# Number of samples and joints
num_samples = len(df)
joint_names = df['joint_names'][0]  # Assuming same order for all rows
num_joints = len(joint_names)

# Convert to numpy arrays for plotting
desired_positions = np.array(df['desired_positions'].tolist())
actual_positions = np.array(df['actual_positions'].tolist())
desired_velocities = np.array(df['desired_velocities'].tolist())
actual_velocities = np.array(df['actual_velocities'].tolist())

# X-axis: sample indices
x = np.arange(num_samples)
t = np.array(df["timestamp"].tolist())
t = t - t[0]
x = t

# Plot Positions
plt.figure(figsize=(12, 6))
for i in range(num_joints):
    plt.scatter(x, desired_positions[:, i], label=f"{joint_names[i]} desired")#, linestyle='--')
    plt.scatter(x, actual_positions[:, i], label=f"{joint_names[i]} actual")#, linestyle='-')
plt.title("Joint Positions (Desired vs Actual)")
plt.xlabel("Sample Index")
plt.ylabel("Position (rad or m)")
plt.legend(ncol=2, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "joint_positions_feedback.pdf")

# Plot Velocities
plt.figure(figsize=(12, 6))
for i in range(num_joints):
    plt.scatter(x, desired_velocities[:, i], label=f"{joint_names[i]} desired")#, linestyle='--')
    plt.scatter(x, actual_velocities[:, i], label=f"{joint_names[i]} actual")#, linestyle='-')
plt.title("Joint Velocities (Desired vs Actual)")
plt.xlabel("Sample Index")
plt.ylabel("Velocity")
plt.legend(ncol=2, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir /"joint_velocities_feedback.pdf")
