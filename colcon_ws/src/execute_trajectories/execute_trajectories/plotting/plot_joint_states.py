import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
plots_dir = project_dir / "plotting" / "plots"

# Load data
df = pd.read_csv(data_dir / "joint_states_log.csv")

# Parse string lists into Python lists
df['position'] = df['position'].apply(ast.literal_eval)
df['velocity'] = df['velocity'].apply(ast.literal_eval)
df['effort'] = df['effort'].apply(ast.literal_eval)

# Get number of samples and joints
num_samples = len(df)
num_joints = len(df['position'][0])

# Prepare arrays for plotting
positions = np.array(df['position'].tolist())
velocities = np.array(df['velocity'].tolist())
efforts = np.array(df['effort'].tolist())

# X-axis: sample indices
x = np.arange(num_samples)
t = np.array(df["timestamp"].tolist())
t = t - t[0]
x = t

joint_names = ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", 
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# Plot positions
plt.figure(figsize=(12, 6))
for i in range(num_joints):
    plt.scatter(x, positions[:, i], label=joint_names[i])#, s=10)
plt.title('Joint Positions')
plt.xlabel('Sample Index')
plt.ylabel('Position (rad or m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "joint_positions.pdf")

# Plot velocities
plt.figure(figsize=(12, 6))
for i in range(num_joints):
    plt.scatter(x, velocities[:, i], label=joint_names[i])#, s=10)
plt.title('Joint Velocities')
plt.xlabel('Sample Index')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "joint_velocities.pdf")

# Plot efforts
plt.figure(figsize=(12, 6))
for i in range(num_joints):
    plt.scatter(x, efforts[:, i], label=joint_names[i])#, s=10)
plt.title('Joint Efforts')
plt.xlabel('Sample Index')
plt.ylabel('Effort')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "joint_efforts.pdf")
