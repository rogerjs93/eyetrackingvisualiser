"""Simple test of autism data visualization"""
import matplotlib.pyplot as plt
import numpy as np
from autism_data_loader import AutismDataLoader
from scipy.ndimage import gaussian_filter

# Load data
loader = AutismDataLoader()
data = loader.load_participant_data(2)
info = loader.get_participant_info(2)

print(f"Loading Participant {info['ParticipantID']}")
print(f"Age: {info['Age']}, Gender: {info['Gender']}, CARS: {info['CARS Score']}")
print(f"Data points: {len(data)}")
print(f"Duration: {data['timestamp'].max()/1000:.1f} seconds")

# Create simple visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = data['x'].values
y = data['y'].values
timestamp = data['timestamp'].values

# Heatmap
heatmap, xedges, yedges = np.histogram2d(x, y, bins=[50, 50], range=[[0, 1920], [0, 1080]])
heatmap_smooth = gaussian_filter(heatmap, sigma=2)
axes[0, 0].imshow(heatmap_smooth.T, origin='lower', cmap='hot', aspect='auto', extent=[0, 1920, 0, 1080])
axes[0, 0].set_title('Gaze Heatmap')
axes[0, 0].set_xlabel('X Position')
axes[0, 0].set_ylabel('Y Position')

# Scan path
colors = np.linspace(0, 1, len(x))
axes[0, 1].scatter(x, y, c=colors, cmap='viridis', s=1, alpha=0.5)
axes[0, 1].scatter(x[0], y[0], c='green', s=100, marker='*', label='Start')
axes[0, 1].scatter(x[-1], y[-1], c='red', s=100, marker='*', label='End')
axes[0, 1].set_title('Scan Path')
axes[0, 1].set_xlim(0, 1920)
axes[0, 1].set_ylim(1080, 0)
axes[0, 1].legend()

# X position over time
axes[1, 0].plot(timestamp/1000, x, alpha=0.5, linewidth=0.5)
axes[1, 0].set_title('X Position Over Time')
axes[1, 0].set_xlabel('Time (seconds)')
axes[1, 0].set_ylabel('X Position')
axes[1, 0].grid(True, alpha=0.3)

# Y position over time
axes[1, 1].plot(timestamp/1000, y, alpha=0.5, linewidth=0.5, color='red')
axes[1, 1].set_title('Y Position Over Time')
axes[1, 1].set_xlabel('Time (seconds)')
axes[1, 1].set_ylabel('Y Position')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autism_test_viz.png', dpi=100)
print("\n✅ Saved visualization as autism_test_viz.png")
plt.close()

print("✅ Test complete!")
