"""
Lightweight Autism Data Visualizer
Quick visualization of autism eye-tracking data without heavy dependencies
"""

import matplotlib.pyplot as plt
import seaborn as sns
from autism_data_loader import AutismDataLoader
import numpy as np
from scipy.ndimage import gaussian_filter

def visualize_participant(participant_id, save_fig=False):
    """Create comprehensive visualization for a participant."""
    
    # Load data
    print(f"Loading Participant {participant_id}...")
    loader = AutismDataLoader()
    data = loader.load_participant_data(participant_id)
    info = loader.get_participant_info(participant_id)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Participant {participant_id} - Age: {info["Age"]}, Gender: {info["Gender"]}, CARS: {info["CARS Score"]}',
                 fontsize=16, fontweight='bold')
    
    x = data['x'].values
    y = data['y'].values
    timestamp = data['timestamp'].values
    duration = data['duration'].values
    
    # 1. Heatmap
    ax1 = plt.subplot(2, 3, 1)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[50, 50],
                                              range=[[0, 1920], [0, 1080]])
    heatmap_smooth = gaussian_filter(heatmap, sigma=2)
    im1 = ax1.imshow(heatmap_smooth.T, origin='lower', cmap='hot', aspect='auto',
                     extent=[0, 1920, 0, 1080])
    ax1.set_title('Gaze Heatmap', fontweight='bold')
    ax1.set_xlabel('X Position (px)')
    ax1.set_ylabel('Y Position (px)')
    plt.colorbar(im1, ax=ax1, label='Density')
    
    # 2. Scan Path
    ax2 = plt.subplot(2, 3, 2)
    colors = np.linspace(0, 1, len(x))
    scatter = ax2.scatter(x, y, c=colors, cmap='viridis', s=5, alpha=0.6)
    ax2.plot(x, y, 'gray', alpha=0.2, linewidth=0.5)
    ax2.scatter(x[0], y[0], c='green', s=200, marker='*', label='Start', zorder=5)
    ax2.scatter(x[-1], y[-1], c='red', s=200, marker='*', label='End', zorder=5)
    ax2.set_title('Scan Path', fontweight='bold')
    ax2.set_xlabel('X Position (px)')
    ax2.set_ylabel('Y Position (px)')
    ax2.set_xlim(0, 1920)
    ax2.set_ylim(1080, 0)
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Time Progress')
    
    # 3. Fixation Duration Distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(duration, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(duration.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {duration.mean():.1f}ms')
    ax3.set_title('Fixation Duration Distribution', fontweight='bold')
    ax3.set_xlabel('Duration (ms)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. X Position Over Time
    ax4 = plt.subplot(2, 3, 4)
    time_sec = timestamp / 1000  # Convert to seconds
    ax4.plot(time_sec, x, color='blue', alpha=0.7, linewidth=1)
    ax4.set_title('X Position Over Time', fontweight='bold')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('X Position (px)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Y Position Over Time
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(time_sec, y, color='red', alpha=0.7, linewidth=1)
    ax5.set_title('Y Position Over Time', fontweight='bold')
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Y Position (px)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    📊 DATA SUMMARY
    
    Total Points: {len(data):,}
    Duration: {timestamp.max()/1000:.1f} seconds
    
    📍 POSITION
    X Range: {x.min():.0f} - {x.max():.0f} px
    Y Range: {y.min():.0f} - {y.max():.0f} px
    
    ⏱️ FIXATIONS
    Mean Duration: {duration.mean():.1f} ms
    Median Duration: {np.median(duration):.1f} ms
    Total Fixation Time: {duration.sum()/1000:.1f} seconds
    
    👤 PARTICIPANT
    Age: {info['Age']} years
    Gender: {info['Gender']}
    CARS Score: {info['CARS Score']}
    Classification: {info['Class']}
    """
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_fig:
        filename = f'participant_{participant_id}_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Saved figure as {filename}")
    
    plt.show()


def compare_participants(participant_ids, save_fig=False):
    """Compare multiple participants side by side."""
    
    loader = AutismDataLoader()
    
    fig, axes = plt.subplots(len(participant_ids), 3, figsize=(15, 4*len(participant_ids)))
    fig.suptitle('Autism Eye-Tracking Comparison', fontsize=16, fontweight='bold')
    
    for idx, pid in enumerate(participant_ids):
        print(f"Loading Participant {pid}...")
        data = loader.load_participant_data(pid)
        info = loader.get_participant_info(pid)
        
        x = data['x'].values
        y = data['y'].values
        duration = data['duration'].values
        
        # Row title
        row_label = f"P{pid} (Age:{info['Age']}, CARS:{info['CARS Score']})"
        axes[idx, 0].text(-0.3, 0.5, row_label, transform=axes[idx, 0].transAxes,
                         fontsize=12, fontweight='bold', rotation=90,
                         verticalalignment='center')
        
        # Heatmap
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[50, 50],
                                                  range=[[0, 1920], [0, 1080]])
        heatmap_smooth = gaussian_filter(heatmap, sigma=2)
        axes[idx, 0].imshow(heatmap_smooth.T, origin='lower', cmap='hot',
                           aspect='auto', extent=[0, 1920, 0, 1080])
        axes[idx, 0].set_title('Heatmap')
        
        # Scan path
        colors = np.linspace(0, 1, len(x))
        axes[idx, 1].scatter(x, y, c=colors, cmap='viridis', s=2, alpha=0.5)
        axes[idx, 1].set_xlim(0, 1920)
        axes[idx, 1].set_ylim(1080, 0)
        axes[idx, 1].set_title('Scan Path')
        
        # Duration distribution
        axes[idx, 2].hist(duration, bins=30, color='steelblue', alpha=0.7)
        axes[idx, 2].axvline(duration.mean(), color='red', linestyle='--')
        axes[idx, 2].set_title(f'Duration (μ={duration.mean():.1f}ms)')
    
    plt.tight_layout()
    
    if save_fig:
        filename = 'participant_comparison.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Saved comparison as {filename}")
    
    plt.show()


if __name__ == '__main__':
    print("=" * 70)
    print("Lightweight Autism Data Visualizer")
    print("=" * 70)
    
    # Get available participants
    loader = AutismDataLoader()
    participants = loader.get_available_participants()
    print(f"\nAvailable participants: {participants}")
    
    # Single participant visualization
    print("\n--- Visualizing Participant 2 ---")
    visualize_participant(2, save_fig=True)
    
    # Compare multiple participants
    print("\n--- Comparing Participants 1, 2, 3 ---")
    compare_participants([1, 2, 3], save_fig=True)
    
    print("\n" + "=" * 70)
    print("✅ Visualization complete!")
    print("=" * 70)
