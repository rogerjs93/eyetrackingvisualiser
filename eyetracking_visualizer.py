"""
Eye-Tracking Data Visualizer
A comprehensive tool for visualizing eye-tracking data with various distributions and patterns
for qualitative analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.spatial import ConvexHull
import pandas as pd


class EyeTrackingVisualizer:
    """
    Visualizer for eye-tracking data with multiple visualization types
    for qualitative analysis.
    """
    
    def __init__(self, data=None, screen_width=1920, screen_height=1080):
        """
        Initialize the visualizer with eye-tracking data.
        
        Parameters:
        -----------
        data : pandas.DataFrame or dict
            Eye-tracking data with columns: 'x', 'y', 'timestamp', 'duration' (optional)
        screen_width : int
            Screen width in pixels
        screen_height : int
            Screen height in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        if data is not None:
            if isinstance(data, dict):
                self.data = pd.DataFrame(data)
            else:
                self.data = data
        else:
            self.data = None
            
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
    def load_data(self, data):
        """Load eye-tracking data."""
        if isinstance(data, dict):
            self.data = pd.DataFrame(data)
        else:
            self.data = data
            
    def create_comprehensive_dashboard(self, save_path=None):
        """
        Create a comprehensive dashboard with all visualizations.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_heatmap(ax1)
        
        # 2. Scatter plot with trajectory
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_scan_path(ax2)
        
        # 3. Fixation duration distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_duration_distribution(ax3)
        
        # 4. Spatial distribution (X-Y marginals)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_spatial_distribution(ax4)
        
        # 5. Temporal pattern
        ax5 = fig.add_subplot(gs[1, 2:])
        self._plot_temporal_pattern(ax5)
        
        # 6. Attention zones
        ax6 = fig.add_subplot(gs[2, :2])
        self._plot_attention_zones(ax6)
        
        # 7. Density contours
        ax7 = fig.add_subplot(gs[2, 2:])
        self._plot_density_contours(ax7)
        
        plt.suptitle('Eye-Tracking Data Analysis Dashboard', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
    def _plot_heatmap(self, ax):
        """Plot heatmap of gaze points."""
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            x, y, bins=[50, 50],
            range=[[0, self.screen_width], [0, self.screen_height]]
        )
        
        extent = [0, self.screen_width, self.screen_height, 0]
        im = ax.imshow(heatmap.T, extent=extent, origin='upper', 
                       cmap='hot', aspect='auto', interpolation='gaussian')
        
        ax.set_title('Gaze Heatmap', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        plt.colorbar(im, ax=ax, label='Fixation Density')
        
    def _plot_scan_path(self, ax):
        """Plot scan path showing gaze trajectory."""
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Plot trajectory with color gradient
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Color by time
        colors = np.linspace(0, 1, len(x))
        
        # Plot line with gradient
        for i in range(len(x) - 1):
            ax.plot(x[i:i+2], y[i:i+2], 'o-', 
                   color=plt.cm.viridis(colors[i]), 
                   alpha=0.6, markersize=3, linewidth=1)
        
        # Add start and end markers
        ax.plot(x[0], y[0], 'go', markersize=15, label='Start', 
               markeredgecolor='white', markeredgewidth=2)
        ax.plot(x[-1], y[-1], 'ro', markersize=15, label='End',
               markeredgecolor='white', markeredgewidth=2)
        
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)
        ax.set_title('Scan Path (Gaze Trajectory)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    def _plot_duration_distribution(self, ax):
        """Plot distribution of fixation durations."""
        if 'duration' in self.data.columns:
            durations = self.data['duration'].values
            
            ax.hist(durations, bins=30, color='steelblue', 
                   alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(durations), color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {np.mean(durations):.1f}ms')
            ax.axvline(np.median(durations), color='green', 
                      linestyle='--', linewidth=2, label=f'Median: {np.median(durations):.1f}ms')
            
            ax.set_title('Fixation Duration Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Duration (ms)')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No duration data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Fixation Duration Distribution', fontsize=12, fontweight='bold')
        
    def _plot_spatial_distribution(self, ax):
        """Plot spatial distribution with marginal histograms."""
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Create scatter plot
        ax.scatter(x, y, alpha=0.3, s=20, color='purple')
        
        # Add marginal distributions
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
        ax_histy = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)
        
        ax_histx.hist(x, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax_histy.hist(y, bins=30, orientation='horizontal', 
                     color='lightcoral', alpha=0.7, edgecolor='black')
        
        ax_histx.tick_params(labelbottom=False)
        ax_histy.tick_params(labelleft=False)
        ax_histx.set_ylabel('Count')
        ax_histy.set_xlabel('Count')
        
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)
        ax.set_title('Spatial Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.grid(True, alpha=0.3)
        
    def _plot_temporal_pattern(self, ax):
        """Plot temporal pattern of gaze positions."""
        if 'timestamp' in self.data.columns:
            time = self.data['timestamp'].values
            x = self.data['x'].values
            y = self.data['y'].values
            
            ax2 = ax.twinx()
            
            ax.plot(time, x, 'b-', alpha=0.7, linewidth=1, label='X Position')
            ax2.plot(time, y, 'r-', alpha=0.7, linewidth=1, label='Y Position')
            
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('X Position (pixels)', color='b')
            ax2.set_ylabel('Y Position (pixels)', color='r')
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            
            ax.set_title('Temporal Gaze Pattern', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No timestamp data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Temporal Gaze Pattern', fontsize=12, fontweight='bold')
    
    def _plot_attention_zones(self, ax):
        """Plot attention zones using clustering."""
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Create grid and calculate density
        from scipy.ndimage import gaussian_filter
        
        heatmap, xedges, yedges = np.histogram2d(
            x, y, bins=[40, 40],
            range=[[0, self.screen_width], [0, self.screen_height]]
        )
        
        # Smooth the heatmap
        heatmap_smooth = gaussian_filter(heatmap, sigma=2)
        
        # Plot with contours
        extent = [0, self.screen_width, self.screen_height, 0]
        im = ax.imshow(heatmap_smooth.T, extent=extent, origin='upper', 
                      cmap='YlOrRd', aspect='auto', alpha=0.6)
        
        # Add contour lines
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        contours = ax.contour(X, Y, heatmap_smooth.T, levels=5, 
                             colors='black', linewidths=1.5, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8)
        
        # Scatter overlay
        ax.scatter(x, y, c='blue', alpha=0.1, s=10)
        
        ax.set_title('Attention Zones (High Density Areas)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        plt.colorbar(im, ax=ax, label='Attention Density')
        
    def _plot_density_contours(self, ax):
        """Plot density contours with statistical regions."""
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Calculate 2D kernel density
        from scipy.stats import gaussian_kde
        
        # Prepare data
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # Sort by density
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = x[idx], y[idx], z[idx]
        
        # Create scatter plot with density coloring
        scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, 
                           s=30, cmap='plasma', alpha=0.6)
        
        # Add confidence ellipse
        self._plot_confidence_ellipse(x, y, ax, n_std=2.0, 
                                     edgecolor='red', linewidth=2,
                                     label='95% Confidence')
        self._plot_confidence_ellipse(x, y, ax, n_std=1.0, 
                                     edgecolor='orange', linewidth=2,
                                     label='68% Confidence')
        
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(self.screen_height, 0)
        ax.set_title('Density Distribution with Confidence Regions', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        plt.colorbar(scatter, ax=ax, label='Point Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_confidence_ellipse(self, x, y, ax, n_std=2.0, **kwargs):
        """Plot confidence ellipse for data distribution."""
        mean_x, mean_y = np.mean(x), np.mean(y)
        cov = np.cov(x, y)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Width and height of ellipse
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        
        ellipse = Ellipse((mean_x, mean_y), width, height, 
                         angle=angle, fill=False, **kwargs)
        ax.add_patch(ellipse)
        
    def generate_report(self, save_path='eyetracking_report.txt'):
        """Generate a text report with key statistics."""
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        x = self.data['x'].values
        y = self.data['y'].values
        
        report = []
        report.append("=" * 60)
        report.append("EYE-TRACKING DATA ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS")
        report.append("-" * 60)
        report.append(f"Total fixation points: {len(self.data)}")
        report.append(f"X Position - Mean: {np.mean(x):.2f}, Std: {np.std(x):.2f}")
        report.append(f"Y Position - Mean: {np.mean(y):.2f}, Std: {np.std(y):.2f}")
        report.append("")
        
        # Duration statistics
        if 'duration' in self.data.columns:
            dur = self.data['duration'].values
            report.append("FIXATION DURATION STATISTICS")
            report.append("-" * 60)
            report.append(f"Mean duration: {np.mean(dur):.2f} ms")
            report.append(f"Median duration: {np.median(dur):.2f} ms")
            report.append(f"Min duration: {np.min(dur):.2f} ms")
            report.append(f"Max duration: {np.max(dur):.2f} ms")
            report.append(f"Total viewing time: {np.sum(dur):.2f} ms")
            report.append("")
        
        # Spatial coverage
        report.append("SPATIAL COVERAGE")
        report.append("-" * 60)
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        coverage = (x_range * y_range) / (self.screen_width * self.screen_height) * 100
        report.append(f"X Range: {x_range:.2f} pixels ({x_range/self.screen_width*100:.1f}% of screen)")
        report.append(f"Y Range: {y_range:.2f} pixels ({y_range/self.screen_height*100:.1f}% of screen)")
        report.append(f"Estimated coverage: {coverage:.2f}% of screen")
        report.append("")
        
        # Time statistics
        if 'timestamp' in self.data.columns:
            time = self.data['timestamp'].values
            report.append("TEMPORAL STATISTICS")
            report.append("-" * 60)
            report.append(f"Total recording time: {time[-1] - time[0]:.2f} ms")
            report.append(f"Average sampling rate: {len(time) / (time[-1] - time[0]) * 1000:.2f} Hz")
            report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to: {save_path}")
        
        return report_text


def main():
    """Example usage of the EyeTrackingVisualizer."""
    # This is an example - replace with your actual data loading
    print("Eye-Tracking Data Visualizer")
    print("=" * 60)
    print("\nTo use this visualizer with your data:")
    print("1. Load your eye-tracking data as a pandas DataFrame or dict")
    print("2. Ensure it has columns: 'x', 'y', 'timestamp', 'duration' (optional)")
    print("3. Create a visualizer instance and call create_comprehensive_dashboard()")
    print("\nExample:")
    print("  data = pd.read_csv('your_data.csv')")
    print("  viz = EyeTrackingVisualizer(data, screen_width=1920, screen_height=1080)")
    print("  viz.create_comprehensive_dashboard(save_path='dashboard.png')")
    print("  viz.generate_report(save_path='report.txt')")
    print("\nFor a demo with sample data, run: python sample_data_generator.py")
    

if __name__ == "__main__":
    main()
