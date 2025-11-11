"""
Sample Eye-Tracking Data Generator
Generates synthetic eye-tracking data for testing and demonstration purposes.
"""

import numpy as np
import pandas as pd
from eyetracking_visualizer import EyeTrackingVisualizer


def generate_sample_eyetracking_data(n_points=500, screen_width=1920, screen_height=1080, 
                                     pattern='natural', seed=42):
    """
    Generate synthetic eye-tracking data with different patterns.
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
    screen_width : int
        Screen width in pixels
    screen_height : int
        Screen height in pixels
    pattern : str
        Pattern type: 'natural', 'reading', 'centered', 'scattered', 'f_pattern', 'z_pattern'
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: x, y, timestamp, duration
    """
    np.random.seed(seed)
    
    if pattern == 'natural':
        # Natural viewing with clusters and exploration
        data = _generate_natural_pattern(n_points, screen_width, screen_height)
        
    elif pattern == 'reading':
        # Reading pattern (left to right, top to bottom)
        data = _generate_reading_pattern(n_points, screen_width, screen_height)
        
    elif pattern == 'centered':
        # Center-focused viewing
        data = _generate_centered_pattern(n_points, screen_width, screen_height)
        
    elif pattern == 'scattered':
        # Random scattered fixations
        data = _generate_scattered_pattern(n_points, screen_width, screen_height)
        
    elif pattern == 'f_pattern':
        # F-pattern (common in web browsing)
        data = _generate_f_pattern(n_points, screen_width, screen_height)
        
    elif pattern == 'z_pattern':
        # Z-pattern (common in simple layouts)
        data = _generate_z_pattern(n_points, screen_width, screen_height)
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return data


def _generate_natural_pattern(n_points, screen_width, screen_height):
    """Generate natural viewing pattern with multiple areas of interest."""
    # Define 4-5 areas of interest (AOIs)
    aois = [
        (screen_width * 0.25, screen_height * 0.3),  # Upper left
        (screen_width * 0.75, screen_height * 0.3),  # Upper right
        (screen_width * 0.5, screen_height * 0.5),   # Center
        (screen_width * 0.35, screen_height * 0.7),  # Lower left
        (screen_width * 0.65, screen_height * 0.7),  # Lower right
    ]
    
    x_coords = []
    y_coords = []
    timestamps = []
    durations = []
    
    current_time = 0
    
    for i in range(n_points):
        # Choose an AOI (with some randomness)
        aoi_idx = np.random.choice(len(aois), p=[0.25, 0.25, 0.3, 0.1, 0.1])
        aoi_x, aoi_y = aois[aoi_idx]
        
        # Add Gaussian noise around the AOI
        x = np.clip(np.random.normal(aoi_x, screen_width * 0.08), 0, screen_width)
        y = np.clip(np.random.normal(aoi_y, screen_height * 0.08), 0, screen_height)
        
        # Generate fixation duration (100-400ms typical)
        duration = np.random.gamma(shape=2, scale=100)
        duration = np.clip(duration, 50, 800)
        
        x_coords.append(x)
        y_coords.append(y)
        timestamps.append(current_time)
        durations.append(duration)
        
        current_time += duration
    
    return pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'timestamp': timestamps,
        'duration': durations
    })


def _generate_reading_pattern(n_points, screen_width, screen_height):
    """Generate reading pattern (left to right, line by line)."""
    x_coords = []
    y_coords = []
    timestamps = []
    durations = []
    
    current_time = 0
    line_height = screen_height * 0.05
    start_x = screen_width * 0.1
    end_x = screen_width * 0.9
    start_y = screen_height * 0.2
    
    current_y = start_y
    
    for i in range(n_points):
        # Progress along the line
        progress = (i % 20) / 20
        x = start_x + progress * (end_x - start_x)
        x += np.random.normal(0, 20)  # Add some noise
        
        y = current_y + np.random.normal(0, 10)
        
        # Move to next line
        if i % 20 == 19:
            current_y += line_height
            if current_y > screen_height * 0.8:
                current_y = start_y
        
        x = np.clip(x, 0, screen_width)
        y = np.clip(y, 0, screen_height)
        
        duration = np.random.gamma(shape=2, scale=120)
        duration = np.clip(duration, 80, 500)
        
        x_coords.append(x)
        y_coords.append(y)
        timestamps.append(current_time)
        durations.append(duration)
        
        current_time += duration
    
    return pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'timestamp': timestamps,
        'duration': durations
    })


def _generate_centered_pattern(n_points, screen_width, screen_height):
    """Generate center-focused viewing pattern."""
    center_x = screen_width / 2
    center_y = screen_height / 2
    
    x_coords = []
    y_coords = []
    timestamps = []
    durations = []
    
    current_time = 0
    
    for i in range(n_points):
        # Concentrate around center with decreasing density outward
        radius = np.random.rayleigh(scale=min(screen_width, screen_height) * 0.15)
        angle = np.random.uniform(0, 2 * np.pi)
        
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        x = np.clip(x, 0, screen_width)
        y = np.clip(y, 0, screen_height)
        
        duration = np.random.gamma(shape=2, scale=150)
        duration = np.clip(duration, 100, 600)
        
        x_coords.append(x)
        y_coords.append(y)
        timestamps.append(current_time)
        durations.append(duration)
        
        current_time += duration
    
    return pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'timestamp': timestamps,
        'duration': durations
    })


def _generate_scattered_pattern(n_points, screen_width, screen_height):
    """Generate scattered random fixations."""
    x_coords = np.random.uniform(screen_width * 0.1, screen_width * 0.9, n_points)
    y_coords = np.random.uniform(screen_height * 0.1, screen_height * 0.9, n_points)
    
    durations = np.random.gamma(shape=2, scale=120, size=n_points)
    durations = np.clip(durations, 50, 700)
    
    timestamps = np.cumsum(durations)
    
    return pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'timestamp': timestamps,
        'duration': durations
    })


def _generate_f_pattern(n_points, screen_width, screen_height):
    """Generate F-pattern (common in web page viewing)."""
    x_coords = []
    y_coords = []
    timestamps = []
    durations = []
    
    current_time = 0
    
    # Top horizontal bar
    n_top = n_points // 3
    for i in range(n_top):
        x = screen_width * 0.1 + (screen_width * 0.8) * (i / n_top)
        y = screen_height * 0.2
        x += np.random.normal(0, 30)
        y += np.random.normal(0, 20)
        
        x = np.clip(x, 0, screen_width)
        y = np.clip(y, 0, screen_height)
        
        duration = np.random.gamma(shape=2, scale=100)
        duration = np.clip(duration, 60, 400)
        
        x_coords.append(x)
        y_coords.append(y)
        timestamps.append(current_time)
        durations.append(duration)
        current_time += duration
    
    # Middle horizontal bar (shorter)
    n_mid = n_points // 4
    for i in range(n_mid):
        x = screen_width * 0.1 + (screen_width * 0.5) * (i / n_mid)
        y = screen_height * 0.45
        x += np.random.normal(0, 30)
        y += np.random.normal(0, 20)
        
        x = np.clip(x, 0, screen_width)
        y = np.clip(y, 0, screen_height)
        
        duration = np.random.gamma(shape=2, scale=100)
        duration = np.clip(duration, 60, 400)
        
        x_coords.append(x)
        y_coords.append(y)
        timestamps.append(current_time)
        durations.append(duration)
        current_time += duration
    
    # Vertical bar (left side)
    n_vert = n_points - n_top - n_mid
    for i in range(n_vert):
        x = screen_width * 0.1
        y = screen_height * 0.2 + (screen_height * 0.6) * (i / n_vert)
        x += np.random.normal(0, 30)
        y += np.random.normal(0, 30)
        
        x = np.clip(x, 0, screen_width)
        y = np.clip(y, 0, screen_height)
        
        duration = np.random.gamma(shape=2, scale=100)
        duration = np.clip(duration, 60, 400)
        
        x_coords.append(x)
        y_coords.append(y)
        timestamps.append(current_time)
        durations.append(duration)
        current_time += duration
    
    return pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'timestamp': timestamps,
        'duration': durations
    })


def _generate_z_pattern(n_points, screen_width, screen_height):
    """Generate Z-pattern viewing."""
    x_coords = []
    y_coords = []
    timestamps = []
    durations = []
    
    current_time = 0
    
    # Divide into 3 segments
    n_per_segment = n_points // 3
    
    # Top horizontal (left to right)
    for i in range(n_per_segment):
        x = screen_width * 0.1 + (screen_width * 0.8) * (i / n_per_segment)
        y = screen_height * 0.2
        x += np.random.normal(0, 30)
        y += np.random.normal(0, 20)
        
        x = np.clip(x, 0, screen_width)
        y = np.clip(y, 0, screen_height)
        
        duration = np.random.gamma(shape=2, scale=110)
        duration = np.clip(duration, 70, 450)
        
        x_coords.append(x)
        y_coords.append(y)
        timestamps.append(current_time)
        durations.append(duration)
        current_time += duration
    
    # Diagonal (top-right to bottom-left)
    for i in range(n_per_segment):
        progress = i / n_per_segment
        x = screen_width * 0.9 - (screen_width * 0.8) * progress
        y = screen_height * 0.2 + (screen_height * 0.6) * progress
        x += np.random.normal(0, 30)
        y += np.random.normal(0, 30)
        
        x = np.clip(x, 0, screen_width)
        y = np.clip(y, 0, screen_height)
        
        duration = np.random.gamma(shape=2, scale=110)
        duration = np.clip(duration, 70, 450)
        
        x_coords.append(x)
        y_coords.append(y)
        timestamps.append(current_time)
        durations.append(duration)
        current_time += duration
    
    # Bottom horizontal (left to right)
    for i in range(n_points - 2 * n_per_segment):
        x = screen_width * 0.1 + (screen_width * 0.8) * (i / n_per_segment)
        y = screen_height * 0.8
        x += np.random.normal(0, 30)
        y += np.random.normal(0, 20)
        
        x = np.clip(x, 0, screen_width)
        y = np.clip(y, 0, screen_height)
        
        duration = np.random.gamma(shape=2, scale=110)
        duration = np.clip(duration, 70, 450)
        
        x_coords.append(x)
        y_coords.append(y)
        timestamps.append(current_time)
        durations.append(duration)
        current_time += duration
    
    return pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'timestamp': timestamps,
        'duration': durations
    })


def demo_all_patterns():
    """Generate and visualize all available patterns."""
    patterns = ['natural', 'reading', 'centered', 'scattered', 'f_pattern', 'z_pattern']
    
    print("Generating eye-tracking data for all patterns...\n")
    
    for pattern in patterns:
        print(f"Processing: {pattern.upper()} pattern")
        print("-" * 60)
        
        # Generate data
        data = generate_sample_eyetracking_data(
            n_points=500, 
            screen_width=1920, 
            screen_height=1080,
            pattern=pattern,
            seed=42
        )
        
        # Create visualizer
        viz = EyeTrackingVisualizer(data, screen_width=1920, screen_height=1080)
        
        # Generate dashboard
        viz.create_comprehensive_dashboard(
            save_path=f'dashboard_{pattern}.png'
        )
        
        # Generate report
        viz.generate_report(save_path=f'report_{pattern}.txt')
        
        print()


def demo_single_pattern(pattern='natural'):
    """Demo with a single pattern."""
    print(f"Generating {pattern.upper()} eye-tracking pattern...\n")
    
    # Generate data
    data = generate_sample_eyetracking_data(
        n_points=500, 
        screen_width=1920, 
        screen_height=1080,
        pattern=pattern,
        seed=42
    )
    
    print(f"Generated {len(data)} data points")
    print("\nData preview:")
    print(data.head(10))
    print("\n")
    
    # Create visualizer
    viz = EyeTrackingVisualizer(data, screen_width=1920, screen_height=1080)
    
    # Generate dashboard
    print("Creating comprehensive dashboard...")
    viz.create_comprehensive_dashboard(save_path=f'dashboard_{pattern}.png')
    
    # Generate report
    print("\nGenerating analysis report...")
    viz.generate_report(save_path=f'report_{pattern}.txt')
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print(f"- Dashboard saved as: dashboard_{pattern}.png")
    print(f"- Report saved as: report_{pattern}.txt")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        if pattern == 'all':
            demo_all_patterns()
        else:
            demo_single_pattern(pattern)
    else:
        # Default demo
        demo_single_pattern('natural')
        
        print("\n\nTo try other patterns, run:")
        print("  python sample_data_generator.py reading")
        print("  python sample_data_generator.py centered")
        print("  python sample_data_generator.py scattered")
        print("  python sample_data_generator.py f_pattern")
        print("  python sample_data_generator.py z_pattern")
        print("  python sample_data_generator.py all  (generates all patterns)")
