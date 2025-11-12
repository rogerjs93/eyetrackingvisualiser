# Eye-Tracking Data Visualizer

A comprehensive research-grade tool for visualizing and analyzing eye-tracking data with AI-powered pattern recognition, statistical comparison, cognitive load analysis, and autism baseline model comparison.

## 🚀 Quick Start

### Option 1: Web Demo (Limited Features)
**[Try it online here!](https://rogerjs93.github.io/eyetrackingvisualiser/)**  
Browser-based demo with synthetic data generation and basic visualizations.

### Option 2: Full Python Version (Recommended)
```bash
# Clone repository
git clone https://github.com/rogerjs93/eyetrackingvisualiser.git
cd eyetrackingvisualiser

# Install dependencies
pip install -r requirements.txt

# Run interactive dashboard
python interactive_dashboard.py
```
Then open: **http://localhost:8050**

## ✨ Features

### 🌐 Web Version (GitHub Pages)
- ✅ Synthetic data generation with 6 pattern types
- ✅ Core visualizations (heatmap, scan path, distributions)
- ✅ Real-time interactive plots with Plotly
- ❌ No real dataset support
- ❌ No ML/AI features
- ❌ No baseline model comparison

### 🐍 Python Version (Full Features)
All web features PLUS:
- ✅ **Real Autism Dataset**: 25 ASD participants (ages 2.7-11.7 years)
- ✅ **AI Baseline Model**: TensorFlow-trained autism gaze pattern model
- ✅ **Similarity Scoring**: Compare new data against baseline (0-100 score)
- ✅ **Pattern Recognition**: ML-powered reading, expertise, confusion detection
- ✅ **Cognitive Load Analysis**: Entropy, fixation rates, task difficulty
- ✅ **Advanced Visualizations**: Sankey diagrams, network graphs, 4D plots
- ✅ **Methodology Explanations**: Built-in documentation for all metrics

## 📊 Core Visualization Types

1. **Gaze Heatmap** - Shows intensity of fixations across the screen
2. **Scan Path** - Displays gaze trajectory with temporal gradient
3. **Fixation Duration Distribution** - Histogram of how long fixations last
4. **Spatial Distribution** - Scatter plot with marginal distributions
5. **Temporal Pattern** - Time-series view of X and Y positions
6. **Attention Zones** - Identifies high-density areas with contours
7. **Density Contours** - Statistical confidence regions for gaze patterns

## 🧠 Autism Baseline Model (Python Only)

**NEW**: Compare eye-tracking data against a trained baseline model!

### What It Does
- Trained on 23 ASD participants using TensorFlow autoencoder
- Extracts 28 features (spatial, temporal, movement, distribution)
- Provides similarity scores (0-100) and deviation analysis
- Identifies which features deviate most from baseline
- Generates detailed comparison reports

### Usage
```python
from baseline_comparator import BaselineComparator

# Load baseline model
comparator = BaselineComparator(model_dir='models/baseline')

# Compare your data
results = comparator.compare_to_baseline(your_data)
print(f"Similarity: {results['similarity_score']}/100")
print(f"Deviation: {results['deviation_level']}")

# Generate report
comparator.generate_comparison_report(your_data, 'report.md')
```

See [`BASELINE_MODEL_SUMMARY.md`](BASELINE_MODEL_SUMMARY.md) for details.

## 🤖 AI-Powered Pattern Recognition (Python Only)

Automatically detect and classify viewing behaviors:
- **Reading Detection** - Identifies left-to-right reading patterns with return sweeps
- **Expertise Classification** - Classifies viewers as novice/intermediate/expert based on path efficiency
- **AOI Detection** - Automatically finds Areas of Interest using DBSCAN clustering
- **Confusion Indicators** - Detects signs of cognitive difficulty (revisits, erratic movements)
- **Narrative Insights** - Generates human-readable analysis of viewing behavior

```python
from pattern_recognition import GazePatternRecognizer

recognizer = GazePatternRecognizer(data)
behavior = recognizer.detect_reading_behavior()
expertise = recognizer.classify_expertise_level()
insights = recognizer.get_narrative_insights()
```

### 📊 Comparative Analysis

Rigorous statistical comparison of multiple sessions:
- **Two-Sample Comparison** - T-tests, KS tests, Hausdorff distance
- **Group Analysis** - ANOVA across multiple sessions
- **A/B Testing** - Effect size calculations (Cohen's d)
- **Consistency Scoring** - Measure pattern reproducibility
- **Outlier Detection** - IQR and Z-score based anomaly detection

```python
from comparative_analysis import ComparativeAnalyzer

analyzer = ComparativeAnalyzer()
results = analyzer.compare_two_sessions(session1, session2)
group_results = analyzer.group_analysis([s1, s2, s3, s4])
ab_test = analyzer.ab_testing(control_group, treatment_group)
```

### 🧠 Cognitive Load Analysis

Measure mental effort and task difficulty:
- **Spatial Entropy** - Attention distribution across screen
- **Fixation Rate Analysis** - Processing speed indicators
- **Saccade Metrics** - Visual search efficiency
- **Ambient/Focal Attention** - Attention mode classification
- **Gaze Transition Entropy** - Pattern predictability
- **Task Difficulty Score** - Overall cognitive load composite

```python
from cognitive_load import CognitiveLoadAnalyzer

analyzer = CognitiveLoadAnalyzer(data)
entropy = analyzer.calculate_spatial_entropy()
difficulty = analyzer.measure_task_difficulty()
report = analyzer.generate_cognitive_load_report()
```

### 🎨 Advanced Visualizations

Publication-ready unique visualizations:
- **Sankey Diagrams** - Gaze flow between screen regions
- **Network Graphs** - AOI relationships and transitions
- **4D Visualization** - X, Y, Time, Duration in single plot
- **Animated Scan Paths** - Replay viewing behavior over time
- **Velocity Heatmaps** - Eye movement speed across regions
- **Comparison Dashboards** - Side-by-side multi-session analysis

```python
from advanced_visualizations import AdvancedVisualizer

viz = AdvancedVisualizer(data)
fig1 = viz.create_sankey_diagram()
fig2 = viz.create_network_graph()
fig3 = viz.create_4d_visualization()
viz.export_interactive_html('report.html')
```

## Installation

### Prerequisites

- Python 3.7 or higher

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy - Numerical computations
- pandas - Data handling
- matplotlib - Core visualization
- seaborn - Statistical plotting
- scipy - Advanced statistics and signal processing
- scikit-learn - Machine learning algorithms
- networkx - Graph analysis

## Quick Start

### Option 1: Web Version (No Installation)

Simply open **[https://rogerjs93.github.io/eyetrackingvisualiser/](https://rogerjs93.github.io/eyetrackingvisualiser/)** in your browser!

The web version includes:
- 6 different eye-tracking patterns
- Interactive visualizations with zoom/pan
- Real-time statistics
- All processing done in your browser

### Option 2: Python Version (Local)

### Run Demo with Sample Data

```bash
python sample_data_generator.py
```

This generates synthetic eye-tracking data with a "natural" viewing pattern and creates a comprehensive dashboard.

### Try Different Patterns

```bash
# Natural viewing with clusters
python sample_data_generator.py natural

# Reading pattern (left to right)
python sample_data_generator.py reading

# Center-focused viewing
python sample_data_generator.py centered

# Random scattered fixations
python sample_data_generator.py scattered

# F-pattern (web browsing)
python sample_data_generator.py f_pattern

# Z-pattern (simple layouts)
python sample_data_generator.py z_pattern

# Generate all patterns
python sample_data_generator.py all
```

## Using with Your Own Data

### Data Format

Your eye-tracking data should be in one of these formats:

1. **Pandas DataFrame**
2. **Python Dictionary**
3. **CSV file** (load with pandas)

### Required Columns

- `x` - X coordinate (pixels)
- `y` - Y coordinate (pixels)
- `timestamp` - Time in milliseconds (optional but recommended)
- `duration` - Fixation duration in milliseconds (optional but recommended)

### Example Usage

```python
import pandas as pd
from eyetracking_visualizer import EyeTrackingVisualizer

# Load your data
data = pd.read_csv('your_eyetracking_data.csv')

# Create visualizer instance
viz = EyeTrackingVisualizer(
    data=data,
    screen_width=1920,  # Your screen width
    screen_height=1080  # Your screen height
)

# Generate comprehensive dashboard
viz.create_comprehensive_dashboard(save_path='my_analysis.png')

# Generate text report
viz.generate_report(save_path='my_report.txt')
```

### Loading from Dictionary

```python
data = {
    'x': [100, 150, 200, 250, 300],
    'y': [200, 220, 240, 260, 280],
    'timestamp': [0, 150, 300, 450, 600],
    'duration': [150, 150, 150, 150, 150]
}

viz = EyeTrackingVisualizer(data, screen_width=1920, screen_height=1080)
viz.create_comprehensive_dashboard()
```

### Loading from CSV

```python
import pandas as pd

# Assuming your CSV has columns: x, y, timestamp, duration
data = pd.read_csv('eyetracking_data.csv')

viz = EyeTrackingVisualizer(data, screen_width=1920, screen_height=1080)
viz.create_comprehensive_dashboard(save_path='dashboard.png')
```

## Output Files

### Dashboard Image
- High-resolution PNG (300 DPI)
- Contains all 7 visualization types
- Suitable for reports and presentations

### Text Report
- Statistical summaries
- Basic statistics (mean, std, range)
- Duration statistics (if available)
- Spatial coverage metrics
- Temporal information (if available)

## Understanding the Visualizations

### 1. Gaze Heatmap
- **Hot colors (red/yellow)** = High fixation density
- **Cool colors (blue/black)** = Low fixation density
- Shows "where people look most"

### 2. Scan Path
- **Green marker** = Starting point
- **Red marker** = Ending point
- **Color gradient** = Temporal progression (purple → yellow)
- Shows "how the gaze moved over time"

### 3. Fixation Duration Distribution
- Histogram showing how long each fixation lasted
- **Red line** = Mean duration
- **Green line** = Median duration
- Helps identify typical fixation patterns

### 4. Spatial Distribution
- Main scatter plot shows all fixations
- **Top histogram** = X-axis distribution
- **Right histogram** = Y-axis distribution
- Shows spatial spread of attention

### 5. Temporal Pattern
- **Blue line** = X position over time
- **Red line** = Y position over time
- Shows movement patterns and stability

### 6. Attention Zones
- Heatmap with contour lines
- Contour lines separate different attention intensity levels
- Helps identify "Areas of Interest" (AOIs)

### 7. Density Contours
- Scatter plot colored by point density
- **Red ellipse** = 95% confidence region
- **Orange ellipse** = 68% confidence region
- Shows statistical distribution of gaze

## Interpreting Results for Qualitative Analysis

### Reading Patterns
- **F-pattern**: Horizontal bars at top, vertical on left (web pages)
- **Z-pattern**: Top horizontal, diagonal, bottom horizontal (simple layouts)
- **Linear**: Left-to-right progression (text reading)

### Attention Distribution
- **Concentrated**: Few high-density zones = focused attention
- **Distributed**: Many scattered fixations = exploration
- **Clustered**: Multiple distinct zones = multiple points of interest

### Temporal Analysis
- **Stable patterns**: Repeated visits to same areas
- **Sequential**: Progressive movement through content
- **Erratic**: Random jumps = confusion or search behavior

## Advanced Usage

### Custom Screen Sizes

```python
# For a 4K monitor
viz = EyeTrackingVisualizer(data, screen_width=3840, screen_height=2160)

# For a laptop
viz = EyeTrackingVisualizer(data, screen_width=1366, screen_height=768)
```

### Programmatic Access to Statistics

```python
# Access raw data
x_coords = viz.data['x'].values
y_coords = viz.data['y'].values

# Calculate custom metrics
import numpy as np
center_distance = np.sqrt((x_coords - 960)**2 + (y_coords - 540)**2)
mean_distance_from_center = np.mean(center_distance)
```

## Sample Data Patterns

The included sample data generator can create six different eye-tracking patterns:

1. **Natural** - Realistic viewing with multiple areas of interest
2. **Reading** - Left-to-right, line-by-line progression
3. **Centered** - Focus concentrated in the center
4. **Scattered** - Random distributed fixations
5. **F-pattern** - Common web browsing pattern
6. **Z-pattern** - Simple layout scanning pattern

## Troubleshooting

### Issue: "No data loaded"
**Solution**: Make sure to pass data when creating the visualizer or use `viz.load_data(data)`

### Issue: Missing visualizations
**Solution**: Ensure your data has `timestamp` and `duration` columns for temporal analyses

### Issue: Plots look cramped
**Solution**: Adjust screen_width and screen_height to match your actual screen dimensions

### Issue: Import errors
**Solution**: Run `pip install -r requirements.txt` to install all dependencies

## File Structure

```
Pythondata visualizer/
├── eyetracking_visualizer.py    # Main visualizer class
├── sample_data_generator.py     # Sample data generator with demos
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

## Examples of Use Cases

### UX Research
Analyze how users interact with your website or application interface

### Reading Studies
Understand reading patterns and comprehension strategies

### Advertising Analysis
Evaluate which elements capture attention in advertisements

### Accessibility Testing
Identify navigation patterns for interface improvements

### Cognitive Research
Study attention patterns in different cognitive tasks

## Dataset Attribution

The autism eye-tracking dataset used in this project is sourced from:

**Eye Tracking Autism Dataset**  
Published by: IMT Kaggle Team  
Available at: [Kaggle - Eye Tracking Autism](https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism)

This dataset contains eye-tracking data from 25 participants with autism spectrum disorder (ASD), ages 2.7-12.3 years, and has been invaluable for developing and testing the autism data analysis features in this visualizer.

## License

This project is open source and available for educational and research purposes.

## Contributing

Feel free to extend this tool with additional visualization types or analysis methods!

## Support

For questions or issues, refer to the code comments in `eyetracking_visualizer.py` for detailed documentation of each method.
