# Methodology Explanations Update Summary

## What Was Added

### 1. **METHODOLOGY.md** (Comprehensive Documentation)
A complete 500+ line markdown document containing:
- Detailed mathematical formulations for all metrics
- Weight justifications and threshold explanations
- Interpretation guidelines with concrete examples
- Academic references for all algorithms
- 14 sections covering all visualizations and analyses

### 2. **methodology_explanations.py** (Python Module)
A Python module providing:
- Dictionary of all methodology explanations
- HTML formatting functions for Dash app
- Markdown formatting for documentation
- Easy integration with interactive dashboard

### 3. **methodology_explanations.js** (JavaScript Module)
A JavaScript version providing:
- Browser-compatible explanation system
- HTML formatting for static website
- Dynamic insertion into web pages
- Consistent with Python version

### 4. **Updated Files**

#### interactive_dashboard.py
- Imported methodology_explanations module
- Added explanations to heatmap view
- Added explanations to scan path view
- Added 4 explanations to AI Patterns tab (reading, expertise, confusion, AOI)
- Added 5 explanations to Cognitive Load tab (entropy, fixation, saccade, attention, difficulty)

#### index.html
- Added script tag for methodology_explanations.js
- Integrated with existing visualizer.js

#### visualizer.js
- Added explanation insertion logic
- Updated analyzeAIPatterns() to show formulas
- Updated analyzeCognitiveLoad() to show formulas
- Added addExplanationsToUI() function for heatmap/scan path

## Features of the Explanation System

### 1. **Expandable Sections**
Each explanation has collapsible details:
- 📐 Mathematical Formulation (click to expand)
- 💡 Interpretation (click to expand)

### 2. **Complete Information**
Every metric includes:
- **Title**: Clear name of the metric
- **Description**: What it measures and why
- **Formulation**: Complete mathematical formula with notation
- **Interpretation**: How to understand the results
- **References**: Academic sources and citations

### 3. **Visual Design**
- Light blue accented boxes (#667eea)
- Clean, readable typography
- Properly formatted code blocks
- Consistent styling across Python and JavaScript

## Coverage

### Pattern Recognition (4 metrics)
✅ Reading Behavior Detection
✅ Expertise Level Classification  
✅ Areas of Interest (AOI) Detection
✅ Confusion Indicator Detection

### Cognitive Load (6 metrics)
✅ Spatial Entropy
✅ Fixation Rate Analysis
✅ Saccade Metrics
✅ Ambient vs Focal Attention
✅ Gaze Transition Entropy
✅ Task Difficulty Score

### Visualizations (4 types)
✅ Gaze Heatmap
✅ Scan Path Visualization
✅ Sankey Diagram (Gaze Flow)
✅ Velocity Heatmap

## Examples of What Users See

### Reading Behavior Explanation
```
ℹ️ Reading Behavior Detection

Description: Detects whether the user is reading, scanning, or exploring based on gaze movement patterns.

📐 Mathematical Formulation [click to expand]
Reading Score = (LTR_ratio × 0.4) + (return_sweeps/n × 0.3) + (regularity × 0.3)

Where:
- LTR_ratio = proportion of rightward movements (left-to-right reading)
- return_sweeps = count of large leftward jumps (>100px)
- regularity = 1 / (variance of angle changes + 0.01)

💡 Interpretation [click to expand]
The behavior with the highest score is selected:
- Reading: High left-to-right progression, regular return sweeps...
- Scanning: Quick fixations, high spatial dispersion...
- Exploring: High entropy (unpredictable), irregular movements...

References: Based on Rayner (1998) 'Eye movements in reading and information processing' and Holmqvist et al. (2011)
```

### Task Difficulty Explanation
```
ℹ️ Task Difficulty Score

Description: Composite measure of cognitive load from multiple indicators.

📐 Mathematical Formulation [click to expand]
Difficulty Score (0-10) = weighted_sum of:

1. Spatial Entropy (30%): entropy/4 × 3
2. Fixation Rate (20%): (rate/10) × 2
3. Mean Duration (20%): (duration/500) × 2
4. Saccade Length (20%): (length/200) × 2
5. Ambient Ratio (10%): ambient_ratio × 1

Normalized: min(sum, 10)

💡 Interpretation [click to expand]
Difficulty levels:
- Low (<4): Easy task, efficient processing
- Moderate (4-7): Normal cognitive load
- High (>7): Difficult task, high cognitive demand

Recommendations:
- High: Simplify interface, add guidance
- Moderate: Monitor user success
- Low: Appropriate difficulty level
```

## Benefits

### For Researchers
- Understand the mathematical foundations
- Validate methodology against literature
- Cite specific algorithms and parameters
- Replicate analyses

### For Users
- Learn what each metric means
- Understand why certain conclusions are drawn
- Make informed decisions about results
- Trust the analysis

### For Developers
- Centralized documentation
- Easy to maintain and update
- Consistent across platforms
- Extensible for new metrics

## Accessibility

- GitHub Pages: https://rogerjs93.github.io/eyetrackingvisualiser/
- Python Dashboard: Run `interactive_dashboard.py`
- Documentation: Read `METHODOLOGY.md`

## Academic Rigor

All formulas are based on peer-reviewed research:
- 15 academic references cited
- Industry-standard algorithms (DBSCAN, Shannon Entropy)
- Empirically validated thresholds
- Transparent methodology

---

**Commit**: 17ffa94
**Date**: 2024
**Status**: ✅ Pushed to GitHub
