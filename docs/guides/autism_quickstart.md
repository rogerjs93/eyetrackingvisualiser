# 🧩 Autism Eye-Tracking Data - Quick Start Guide

## ✅ What's Been Added

Your visualizer now supports **real autism eye-tracking data** from 25 participants!

### New Files Created:
1. **`autism_data_loader.py`** - Data loading and processing module
2. **`quick_autism_viz.py`** - Lightweight visualizer (works right now!)
3. **`test_autism_integration.py`** - Integration test script
4. **`AUTISM_DATA_README.md`** - Complete documentation

### Updated Files:
- **`interactive_dashboard.py`** - Now supports autism data selection

## 🚀 Quick Start (3 Options)

### Option 1: Lightweight Visualizer (RECOMMENDED - Works Now!)
This uses matplotlib instead of Dash, so it starts instantly:

```bash
python quick_autism_viz.py
```

This will:
- Show comprehensive analysis for Participant 2
- Compare Participants 1, 2, and 3 side-by-side
- Save PNG images of the visualizations
- **No heavy imports, starts immediately!**

### Option 2: Test the Data Loader
```bash
python test_autism_integration.py
```

This verifies that the autism data loads correctly and is compatible with all analysis features.

### Option 3: Interactive Dashboard (Full Features)
```bash
python interactive_dashboard.py
```

Then in the browser (http://localhost:8050):
1. Select **Data Source**: "🧩 Autism Dataset"
2. Choose a **Participant** (1-25)
3. Click **🔄 Generate Data**
4. Explore all analysis tabs with real autism data!

## 📊 Dataset Information

### Participants
- **Total**: 25 participants with Autism Spectrum Disorder (ASD)
- **Age Range**: 2.7 - 12.3 years
- **Gender**: Mixed (Male and Female)
- **CARS Scores**: 27.0 - 36.5 (Mild to Moderate autism)

### Data Quality
- **Points per session**: 15,000 - 100,000 fixations
- **Session duration**: 5-15 minutes
- **Sampling**: High-quality eye-tracker output
- **Format**: Processed to standard x, y, timestamp, duration format

### Example Participants:
| ID | Age | Gender | CARS | Data Points |
|----|-----|--------|------|-------------|
| 1  | 7.0 | M      | 32.5 | 22,864      |
| 2  | 8.9 | F      | 36.5 | 15,172      |
| 3  | 4.4 | M      | 27.0 | 102,066     |

## 💻 Usage Examples

### Visualize a Single Participant
```python
from autism_data_loader import AutismDataLoader

loader = AutismDataLoader()
data = loader.load_participant_data(2)
info = loader.get_participant_info(2)

print(f"Participant {info['ParticipantID']}")
print(f"Age: {info['Age']}, CARS: {info['CARS Score']}")
print(f"Total fixations: {len(data)}")
```

### Compare Multiple Participants
```python
# Load group data
data_dict = loader.load_group_comparison_data([1, 2, 3, 4, 5])

# Use with your existing comparative analysis
from comparative_analysis import ComparativeAnalyzer
analyzer = ComparativeAnalyzer(data_dict)
results = analyzer.compare_sessions()
```

### Run All Analyses on Autism Data
```python
from autism_data_loader import AutismDataLoader
from pattern_recognition import GazePatternRecognizer
from cognitive_load import CognitiveLoadAnalyzer
from advanced_visualizations import AdvancedVisualizer

# Load data
loader = AutismDataLoader()
data = loader.load_participant_data(5)

# Pattern recognition
patterns = GazePatternRecognizer(data)
reading = patterns.detect_reading_behavior()
expertise = patterns.classify_expertise_level()
confusion = patterns.detect_confusion_indicators()

# Cognitive load
cognitive = CognitiveLoadAnalyzer(data)
difficulty = cognitive.measure_task_difficulty()
entropy = cognitive.calculate_spatial_entropy()

# Advanced visualizations
viz = AdvancedVisualizer(data, 1920, 1080)
sankey = viz.create_sankey_diagram()
network = viz.create_network_graph()
```

## 🎯 What You Can Do Now

### Research Applications:
1. **Pattern Analysis**: Compare gaze patterns between participants with different CARS scores
2. **Cognitive Load**: Measure task difficulty for autism participants
3. **Developmental Studies**: Analyze differences across age groups (2.7 - 12.3 years)
4. **Gender Differences**: Compare male vs female gaze patterns
5. **Severity Correlation**: Study relationship between CARS scores and eye-tracking metrics

### Visual Outputs:
- ✅ Heatmaps showing attention distribution
- ✅ Scan paths with temporal progression
- ✅ Duration distributions and statistics
- ✅ Temporal analysis of gaze positions
- ✅ AI-detected confusion and expertise levels
- ✅ Cognitive load measurements
- ✅ Network graphs of attention flow
- ✅ 4D visualizations combining space and time

## 📁 Data Structure

```
data/autism/
├── Metadata_Participants.csv          # Participant info
│   ├── ParticipantID (1-25)
│   ├── Age (years)
│   ├── Gender (M/F)
│   ├── Class (ASD)
│   └── CARS Score (27-36.5)
│
└── Eye-tracking Output/               # Raw eye-tracking data
    ├── 1.csv                          # Participant 1 data
    ├── 2.csv                          # Participant 2 data
    └── ... (25 files total)
```

## 🔍 Data Processing Pipeline

The loader automatically:
1. ✅ **Averages** left and right eye positions
2. ✅ **Removes** blink events
3. ✅ **Validates** coordinates (within screen bounds)
4. ✅ **Calculates** fixation durations from timestamps
5. ✅ **Normalizes** timestamps to start at 0
6. ✅ **Filters** invalid/missing data points

## ⚠️ Notes

### Performance:
- Large datasets (100K+ points) take 5-10 seconds to load
- First load includes CSV parsing and processing
- Data is cached in memory after loading

### Data Warnings:
- You may see `DtypeWarning` - this is **safe to ignore**
- The loader handles mixed data types correctly
- All data is validated and cleaned automatically

### CARS Score Interpretation:
- **27-29**: Mild autism
- **30-36**: Moderate autism  
- **37+**: Severe autism (not in this dataset)

## 🎓 Next Steps

### Try These:
1. **Run the quick visualizer** to see immediate results
2. **Compare participants** with different CARS scores
3. **Analyze age-related patterns** (younger vs older)
4. **Study confusion indicators** in autism gaze patterns
5. **Measure cognitive load** across different participants
6. **Export publication-ready figures** for research

### Research Questions You Can Answer:
- Do higher CARS scores correlate with different gaze patterns?
- How does age affect visual exploration in autism?
- Are there gender differences in attention distribution?
- What cognitive load patterns are unique to autism?
- How do fixation durations differ from neurotypical patterns?

## 📞 Support

If you encounter issues:
1. Check that data files exist in `data/autism/`
2. Verify Python packages are installed (pandas, numpy, scipy, matplotlib)
3. Try the test script: `python test_autism_integration.py`
4. Use the lightweight visualizer if dashboard has import issues

## 🎉 Summary

✅ **25 real autism participants** ready to analyze  
✅ **All features work** with real data  
✅ **Quick visualizer** available for instant results  
✅ **Full documentation** provided  
✅ **Tested and validated** integration  
✅ **Research-ready** dataset with metadata  

**You can now explore real autism eye-tracking data with all the sophisticated analysis tools!**
