# Autism Eye-Tracking Data Integration

## Overview
The visualizer now supports real autism eye-tracking data in addition to synthetic data generation!

## Data Location
The autism dataset is located in:
```
data/autism/
├── Metadata_Participants.csv      # Participant information (ID, Age, Gender, CARS Score)
└── Eye-tracking Output/           # Individual CSV files per participant
    ├── 1.csv
    ├── 2.csv
    └── ... (25 participants total)
```

## Features

### Participant Information
Each participant has:
- **ParticipantID**: Unique identifier (1-25)
- **Age**: Age in years
- **Gender**: M (Male) or F (Female)
- **Class**: Diagnosis classification (ASD)
- **CARS Score**: Childhood Autism Rating Scale score

### Eye-Tracking Data
Each CSV file contains:
- **Point of Regard**: Gaze coordinates (X, Y) in pixels
- **Timestamps**: Recording time in milliseconds
- **Event Types**: Fixation, Saccade, Blink
- **Duration**: Calculated fixation durations
- **Pupil Data**: Pupil size and diameter measurements

## Using the Dashboard

### 1. Start the Interactive Dashboard
```bash
python interactive_dashboard.py
```

### 2. Select Data Source
- In the control panel, select **Data Source**: "🧩 Autism Dataset"
- This will enable the **Participant** dropdown
- Select a participant (Participant 1-25)

### 3. Load Data
- Click **🔄 Generate Data** button
- The dashboard will load the real autism eye-tracking data
- Participant information will be displayed in the statistics panel

### 4. Analyze
All analysis features work with real data:
- ✅ Heatmaps and scan paths
- ✅ Distribution analysis
- ✅ Temporal analysis
- ✅ Attention zones
- ✅ AI Pattern Recognition
- ✅ Cognitive Load Analysis
- ✅ Advanced Visualizations

## Using the Data Loader Programmatically

```python
from autism_data_loader import AutismDataLoader

# Initialize loader
loader = AutismDataLoader()

# Get available participants
participants = loader.get_available_participants()
print(f"Found {len(participants)} participants")  # Output: Found 25 participants

# Load specific participant
participant_id = 2
data = loader.load_participant_data(participant_id)

# Get participant information
info = loader.get_participant_info(participant_id)
print(f"Age: {info['Age']}, Gender: {info['Gender']}, CARS: {info['CARS Score']}")

# Data is in standard format with columns: x, y, timestamp, duration, event_type
print(data.head())
```

## Data Format
The loader automatically converts raw eye-tracking data to the standard format:

| Column | Description | Type |
|--------|-------------|------|
| x | X coordinate in pixels | float |
| y | Y coordinate in pixels | float |
| timestamp | Time in milliseconds (normalized to 0) | float |
| duration | Fixation duration in ms | float |
| event_type | Fixation/Saccade/Blink | string |

## Data Processing
The loader performs:
1. **Binocular averaging**: Averages left and right eye positions
2. **Blink removal**: Filters out blink events
3. **Coordinate validation**: Ensures coordinates are within screen bounds
4. **Duration calculation**: Computes fixation durations from timestamps
5. **Timestamp normalization**: Sets first timestamp to 0

## Statistics

Dataset overview:
- **Total Participants**: 25
- **Age Range**: 2.7 - 12.3 years
- **Gender Distribution**: Mixed M/F
- **CARS Score Range**: 27.0 - 36.5
- **Average Session Duration**: ~5-10 minutes per participant
- **Average Data Points**: 15,000-100,000 per participant

## Comparative Analysis
You can compare autism participants:
```python
# Load multiple participants for group analysis
loader = AutismDataLoader()
data_dict = loader.load_group_comparison_data([1, 2, 3, 4, 5])

# Use with ComparativeAnalyzer
from comparative_analysis import ComparativeAnalyzer
analyzer = ComparativeAnalyzer(data_dict)
comparison = analyzer.compare_sessions()
```

## Notes
- Large datasets may take a few seconds to load
- Some participants have more data points than others
- The CARS (Childhood Autism Rating Scale) score indicates autism severity:
  - 27-29: Mild autism
  - 30-36: Moderate autism
  - 37+: Severe autism

## Citation
If you use this autism dataset in research, please cite the original source appropriately.

## Troubleshooting

### "No data file found for participant X"
- Ensure the CSV file exists in `data/autism/Eye-tracking Output/`
- Check that the file is named correctly (e.g., `2.csv` for participant 2)

### DtypeWarning during loading
- This is normal and safe to ignore
- It occurs because the raw CSV has mixed data types
- The loader handles this correctly

### Slow loading
- Large datasets (100,000+ points) may take 5-10 seconds to load
- This is normal for real data processing
- The data is cached once loaded
