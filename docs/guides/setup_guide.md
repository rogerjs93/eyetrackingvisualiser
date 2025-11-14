# 🚀 Quick Setup Guide - Full Python Version

## Why Use the Python Version?

The GitHub Pages web demo only shows **synthetic data generation**. For the **full feature set**, you need to run the Python dashboard locally:

### Python Version Includes:
✅ **Real Autism Dataset** - 25 ASD participants  
✅ **AI Baseline Model** - Compare against trained autism patterns  
✅ **Similarity Scoring** - 0-100 scores with deviation analysis  
✅ **ML Pattern Recognition** - Automated gaze behavior detection  
✅ **Cognitive Load Analysis** - Advanced metrics  
✅ **Methodology Explanations** - Built-in documentation  

## 📦 Installation (5 minutes)

### Prerequisites
- Python 3.7+ installed
- Git installed

### Step 1: Clone Repository
```bash
git clone https://github.com/rogerjs93/eyetrackingvisualiser.git
cd eyetrackingvisualiser
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- dash (web framework)
- plotly (visualizations)
- pandas, numpy, scipy (data processing)
- scikit-learn (ML algorithms)
- tensorflow (baseline model)

### Step 3: Download Autism Dataset (Optional)

**If you want to use the real autism data:**

1. Visit: https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism
2. Download the dataset
3. Extract to `data/autism/` folder:
   ```
   data/autism/
   ├── Metadata_Participants.csv
   └── Eye-tracking Output/
       ├── 1.csv
       ├── 2.csv
       └── ... (25 files)
   ```

**Note**: The baseline model works with or without the autism dataset!

### Step 4: Run Dashboard
```bash
python interactive_dashboard.py
```

### Step 5: Open Browser
Navigate to: **http://localhost:8050**

## 🎯 Using the Features

### 1. Synthetic Data (No Dataset Required)
- Select **"Synthetic Data"** from dropdown
- Choose pattern type (Reading, Natural, etc.)
- Click **"Generate Synthetic Data"**
- Explore all 8 visualization tabs

### 2. Real Autism Data (Dataset Required)
- Select **"Autism Dataset"** from dropdown
- Choose participant (P001-P025)
- Click **"Load Participant Data"**
- View participant info (Age, Gender, CARS score)
- Analyze real autism gaze patterns

### 3. Baseline Model Comparison
```python
# In Python terminal or script
from baseline_comparator import BaselineComparator

# Load comparator
comparator = BaselineComparator()

# Load your data (pandas DataFrame with x, y, timestamp, duration)
import pandas as pd
data = pd.read_csv('your_data.csv')

# Compare
results = comparator.compare_to_baseline(data)
print(f"Similarity: {results['similarity_score']}/100")
print(f"Deviation: {results['deviation_level']}")

# Generate report
comparator.generate_comparison_report(data, 'my_report.md')
```

### 4. Build Custom Baseline
If you have your own dataset:
```bash
# Add your data to data/autism/
# Then rebuild baseline
python baseline_model_builder.py
```

## 📊 Dashboard Features

### Tab 1: Overview Dashboard
- Heatmap, scan path, duration histogram, spatial distribution
- All-in-one view

### Tab 2: Heatmap & Scan Path
- Detailed gaze heatmap with density visualization
- Scan path with start/end markers
- Temporal gradient showing time progression

### Tab 3: Distributions
- X/Y position distributions
- Duration distribution
- Density scatter plot

### Tab 4: Temporal Analysis
- X/Y positions over time
- Movement velocity analysis
- Time-based patterns

### Tab 5: Attention Zones
- Attention density heatmap
- Contour lines showing focus areas
- Fixation overlays

### Tab 6: 🤖 AI Pattern Recognition
- Reading behavior detection
- Expertise level classification
- Confusion indicators
- Areas of Interest (AOI) detection

### Tab 7: 🧠 Cognitive Load
- Spatial entropy
- Fixation rate
- Saccade metrics
- Ambient vs. focal attention
- Task difficulty score (0-10)

### Tab 8: 🎨 Advanced Visualizations
- Sankey diagram (gaze flow)
- Network graph (AOI transitions)
- 4D visualization (x, y, time, duration)
- Velocity heatmap

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Use different port
python interactive_dashboard.py --port 8051
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### TensorFlow Issues (Windows)
```bash
# Install CPU version
pip install tensorflow-cpu
```

### Data Loading Errors
- Check file paths in `data/autism/`
- Ensure CSV format matches expected structure
- Verify file permissions

## 💡 Tips

1. **Start with Synthetic Data** - Learn the interface before using real data
2. **Compare Participants** - Load different participants to see pattern variations
3. **Export Reports** - Use baseline comparator to generate shareable reports
4. **Adjust Resolution** - Change screen resolution for different display sizes
5. **Read Methodology** - Click info icons for metric explanations

## 📚 Learn More

- **Baseline Model**: See `BASELINE_MODEL_SUMMARY.md`
- **Model Details**: Check `models/baseline/README.md`
- **Dataset Info**: Read `data/autism/README.md`
- **Full README**: See `README.md`

## 🆘 Get Help

1. Check documentation files
2. Review example code in scripts
3. Open GitHub issue: https://github.com/rogerjs93/eyetrackingvisualiser/issues

## 🎓 For Researchers

### Citation
```bibtex
@software{eyetracking_visualizer_2025,
  author = {Roger},
  title = {Eye-Tracking Data Visualizer with Autism Baseline Model},
  year = {2025},
  url = {https://github.com/rogerjs93/eyetrackingvisualiser}
}
```

### Dataset Citation
```bibtex
@dataset{autism_eyetracking_2024,
  author = {IMT Kaggle Team},
  title = {Eye Tracking Autism Dataset},
  year = {2024},
  url = {https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism}
}
```

---

**Ready to explore? Run `python interactive_dashboard.py` and open http://localhost:8050** 🚀
