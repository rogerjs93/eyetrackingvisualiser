# 🧠 Autism Baseline Model

This directory contains a TensorFlow-based baseline model trained on 23 participants with autism spectrum disorder (ASD). The model learns typical autism gaze patterns and can be used to compare new eye-tracking data against this baseline.

## 📊 Model Information

- **Model Type**: Autoencoder (unsupervised learning)
- **Architecture**: 28 → 64 → 32 → 16 (latent) → 32 → 64 → 28
- **Training Data**: 23 participants with ASD
- **Age Range**: 2.7 - 11.7 years
- **CARS Score Range**: 20.0 - 42.5
- **Total Parameters**: 9,708
- **Training Accuracy**: Validation MAE = 0.7196

## 📁 Files

### `autism_baseline_model.keras` (13.6 MB)
TensorFlow/Keras model file. This is the trained neural network that learns to reconstruct typical autism gaze patterns.

### `scaler.pkl` (2.3 KB)
StandardScaler for feature normalization. Ensures that new data is scaled the same way as the training data.

### `baseline_statistics.json` (5.5 KB)
Statistical baseline including:
- Mean, std, min, max, median for each of the 28 features
- Age and CARS score statistics
- Feature names and descriptions

### `model_metadata.json` (0.6 KB)
Model metadata including:
- Training date
- Number of participants
- Feature descriptions
- Usage instructions

## 🔬 How to Use

### Basic Comparison

```python
from baseline_comparator import BaselineComparator

# Initialize comparator
comparator = BaselineComparator(model_dir='models/baseline')

# Load your eye-tracking data
# data should be a pandas DataFrame with columns: x, y, timestamp, duration
import pandas as pd
data = pd.read_csv('your_data.csv')

# Compare to baseline
results = comparator.compare_to_baseline(data)

print(f"Similarity Score: {results['similarity_score']:.1f}/100")
print(f"Deviation Level: {results['deviation_level']}")
print(f"Interpretation: {results['deviation_interpretation']}")
```

### Generate Detailed Report

```python
# Generate markdown report
report = comparator.generate_comparison_report(
    data, 
    output_path='comparison_report.md'
)
```

## 📈 Features Extracted

The model analyzes 28 features across multiple dimensions:

### Spatial Features (10)
- X/Y position statistics (mean, std, min, max, range)

### Temporal Features (7)
- Duration statistics (mean, std, median, quartiles)
- Time span and sampling rate

### Movement Features (7)
- Path length, efficiency, velocities
- Distance statistics

### Distribution Features (4)
- Spatial entropy
- Concentration metrics (distance from center)

## 🎯 Interpretation Guide

### Similarity Score (0-100)
- **90-100**: Very similar to typical autism patterns
- **70-89**: Moderately similar
- **50-69**: Some similarities with notable differences
- **Below 50**: Significantly different from baseline

### Deviation Level
- **Low** (Z-score < 1.0): Pattern falls within typical autism range
- **Moderate** (Z-score 1.0-2.0): Some unusual characteristics
- **High** (Z-score > 2.0): Significantly different from typical autism patterns

### Z-Scores
Individual feature Z-scores show how many standard deviations away from the baseline mean:
- **|Z| < 1**: Within normal variation (68%)
- **|Z| < 2**: Moderate deviation (95%)
- **|Z| > 2**: Significant deviation (outlier)

## 🔧 Updating the Model

To retrain the model with new participants:

```bash
python baseline_model_builder.py
```

This will:
1. Load all participants from `data/autism/`
2. Extract features from each participant
3. Train a new autoencoder model
4. Save updated model files

## 📚 Research Applications

### Clinical Assessment
- Compare individual patients against population baseline
- Track changes over time
- Identify atypical patterns requiring attention

### Intervention Evaluation
- Measure pre/post intervention changes
- Quantify treatment effectiveness
- Monitor progress

### Comparative Studies
- Compare ASD vs. neurotypical populations
- Age-based comparisons
- Severity correlation analysis

## ⚠️ Important Notes

1. **Dataset Limitation**: Model trained on 23 participants with specific age and CARS score ranges
2. **Generalization**: May not generalize well outside training data distribution
3. **Complementary Tool**: Should be used alongside clinical judgment, not as sole diagnostic tool
4. **Data Quality**: Requires properly formatted eye-tracking data with x, y, timestamp, duration columns

## 📖 Citation

If you use this baseline model in research, please cite:

**Dataset Source**:
- Eye Tracking Autism Dataset by IMT Kaggle Team
- Available at: https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism

**Model**:
- Autism Baseline Gaze Pattern Model
- Eye-Tracking Data Visualizer Project
- GitHub: https://github.com/rogerjs93/eyetrackingvisualiser

## 📧 Support

For questions or issues with the baseline model:
1. Check that your data format matches the expected structure
2. Ensure all 28 features can be extracted from your data
3. Verify that timestamps are properly normalized (starting from 0)

## 🔄 Version History

- **v1.0** (November 2025): Initial release with 23 ASD participants
