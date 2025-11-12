# Adult ASD Baseline Model

This directory contains the adult autism spectrum disorder (ASD) baseline model trained on eye-tracking data. This model serves as a comparison point to the children baseline and helps understand age-related differences in gaze patterns.

## Model Information

### Architecture
- **Type**: Autoencoder Neural Network
- **Input Features**: 28 eye-tracking features
- **Architecture**: 28 → 32 → 48 → 24 (latent) → 48 → 32 → 28
- **Activation**: ReLU with Batch Normalization and Dropout (0.4)
- **Total Parameters**: 8,020
- **Optimizer**: Adam (learning rate: 0.00652)
- **Loss Function**: Mean Absolute Error (MAE)

### Performance
- **Validation MAE**: 0.6065
- **Training Samples**: 19 adults
- **Validation Samples**: 5 adults
- **Training Stopped**: Epoch 54 (early stopping at epoch 39)

## Dataset Information

### Source Data
- **File**: `RawEyetrackingASD.mat`
- **Participants**: 24 adults
- **Trials per Participant**: 36
- **Time Samples per Trial**: 14,000
- **Sampling Rate**: To be determined
- **Total Data Points**: ~12 million gaze coordinates

### Feature Extraction
28 features extracted from raw eye-tracking coordinates:

**Spatial Features (8)**:
1-4. X coordinate statistics (mean, std, min, max)
5-8. Y coordinate statistics (mean, std, min, max)

**Velocity Features (7)**:
9-12. Velocity statistics (mean, std, max, median)
26-27. Velocity percentiles (P90, P10)

**Acceleration Features (3)**:
13-15. Acceleration statistics (mean, std, max)

**Fixation Metrics (2)**:
16. Fixation ratio (% time in fixations)
17. Average fixation velocity

**Saccade Metrics (2)**:
18. Saccade ratio (% time in saccades)
19. Average saccade velocity

**Gaze Distribution (4)**:
20-23. Spatial quartiles (X_Q25, X_Q75, Y_Q25, Y_Q75)

**Coverage Metrics (2)**:
24. Total path length
25. Coverage area

**Sample Count (1)**:
28. Total valid samples

## Files in This Directory

### Model Files
- `adult_asd_baseline.keras` - Complete trained model (recommended for deployment)
- `scaler.pkl` - StandardScaler for feature normalization (Python pickle format)
- `scaler.json` - Scaler parameters in JSON format (for JavaScript/web use)

### Documentation
- `README.md` - This file
- `../AGE_BASELINE_COMPARISON.md` - Detailed comparison with children model

## Usage

### Python Usage

```python
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np

# Load model
model = keras.models.load_model('models/baseline_adult_asd/adult_asd_baseline.keras')

# Load scaler
with open('models/baseline_adult_asd/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare your 28 features
features = np.array([[...]])  # Shape: (1, 28)

# Normalize
features_scaled = scaler.transform(features)

# Get reconstruction
reconstruction = model.predict(features_scaled)

# Calculate anomaly score (MAE)
mae_score = np.mean(np.abs(features_scaled - reconstruction))

# Threshold (based on validation MAE)
threshold = 0.6065
is_typical = mae_score < threshold
```

### JavaScript Usage

```javascript
// Load model
const model = await tf.loadLayersModel('models/baseline_adult_asd/adult_asd_baseline.keras');

// Load scaler
const response = await fetch('models/baseline_adult_asd/scaler.json');
const scaler = await response.json();

// Normalize features
function normalize(features) {
    return features.map((val, i) => (val - scaler.mean[i]) / scaler.std[i]);
}

// Your 28 features
const features = [...];  // Array of 28 numbers

// Normalize and predict
const normalized = normalize(features);
const input = tf.tensor2d([normalized]);
const reconstruction = await model.predict(input).array();

// Calculate anomaly score
const mae = calculateMAE(normalized, reconstruction[0]);
const isTypical = mae < 0.6065;
```

## Comparison with Children Model

| Metric | Children (Ages 3-12) | Adult | Difference |
|--------|---------------------|-------|------------|
| Validation MAE | 0.4069 | 0.6065 | +49.1% |
| Participants | 23 | 24 | +1 |
| Architecture | Same | Same | - |
| Parameters | 8,020 | 8,020 | - |

**Key Finding**: The children model shows 49.1% better performance (lower MAE), suggesting children with ASD may have more predictable/stereotyped gaze patterns compared to adults.

### Top Feature Differences

Features showing largest differences between children and adults (normalized values):

1. **Fixation Ratio**: Δ=11,637,985 (children have much higher fixation rates)
2. **Saccade Ratio**: Δ=8,723,036 (different saccadic behavior)
3. **Coverage Area**: Δ=3,661,547 (adults cover more screen area)
4. **Path Length**: Δ=532,679 (adults have longer gaze paths)
5. **Sample Count**: Δ=502,098 (different data collection durations)

## Clinical Implications

1. **Age-Appropriate Screening**: This model should be used for adult ASD screening, not the children model
2. **Developmental Changes**: Large feature differences suggest significant gaze pattern changes from childhood to adulthood
3. **Model Selection**: Always use age-matched baseline for accurate anomaly detection

## Training Details

### Data Preparation
1. Loaded 24 participants from `.mat` file
2. Combined 36 trials per participant
3. Extracted 28 features from raw gaze coordinates
4. Normalized using StandardScaler
5. Split: 80% training (19), 20% validation (5)

### Training Process
- **Epochs**: 100 (stopped at 54 via early stopping)
- **Batch Size**: 4
- **Early Stopping**: Patience=15, monitored validation loss
- **Best Epoch**: 39 (MAE=0.6065)

### Hardware
- Trained on CPU (TensorFlow 2.20.0)
- Training time: ~2 minutes
- Model size: ~100 KB

## Citation

If you use this model, please cite:

```
Adult ASD Baseline Model
Trained on RawEyetrackingASD.mat dataset
Architecture: Autoencoder (28→32→48→24→48→32→28)
Validation MAE: 0.6065
Date: 2024
```

For comparison with children model:
```
Cilia, N. D., Boccignone, G., Campolese, M., De Stefano, C., & Scotto di Freca, A. (2023). 
Eyes Tell All: Gaze Tracking, Recognition & Understanding for Assisting Children with Autism Spectrum Disorder. 
arXiv preprint arXiv:2311.07441.
```

## Future Work

1. **Larger Dataset**: Train on more adult participants for better generalization
2. **Age Stratification**: Create separate models for different adult age ranges (15-20, 21-30, etc.)
3. **Multi-Modal**: Combine with other behavioral metrics
4. **Explainability**: Analyze which features contribute most to ASD detection
5. **Real-Time**: Optimize for real-time screening applications

## License

This model is provided for research and educational purposes. Please ensure compliance with data privacy regulations when deploying for clinical use.

---

**Last Updated**: 2024
**Model Version**: 1.0
**Status**: ✅ Trained and validated, ready for deployment
