# Dual-Strategy Training Plan: Small ASD vs Large Neurotypical

## Strategy Overview

### **Strategy A: Lightweight Models for Small ASD Datasets** 
- **Use Case**: Children ASD (25 samples), Adult ASD (24 samples from RawEyetrackingASD.mat)
- **Approach**: Simple, robust models that avoid overfitting
- **Architecture**: Feature selection + lightweight networks

### **Strategy B: Complex Models for Large Neurotypical Dataset**
- **Use Case**: Neurotypical baseline (500k+ fixations from 20+ studies)
- **Approach**: Deep learning with full feature set
- **Architecture**: Deep autoencoders, ensemble methods, advanced features

---

## Current Data Inventory

### ✅ Small Datasets (Need Lightweight Approach)

| Dataset | Samples | Age Range | Source | Status |
|---------|---------|-----------|---------|--------|
| **Children ASD** | 25 | 2.7-12.3 years | Kaggle | ✅ Processed |
| **Adult ASD** | 24 | Unknown | RawEyetrackingASD.mat | ⚠️ Need to process |

### ✅ Large Dataset (Can Use Complex Approach)

| Dataset | Fixations | Participants | Studies | Source | Status |
|---------|-----------|--------------|---------|--------|--------|
| **Neurotypical** | 2,400,000+ | 700+ | 25 studies | Dryad database | ⚠️ Need to process |

**Neurotypical Studies Available:**
- Age Study (58 participants, ages 7.6-80.6)
- Baseline (48 participants, age 19-28)
- Bias (43 participants, age 19-28)
- Memory I & II (79 participants total)
- Tactile (57 participants)
- Face experiments (133 participants)
- And 15+ more studies...

---

## Implementation Plan

### Phase 1: Deploy Lightweight ASD Models (Current Priority) ✅

**Status**: Optimized children ASD model trained (MAE 0.4231, only 4% worse than baseline)

**Next Steps**:
1. ✅ Export optimized children model to TFJS
2. Update JavaScript to use 20 selected features
3. Test in browser
4. Deploy to GitHub Pages

**Selected Features (20)**:
```
x_std, fixation_count, saccade_velocity_mean, saccade_velocity_std,
saccade_amplitude_mean, saccade_amplitude_std, fixation_dispersion,
gaze_entropy, center_bias, edge_bias, roi_focus, vertical_horizontal_ratio,
temporal_consistency, saccade_direction_entropy, spatial_autocorr_x,
spatial_autocorr_y, fixation_cluster_density, spatial_revisitation_rate,
saccade_amplitude_entropy, fixation_duration_entropy
```

**Model Specs**:
- Architecture: 20→16→8→16→20
- Parameters: 1,084
- Regularization: L2 + 30% dropout
- Training: 5-fold CV

---

### Phase 2: Process and Train Neurotypical Baseline (Next)

**Approach**: Full complexity since we have abundant data

#### Step 2.1: Data Extraction
```python
# Process HDF5 files from neurotypical database
# Target: 1000+ samples from diverse studies
# Extract all 43 features without selection
```

**Recommended Studies** (for age-matched baseline):
- Age Study (children 7.6 years → matches ASD dataset)
- Baseline (adults 19-28 → general neurotypical)
- Memory I/II (diverse ages 18-49)
- Tactile (ages 18-29)

#### Step 2.2: Deep Model Architecture
```python
# Full 43-feature deep autoencoder
Input: 43 features
  ↓
Dense(128) + BN + Dropout(0.2)
  ↓
Dense(64) + BN + Dropout(0.2)
  ↓
Dense(32) + BN
  ↓
Latent(16)
  ↓
Dense(32) + BN + Dropout(0.2)
  ↓
Dense(64) + BN + Dropout(0.2)
  ↓
Dense(128) + BN + Dropout(0.2)
  ↓
Output: 43 features

Parameters: ~25,000
Samples needed: 500+
```

#### Step 2.3: Advanced Training
- **No feature selection**: Use all 43 features
- **Larger batches**: 32-64 samples
- **More epochs**: 200-300 with early stopping
- **Ensemble**: Train 5 models, average predictions
- **Data split**: 80/10/10 train/val/test

**Expected Performance**: MAE < 0.35 (better than current baseline 0.4069)

---

### Phase 3: Adult ASD Model (After Neurotypical)

**Why After?**: Can use neurotypical as transfer learning base

#### Approach:
1. Load RawEyetrackingASD.mat (24 adult ASD samples)
2. Extract 43 features
3. Apply lightweight strategy (like children ASD)
4. OR use transfer learning from neurotypical model

---

## Training Scripts Structure

### Current Scripts:
- ✅ `prepare_training_data.py` - Data preparation
- ✅ `train_enhanced_model.py` - Original (too complex)
- ✅ `train_optimized_model.py` - Lightweight for small datasets

### Needed Scripts:
- ⚠️ `extract_neurotypical_data.py` - Process HDF5 files
- ⚠️ `train_neurotypical_model.py` - Deep model for large dataset
- ⚠️ `train_adult_asd_model.py` - Adult ASD with lightweight approach

---

## Deployment Architecture

### Browser Models (TensorFlow.js):

```
models/
├── ACTIVE/
│   ├── children_asd_optimized_tfjs/     # Lightweight (20 features)
│   │   ├── model.json
│   │   ├── preprocessing.json            # Feature selection info
│   │   └── weights.bin
│   │
│   ├── adult_asd_optimized_tfjs/        # Lightweight (20 features)
│   │   └── ...
│   │
│   └── neurotypical_deep_tfjs/          # Deep model (43 features)
│       ├── model.json
│       └── weights.bin
│
└── LEGACY/
    └── baseline_*_tfjs/                  # Old 28-feature models
```

### JavaScript Feature Handling:

```javascript
// baseline_model_web.js

function extractFeatures(data) {
    // Always extract all 43 features
    const allFeatures = [
        // ... extract all 43 ...
    ];
    
    // Apply feature selection based on model type
    if (modelType === 'children_asd' || modelType === 'adult_asd') {
        // Use only 20 selected features
        const selectedIndices = [1, 10, 11, 12, ...]; // From preprocessing.json
        return selectedIndices.map(i => allFeatures[i]);
    } else {
        // Neurotypical uses all 43
        return allFeatures;
    }
}
```

---

## Performance Expectations

| Model | Dataset | Samples | Features | Architecture | Expected MAE |
|-------|---------|---------|----------|--------------|--------------|
| **Children ASD** | 25 | 20 | Lightweight | 0.42 | ✅ Trained |
| **Adult ASD** | 24 | 20 | Lightweight | ~0.45 | Pending |
| **Neurotypical** | 1000+ | 43 | Deep | < 0.35 | Pending |

---

## Next Immediate Actions

1. **Export Current Model** (15 min)
   - Convert optimized children ASD model to TFJS
   - Include preprocessing.json with feature indices

2. **Update JavaScript** (30 min)
   - Modify extractFeatures() to handle feature selection
   - Load preprocessing.json
   - Select correct 20 features for ASD models

3. **Test & Deploy** (30 min)
   - Test with sample CSV in browser
   - Verify predictions
   - Push to GitHub Pages

4. **Start Neurotypical Processing** (2-3 hours)
   - Create HDF5 extraction script
   - Process top 5 studies
   - Prepare training data

---

## Long-Term Vision

### Comparison Dashboard:
```
Upload CSV → Extract Features (43)
            ↓
      ┌─────┴─────┬─────────┐
      ↓           ↓         ↓
  Children ASD  Adult ASD  Neurotypical
  (20 features) (20 feat.) (43 features)
      ↓           ↓         ↓
   Similarity  Similarity Similarity
      ↓           ↓         ↓
   Age-specific analysis + Clinical insights
```

### Research Output:
- Baseline comparison across age groups
- Feature importance analysis
- Clinical pattern identification
- Publication-ready results

---

## Summary

**Current Status**: ✅ Phase 1 nearly complete (lightweight children ASD model optimized)

**Immediate Priority**: Export → Update JS → Deploy

**Next Major Work**: Process neurotypical database for deep model training

**Key Insight**: Different strategies for different data sizes prevents overfitting while maximizing performance where possible.
