# Age-Specific ASD Baseline Deployment Summary

## 🎯 Project Completion

Successfully deployed **two age-specific ASD baseline models** with proper attribution to original research papers.

---

## ✅ What Was Accomplished

### 1. Children ASD Baseline Model (Ages 3-12)
- **Dataset**: Eye-tracking data from 23 children with ASD
- **Source**: Cilia et al. (2023) research dataset
- **Performance**: MAE 0.4069 (optimized, 30.7% better than original)
- **Architecture**: 28→32→48→24→48→32→28 (8,020 parameters)
- **Status**: ✅ Deployed to GitHub Pages

**Research Citation**:
```
Cilia, N. D., Boccignone, G., Campolese, M., De Stefano, C., & Scotto di Freca, A. (2023).
Eyes Tell All: Gaze Tracking, Recognition & Understanding for Assisting Children with Autism Spectrum Disorder.
arXiv preprint arXiv:2311.07441.
```

**Links**:
- Dataset: https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism
- Paper: https://www.researchgate.net/publication/369708398

---

### 2. Adult ASD Baseline Model
- **Dataset**: Eye-tracking data from 24 adults with ASD
- **Source**: RawEyetrackingASD.mat (36 trials per participant, 14,000 samples per trial)
- **Performance**: MAE 0.6065
- **Architecture**: Same as children model (28→32→48→24→48→32→28)
- **Status**: ✅ Trained and deployed to GitHub Pages

**Key Finding**: Adult model has 49.1% higher MAE than children model, suggesting:
- Adults with ASD show more variable gaze patterns
- Children with ASD may have more stereotyped/predictable gaze behavior
- Age-appropriate baselines are critical for accurate screening

---

### 3. Web Interface Updates

#### Age Selection Feature
- ✅ **Dropdown menu** to select between Children (3-12) and Adult baselines
- ✅ **Dynamic model loading** - loads appropriate model based on age selection
- ✅ **Age-specific thresholds** - uses correct MAE threshold for each model
- ✅ **Info sections** - displays dataset details for each age group

#### Proper Attribution
- ✅ **Full research citations** in footer with author names
- ✅ **Paper titles** and publication details
- ✅ **Dataset links** to original sources (Kaggle, ResearchGate)
- ✅ **Clear age ranges** for each baseline model

---

## 📊 Model Comparison

| Metric | Children (3-12) | Adult |
|--------|----------------|-------|
| **Participants** | 23 | 24 |
| **Validation MAE** | 0.4069 | 0.6065 |
| **Architecture** | 8,020 params | 8,020 params |
| **Performance** | Baseline | +49.1% MAE |
| **Interpretation** | More predictable | More variable |

### Top Feature Differences

1. **Fixation Ratio**: Massive difference (children fixate much more)
2. **Saccade Ratio**: Different saccadic behavior between age groups
3. **Coverage Area**: Adults scan more of the screen (Δ=3.66M)
4. **Path Length**: Adults have longer gaze paths (Δ=532K)
5. **Sample Count**: Different data collection characteristics (Δ=502K)

---

## 🚀 Deployment Details

### GitHub Pages URL
- **Live Site**: https://rogerjs93.github.io/eyetrackingvisualiser/

### Files Deployed

#### Children Model
```
models/baseline_children_asd/
├── optimized_autoencoder.keras    # Main model file
├── scaler.json                    # Normalization parameters
├── scaler.pkl                     # Python scaler object
├── comparison_results.json        # Performance metrics
└── README.md                      # Full documentation
```

#### Adult Model
```
models/baseline_adult_asd/
├── adult_asd_baseline.keras       # Main model file
├── scaler.json                    # Normalization parameters
├── scaler.pkl                     # Python scaler object
└── README.md                      # Full documentation
```

#### Web Interface
```
├── index.html                     # Main interface with age selection
├── baseline_model_web.js          # TensorFlow.js model loader
└── models/
    ├── AGE_BASELINE_COMPARISON.md # Detailed comparison report
    └── [both baseline directories]
```

---

## 🔬 Technical Implementation

### JavaScript Model Loader
```javascript
// Dynamic model loading based on age selection
if (currentAgeGroup === 'adult') {
    baselineModel.modelPath = 'models/baseline_adult_asd/adult_asd_baseline.keras';
    baselineModel.scalerPath = 'models/baseline_adult_asd/scaler.json';
    baselineModel.threshold = 0.6065; // Adult threshold
} else {
    baselineModel.modelPath = 'models/baseline_children_asd/optimized_autoencoder.keras';
    baselineModel.scalerPath = 'models/baseline_children_asd/scaler.json';
    baselineModel.threshold = 0.4069; // Children threshold
}
```

### HTML Age Selection
```html
<select id="age-group-select">
    <option value="children">Children (Ages 3-12) - Cilia et al. (2023)</option>
    <option value="adult">Adult - Trained on 24 participants</option>
</select>
```

---

## 📝 Research Attribution

### Footer Citations (as displayed on website)

**Children ASD Baseline (Ages 3-12):**
> Cilia, N. D., Boccignone, G., Campolese, M., De Stefano, C., & Scotto di Freca, A. (2023).
> *Eyes Tell All: Gaze Tracking, Recognition & Understanding for Assisting Children with Autism Spectrum Disorder.*
> arXiv preprint arXiv:2311.07441.
> 
> Dataset: [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism) | 
> Paper: [ResearchGate](https://www.researchgate.net/publication/369708398)

**Adult ASD Baseline:**
> Trained on 24 adult participants with ASD (36 trials each).
> Dataset: RawEyetrackingASD.mat
> Validation MAE: 0.6065 | Architecture: 28→32→48→24→48→32→28

---

## 🎓 Clinical Implications

1. **Age-Appropriate Screening**
   - Use children baseline for ages 3-12
   - Use adult baseline for older participants
   - Mismatched age groups will produce incorrect results

2. **Developmental Differences**
   - Children show more stereotyped gaze patterns (lower MAE)
   - Adults show more variability (higher MAE)
   - Feature differences suggest developmental changes in ASD

3. **Model Selection Importance**
   - 49.1% performance difference between age groups
   - Age-matched baseline is critical for accuracy
   - Website warns users to select appropriate age group

---

## 📈 Performance Metrics

### Children Model (Optimized)
- **Validation MAE**: 0.4069
- **Improvement**: 30.7% better than original (0.7196)
- **Training**: 15 keras-tuner trials
- **Best hyperparameters**:
  - Encoder layers: 32, 48
  - Latent dimension: 24
  - Dropout: 0.4
  - Learning rate: 0.00652

### Adult Model
- **Validation MAE**: 0.6065
- **Training**: 54 epochs (early stopping at 39)
- **Batch size**: 4
- **Same architecture**: For fair comparison with children model

---

## 🔄 Git Commits

### Commit 1: Adult Baseline Model
```
734573a - Add adult ASD baseline model for age-group comparison
- Trained on RawEyetrackingASD.mat (24 adults, 36 trials each)
- Same architecture as children model: 28->32->48->24->48->32->28
- Validation MAE: 0.6065 vs children 0.4069 (49.1% difference)
- Created comprehensive comparison report showing feature differences
```

### Commit 2: Age-Specific Selection
```
d3a5955 - Add age-specific baseline selection with proper attribution
- Dropdown to select between Children (3-12) and Adult baselines
- Dynamic model loading based on age selection
- Children model: Cilia et al. (2023) - MAE 0.4069
- Adult model: 24 participants - MAE 0.6065
- Proper research citations in footer with full paper details
```

---

## ✨ User Experience

### Before
- Single children baseline only
- No age selection
- Limited attribution

### After
- ✅ **Two age-specific baselines** (children & adult)
- ✅ **Easy age selection** via dropdown
- ✅ **Full research citations** with authors, titles, and links
- ✅ **Clear warnings** about age-appropriate usage
- ✅ **Detailed info sections** for each age group
- ✅ **Dynamic model loading** with proper thresholds

---

## 🎉 Project Status

### ✅ Completed
- [x] Trained optimized children baseline (30.7% improvement)
- [x] Trained adult baseline from .mat file
- [x] Created comprehensive comparison analysis
- [x] Deployed both models to GitHub Pages
- [x] Added age selection to web interface
- [x] Included full research citations
- [x] Updated all documentation

### 📚 Documentation Created
1. `models/baseline_children_asd/README.md` - Children model documentation
2. `models/baseline_adult_asd/README.md` - Adult model documentation
3. `models/AGE_BASELINE_COMPARISON.md` - Detailed age comparison
4. `DATASET_DOCUMENTATION.md` - Dataset comparison guide
5. `AGE_SPECIFIC_DEPLOYMENT_SUMMARY.md` - This file

---

## 🔮 Future Enhancements

1. **Additional Age Groups**
   - Teen baseline (13-17 years)
   - Elderly baseline (60+ years)
   - Longitudinal tracking across lifespan

2. **Enhanced Features**
   - Real-time age detection from gaze patterns
   - Confidence scores for age group matching
   - Multi-model ensemble predictions

3. **Research Extensions**
   - Gender-specific baselines
   - Severity level classification
   - Intervention effectiveness tracking

---

## 📞 Repository Information

- **GitHub**: https://github.com/rogerjs93/eyetrackingvisualiser
- **GitHub Pages**: https://rogerjs93.github.io/eyetrackingvisualiser/
- **License**: Open Source
- **Status**: ✅ Production Ready

---

## 🙏 Acknowledgments

Special thanks to:

1. **Cilia et al. (2023)** for providing the children ASD dataset and pioneering research in eye-tracking for autism assessment
2. **Original dataset contributors** for making research data publicly available
3. **TensorFlow.js team** for enabling browser-based ML
4. **Open source community** for tools and libraries

---

**Last Updated**: November 12, 2024  
**Version**: 2.0 (Age-Specific Baselines)  
**Status**: ✅ Live on GitHub Pages
