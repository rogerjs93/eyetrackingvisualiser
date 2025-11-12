# 📚 Dataset Documentation & Model Organization

## Overview

This project now properly distinguishes between **children** and **adult** autism eye-tracking datasets, with age-appropriate baseline models for clinical screening and research.

---

## 🧒 Children ASD Baseline Model (Ages 3-12) - **CURRENT MODEL**

### Research Foundation
**Paper:** Eye-tracking Dataset to Support the Research on Autism Spectrum Disorder  
**Authors:** Cilia, F., et al. (2023)  
**Link:** https://www.researchgate.net/publication/369708398_Eye-tracking_Dataset_to_Support_the_Research_on_Autism_Spectrum_Disorder  
**Dataset:** https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism  

### Dataset Characteristics
- **Participants:** 25 children with ASD (23-24 usable)
- **Age Range:** 3-12 years
- **Sampling Rate:** 60 Hz
- **Duration:** ~5 minutes per participant
- **Stimuli:** Simple, standardized (photos, short videos, cartoons)
- **Data Format:** CSV files + MATLAB (.mat) file

### Clinical Relevance ✅
| Factor | Details |
|--------|---------|
| **Target Age** | 3-12 years (critical early diagnosis window) |
| **Applicability** | Clinical screening and early intervention |
| **Stimuli** | Simple, reproducible in clinical settings |
| **Sample Balance** | Includes ASD and typically developing (TD) groups |
| **Purpose** | Designed explicitly for ASD research and diagnostics |

### Model Performance
- **Original Model:** MAE 0.7196 (9,708 parameters)
- **Optimized Model:** MAE 0.4069 (8,020 parameters)
- **Improvement:** 43.5% better during training, 30.7% on real autism data
- **Architecture:** 28→32→48→24→48→32→28 (hyperparameter-tuned)

### Location
`models/baseline_children_asd/`
- `optimized_autoencoder.keras` (RECOMMENDED)
- `autism_baseline_model.keras` (Original)
- `baseline_statistics.json`
- `scaler.pkl`
- `README.md` (full documentation)

---

## 👨 Adult ASD Dataset (Ages 15-30) - **NOT YET IMPLEMENTED**

### Research Foundation
**Paper:** Distinct neural mechanisms of social orienting and mentalizing revealed by independent measures of neural and eye movement typicality  
**Authors:** Ramot, M., et al. (2019)  
**Published:** Nature Human Behaviour  
**Link:** https://nih.figshare.com/articles/dataset/Eye_tracking_data_for_participants_with_Autism_Spectrum_Disorders/10324877  

### Dataset Characteristics
- **Participants:** 36 males with ASD + matched controls
- **Age Range:** 15-30 years (adolescents/young adults)
- **Sampling Rate:** 1000 Hz (very high precision!)
- **Duration:** Longer sessions with complex stimuli
- **Stimuli:** Movie clips of social interactions
- **Data Format:** MATLAB format with fMRI integration
- **Additional:** Links gaze patterns to neural (fMRI) activity

### Research Focus 🔬
| Factor | Details |
|--------|---------|
| **Target Age** | 15-30 years (older adolescents/adults) |
| **Applicability** | Research on neural mechanisms, not early screening |
| **Stimuli** | Complex movies (harder to standardize clinically) |
| **Sample Size** | Small (36 males), limited generalizability |
| **Purpose** | Understanding neural basis of gaze in adults with ASD |

### Pros & Cons

**Advantages:**
- ✅ Extremely high temporal resolution (1000 Hz)
- ✅ Links gaze behavior to brain activity (fMRI)
- ✅ Well-controlled experimental design
- ✅ "Eye movement typicality" metrics as benchmarks

**Limitations:**
- ❌ Wrong age group for early clinical diagnosis
- ❌ Small, male-only sample
- ❌ Complex stimuli difficult to reproduce clinically
- ❌ Optimized for research, not clinical screening

### Future Use
This dataset could be used to create:
- **Adult ASD Baseline Model** for ages 15-30
- Comparison studies of gaze patterns across development
- High-precision scan-path analysis
- Research on social attention mechanisms

### Location
**Not yet implemented** - Would be in `models/baseline_adults_asd/`

---

## 📊 Comparison Summary

| Aspect | Children Dataset (Current) | Adult Dataset (Future) |
|--------|---------------------------|----------------------|
| **Age** | 3-12 years | 15-30 years |
| **Clinical Use** | ✅ Early screening | ❌ Research only |
| **Sample Size** | 25 (23-24 usable) | 36 (males only) |
| **Sampling Rate** | 60 Hz | 1000 Hz |
| **Stimuli** | Simple, standardized | Complex movies |
| **Format** | CSV + MAT | MAT with fMRI |
| **Status** | ✅ **IMPLEMENTED** | ⏳ Future work |

---

## 🎯 Why Separate Models?

### 1. **Developmental Differences**
- Children's eye movements differ fundamentally from adults
- Neural development affects gaze patterns
- Different attention strategies at different ages

### 2. **Clinical Window**
- Ages 3-12: Critical for early diagnosis and intervention
- Early detection has the most impact on outcomes
- Children's baseline needed for pediatric screening

### 3. **Stimulus Appropriateness**
- Children respond better to simple, engaging stimuli
- Adults can handle complex social scenes
- Different cognitive loads affect gaze behavior

### 4. **Standardization Needs**
- Clinical tools need reproducible, simple protocols
- Research can use complex, controlled stimuli
- Different goals = different optimal designs

---

## 🚀 Future Work

### Phase 1: Complete Children Model ✅
- [x] Train baseline model on children dataset
- [x] Optimize hyperparameters (30.7% improvement achieved)
- [x] Document age range and clinical applicability
- [x] Update web interface with proper attribution

### Phase 2: Add Adult Model ⏳
- [ ] Download and process Ramot et al. dataset
- [ ] Extract same 28 features from adult data
- [ ] Train separate adult baseline model
- [ ] Compare children vs adult patterns
- [ ] Document developmental differences

### Phase 3: Multi-Age Analysis 🔮
- [ ] Create age-specific models (3-5, 6-8, 9-12, 13-17, 18-30)
- [ ] Longitudinal tracking of pattern changes
- [ ] Intervention response modeling
- [ ] Cross-dataset validation

---

## 📁 Current Project Structure

```
models/
├── baseline_children_asd/          # ✅ CURRENT - Ages 3-12
│   ├── optimized_autoencoder.keras # Best model (30.7% better)
│   ├── autism_baseline_model.keras # Original model
│   ├── baseline_statistics.json    # Feature statistics
│   ├── scaler.pkl                  # Normalization parameters
│   ├── comparison_results.json     # Performance comparison
│   └── README.md                   # Full documentation
│
├── baseline_adults_asd/            # ⏳ FUTURE - Ages 15-30
│   └── (not yet implemented)
│
└── baseline_tfjs/                  # Web deployment
    ├── model.json
    ├── group1-shard1of1.bin
    ├── scaler.json
    └── baseline_statistics.json
```

---

## 📖 Citations

### Children Dataset (Currently Used)
```
Cilia, F., et al. (2023). Eye-tracking Dataset to Support the Research on 
Autism Spectrum Disorder. ResearchGate.
DOI: 10.13140/RG.2.2.27528.12808
```

### Adult Dataset (For Future Use)
```
Ramot, M., Walsh, C., Martin, A. (2019). Distinct neural mechanisms of 
social orienting and mentalizing revealed by independent measures of neural 
and eye movement typicality. Nature Human Behaviour.
Figshare Dataset: 10324877
```

---

## ⚠️ Important Usage Notes

### For Researchers
- Children model: Use for ages 3-12 only
- Adult model (when available): Use for ages 15-30 only
- Do not mix age groups - developmental patterns differ significantly
- Always cite the appropriate dataset

### For Clinicians
- This is a **screening support tool**, not a diagnostic instrument
- Results should be interpreted by qualified professionals
- Always combine with comprehensive clinical assessment
- Consider child's developmental stage and context

### For Developers
- Age-appropriate models are in separate directories
- Web interface currently uses children model only
- Future: Add model selection based on participant age
- Maintain separate scalers and statistics per age group

---

**Last Updated:** November 12, 2025  
**Model Version:** Children ASD Baseline v1.0 (Optimized)  
**Status:** Production-ready for ages 3-12
