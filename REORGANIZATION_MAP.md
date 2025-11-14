# Repository Reorganization Map

**Date**: November 14, 2025  
**Purpose**: Document file moves for professional repository structure

## New Directory Structure

```
eyetrackingvisualiser/
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── docs/                             # 📚 All documentation
│   ├── methodology/                  # Research & feature engineering
│   ├── deployment/                   # Deployment guides & summaries
│   ├── datasets/                     # Data documentation
│   ├── guides/                       # User guides & tutorials
│   └── history/                      # Historical reports & fixes
│
├── scripts/                          # 🐍 Python scripts
│   ├── training/                     # Model training scripts
│   ├── preprocessing/                # Data preparation
│   ├── conversion/                   # Model format conversion
│   ├── analysis/                     # Data analysis & visualization
│   └── utilities/                    # Helper scripts
│
├── web/                              # 🌐 Browser application
│   ├── html/                         # HTML files
│   ├── js/                           # JavaScript modules
│   └── assets/                       # Images, icons, etc.
│
├── models/                           # 🤖 Trained models
│   ├── production/                   # Production-ready TFJS models
│   │   ├── children_asd_optimized/   # Optimized children model
│   │   ├── adult_asd/                # Adult ASD model
│   │   └── neurotypical/             # Neurotypical baseline
│   ├── training/                     # Keras training models
│   │   ├── children_asd/
│   │   ├── adult_asd/
│   │   └── neurotypical/
│   └── archive/                      # Legacy/deprecated models
│
├── data/                             # 📊 Datasets
│   ├── raw/                          # Original unprocessed data
│   │   ├── autism/                   # ASD datasets
│   │   └── standard/                 # Neurotypical datasets
│   └── processed/                    # Prepared training data
│
└── tests/                            # 🧪 Test files
    ├── unit/                         # Unit tests
    ├── integration/                  # Integration tests
    └── browser/                      # Browser tests
```

---

## File Movement Plan

### 📚 Documentation → `docs/`

#### Methodology Documentation → `docs/methodology/`
- `RESEARCH_METHODOLOGY.md` → `docs/methodology/research_methodology.md`
- `METHODOLOGY.md` → `docs/methodology/feature_engineering.md`
- `PHASE2_TEST_RESULTS.md` → `docs/methodology/phase2_validation_results.md`
- `MODEL_OPTIMIZATION_GUIDE.md` → `docs/methodology/optimization_guide.md`

#### Deployment Documentation → `docs/deployment/`
- `DEPLOYMENT_SUMMARY.md` → `docs/deployment/optimized_model_deployment.md`
- `DUAL_STRATEGY_PLAN.md` → `docs/deployment/dual_strategy_plan.md`
- `AGE_SPECIFIC_DEPLOYMENT_SUMMARY.md` → `docs/deployment/age_specific_deployment.md`
- `WEB_AI_IMPLEMENTATION_SUMMARY.md` → `docs/deployment/web_ai_implementation.md`
- `MODEL_ORGANIZATION_PLAN.md` → `docs/deployment/model_organization.md`

#### Dataset Documentation → `docs/datasets/`
- `AUTISM_DATA_README.md` → `docs/datasets/autism_data_readme.md`
- `ADULT_DATASET_INSTRUCTIONS.md` → `docs/datasets/adult_dataset_instructions.md`
- `DATASET_DOCUMENTATION.md` → `docs/datasets/dataset_overview.md`

#### User Guides → `docs/guides/`
- `QUICKSTART.md` → `docs/guides/quickstart.md`
- `AUTISM_QUICKSTART.md` → `docs/guides/autism_quickstart.md`
- `SETUP_GUIDE.md` → `docs/guides/setup_guide.md`

#### Historical Reports → `docs/history/`
- `BASELINE_MODEL_SUMMARY.md` → `docs/history/baseline_model_summary.md`
- `EXPLANATION_UPDATE.md` → `docs/history/explanation_update.md`
- `FIX_SUMMARY.md` → `docs/history/fix_summary.md`
- `comparison_report_1.md` → `docs/history/comparison_report_1.md`

---

### 🐍 Python Scripts → `scripts/`

#### Training Scripts → `scripts/training/`
- `train_enhanced_model.py` → `scripts/training/train_enhanced_model.py`
- `train_optimized_model.py` → `scripts/training/train_optimized_model.py`
- `train_adult_baseline.py` → `scripts/training/train_adult_baseline.py`
- `train_neurotypical_baseline.py` → `scripts/training/train_neurotypical_baseline.py`
- `baseline_model_builder.py` → `scripts/training/baseline_model_builder.py`
- `advanced_model_trainer.py` → `scripts/training/advanced_model_trainer.py`

#### Preprocessing Scripts → `scripts/preprocessing/`
- `prepare_training_data.py` → `scripts/preprocessing/prepare_training_data.py`
- `autism_data_loader.py` → `scripts/preprocessing/autism_data_loader.py`
- `download_adult_dataset.py` → `scripts/preprocessing/download_adult_dataset.py`
- `sample_data_generator.py` → `scripts/preprocessing/sample_data_generator.py`

#### Conversion Scripts → `scripts/conversion/`
- `export_optimized_model.py` → `scripts/conversion/export_optimized_model.py`
- `export_model.py` → `scripts/conversion/export_model.py`
- `convert_models_to_tfjs.py` → `scripts/conversion/convert_models_to_tfjs.py`
- `convert_model_to_tfjs.py` → `scripts/conversion/convert_model_to_tfjs.py`
- `convert_optimized_model.py` → `scripts/conversion/convert_optimized_model.py`
- `convert_scaler.py` → `scripts/conversion/convert_scaler.py`
- `manual_tfjs_converter.py` → `scripts/conversion/manual_tfjs_converter.py`
- `simple_tfjs_converter.py` → `scripts/conversion/simple_tfjs_converter.py`
- `tfjs_clean_converter.py` → `scripts/conversion/tfjs_clean_converter.py`
- `convert_via_savedmodel.py` → `scripts/conversion/convert_via_savedmodel.py`
- `convert_via_saved_model.py` → `scripts/conversion/convert_via_saved_model.py`

#### Analysis Scripts → `scripts/analysis/`
- `baseline_comparator.py` → `scripts/analysis/baseline_comparator.py`
- `comparative_analysis.py` → `scripts/analysis/comparative_analysis.py`
- `compare_age_baselines.py` → `scripts/analysis/compare_age_baselines.py`
- `compare_mat_vs_csv.py` → `scripts/analysis/compare_mat_vs_csv.py`
- `pattern_recognition.py` → `scripts/analysis/pattern_recognition.py`
- `cognitive_load.py` → `scripts/analysis/cognitive_load.py`

#### Utility Scripts → `scripts/utilities/`
- `analyze_mat_structure.py` → `scripts/utilities/analyze_mat_structure.py`
- `inspect_mat_file.py` → `scripts/utilities/inspect_mat_file.py`
- `check_weights.py` → `scripts/utilities/check_weights.py`

---

### 🌐 Web Files → `web/`

#### HTML Files → `web/html/`
- `index.html` → Root (keep for GitHub Pages)
- `index.html` → `web/html/index.html` (backup copy)
- `index_old.html` → `web/html/index_old.html`
- `test_web.html` → `tests/browser/test_web.html`
- `test_optimized_model.html` → `tests/browser/test_optimized_model.html`

#### JavaScript Files → `web/js/`
- `baseline_model_web.js` → `web/js/baseline_model_web.js`
- `visualizer.js` → `web/js/visualizer.js`
- `methodology_explanations.js` → `web/js/methodology_explanations.js`

#### Assets → `web/assets/`
- `autism_test_viz.png` → `web/assets/autism_test_viz.png`
- `dashboard_natural.png` → `web/assets/dashboard_natural.png`

---

### 🤖 Models → `models/` (Reorganized)

#### Production Models → `models/production/`
- `models/ACTIVE/children_asd_optimized_tfjs/` → `models/production/children_asd_optimized/`
- `models/baseline_adult_asd_tfjs/` → `models/production/adult_asd/`
- `models/baseline_neurotypical_tfjs/` → `models/production/neurotypical/`

#### Training Models → `models/training/`
- `models/children_asd_optimized/` → `models/training/children_asd_optimized/`
- `models/children_asd_enhanced/` → `models/training/children_asd_enhanced/`
- `models/baseline_adult_asd/` → `models/training/adult_asd/`
- `models/baseline_neurotypical/` → `models/training/neurotypical/`

#### Archive → `models/archive/`
- `models/baseline/` → `models/archive/baseline/`
- `models/baseline_advanced/` → `models/archive/baseline_advanced/`
- `models/baseline_saved_model/` → `models/archive/baseline_saved_model/`
- `models/baseline_tfjs/` → `models/archive/baseline_tfjs/`
- `models/optimized_tfjs/` → `models/archive/optimized_tfjs/`
- `models/baseline_children_asd/` → `models/archive/baseline_children_asd/`
- `models/baseline_children_asd_tfjs/` → `models/archive/baseline_children_asd_tfjs/`

#### Documentation
- `models/AGE_BASELINE_COMPARISON.md` → `docs/deployment/age_baseline_comparison.md`

---

### 🧪 Test Files → `tests/`

#### Unit Tests → `tests/unit/`
- `test_data_format.py` → `tests/unit/test_data_format.py`
- `test_timestamp.py` → `tests/unit/test_timestamp.py`

#### Integration Tests → `tests/integration/`
- `test_autism_integration.py` → `tests/integration/test_autism_integration.py`
- `test_model_comparison.py` → `tests/integration/test_model_comparison.py`

#### Browser Tests → `tests/browser/`
- `test_web.html` → `tests/browser/test_web.html`
- `test_optimized_model.html` → `tests/browser/test_optimized_model.html`

---

### 📊 Data → `data/` (Reorganized)

#### Raw Data → `data/raw/`
- `data/autism/` → `data/raw/autism/`
- `data/standard/` → `data/raw/standard/`
- `data/text.csv` → `data/raw/text.csv`

#### Processed Data → `data/processed/`
- `data/prepared/` → `data/processed/prepared/`

---

### ⚙️ Files Staying in Root

- `README.md` - Main project documentation
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `index.html` - GitHub Pages entry point
- `venv/` - Virtual environment (gitignored)
- `__pycache__/` - Python cache (gitignored)

---

### 🗑️ Files to Remove (Deprecated/Duplicates)

Consider removing or consolidating:
- `eyetracking_visualizer.py` - Check if still needed
- `interactive_dashboard.py` - Merge with main visualizer?
- `quick_autism_viz.py` - Move to examples?
- `test_simple_viz.py` - Consolidate tests
- `demo_advanced_features.py` - Move to examples?
- `methodology_explanations.py` - Duplicate of .js version?

---

## Code Path Updates Required

### JavaScript Path Updates

**File**: `web/js/baseline_model_web.js`
```javascript
// OLD: this.modelPath = 'models/baseline_children_asd_tfjs/model.json';
// NEW: this.modelPath = 'models/production/children_asd/model.json';

// OLD: this.modelPath = 'models/ACTIVE/children_asd_optimized_tfjs/model.json';
// NEW: this.modelPath = 'models/production/children_asd_optimized/model.json';

// OLD: this.modelPath = 'models/baseline_adult_asd_tfjs/model.json';
// NEW: this.modelPath = 'models/production/adult_asd/model.json';

// OLD: this.modelPath = 'models/baseline_neurotypical_tfjs/model.json';
// NEW: this.modelPath = 'models/production/neurotypical/model.json';
```

**File**: `index.html`
```html
<!-- OLD: <script src="baseline_model_web.js"></script> -->
<!-- NEW: <script src="web/js/baseline_model_web.js"></script> -->

<!-- OLD: <script src="visualizer.js"></script> -->
<!-- NEW: <script src="web/js/visualizer.js"></script> -->
```

### Python Path Updates

**Training Scripts** (in `scripts/training/`):
```python
# OLD: model_save_dir = 'models/children_asd_optimized'
# NEW: model_save_dir = '../models/training/children_asd_optimized'

# OLD: data_path = 'data/prepared/children_asd_43features.npy'
# NEW: data_path = '../data/processed/prepared/children_asd_43features.npy'
```

**Preprocessing Scripts** (in `scripts/preprocessing/`):
```python
# OLD: input_dir = 'data/autism/Eye-tracking Output'
# NEW: input_dir = '../data/raw/autism/Eye-tracking Output'

# OLD: output_file = 'data/prepared/children_asd_43features.npy'
# NEW: output_file = '../data/processed/prepared/children_asd_43features.npy'
```

**Conversion Scripts** (in `scripts/conversion/`):
```python
# OLD: model_path = 'models/children_asd_optimized/model.keras'
# NEW: model_path = '../models/training/children_asd_optimized/model.keras'

# OLD: output_dir = 'models/ACTIVE/children_asd_optimized_tfjs'
# NEW: output_dir = '../models/production/children_asd_optimized'
```

---

## Implementation Steps

### Phase 1: Copy Files (Safe)
1. Copy all files to new locations (keep originals)
2. Update all code references
3. Test functionality
4. Verify models load correctly

### Phase 2: Git Operations
1. Use `git mv` for important files to preserve history
2. Commit reorganization with detailed message
3. Update GitHub Pages configuration if needed

### Phase 3: Cleanup
1. Remove duplicate files from root
2. Update .gitignore for new structure
3. Create symbolic links if needed for backward compatibility

### Phase 4: Documentation
1. Update README.md with new structure
2. Add navigation to docs/
3. Create CONTRIBUTING.md with structure guidelines

---

## Benefits of New Structure

✅ **Clear Separation**: Docs, scripts, models, web app clearly separated  
✅ **Scalability**: Easy to add new models, scripts, documentation  
✅ **Professional**: Standard project layout for ML repositories  
✅ **Maintainability**: Easy to find and update files  
✅ **Collaboration**: Clear structure for contributors  
✅ **Deployment**: Production models clearly identified  

---

## Rollback Plan

If issues arise:
1. Original files preserved until Phase 3
2. Git history intact with `git mv`
3. Can revert commit if needed
4. Documentation maps old → new paths

---

**Next Action**: Begin Phase 1 - Copy files to new locations and update code references
