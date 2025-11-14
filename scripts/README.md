# Python Scripts

**Collection of Python scripts for training, preprocessing, conversion, and analysis**

---

## 📁 Script Categories

### 🎓 [Training](training/)
Model training scripts for different age groups and strategies

- `train_optimized_model.py` - Train optimized model for small datasets (20 features)
- `train_enhanced_model.py` - Train enhanced model with all 43 features
- `train_adult_baseline.py` - Train adult ASD baseline model
- `train_neurotypical_baseline.py` - Train neurotypical baseline model
- `baseline_model_builder.py` - Baseline model architecture builder
- `advanced_model_trainer.py` - Advanced training with hyperparameter tuning

### 🔧 [Preprocessing](preprocessing/)
Data preparation and feature extraction

- `prepare_training_data.py` - Prepare training data with 43 features
- `autism_data_loader.py` - Load and validate autism datasets
- `download_adult_dataset.py` - Download adult ASD dataset
- `sample_data_generator.py` - Generate synthetic test data

### 🔄 [Conversion](conversion/)
Model format conversion (Keras ↔ TensorFlow.js)

- `export_optimized_model.py` - Export optimized model to TFJS (production)
- `export_model.py` - General model export utility
- `convert_models_to_tfjs.py` - Batch convert multiple models
- `convert_model_to_tfjs.py` - Single model conversion
- `convert_optimized_model.py` - Optimized model specific conversion
- `convert_scaler.py` - Convert scikit-learn scaler to JSON
- `manual_tfjs_converter.py` - Manual TFJS conversion (custom exporter)
- `simple_tfjs_converter.py` - Simple TFJS conversion
- `tfjs_clean_converter.py` - Clean TFJS conversion
- `convert_via_savedmodel.py` - Convert via SavedModel format
- `convert_via_saved_model.py` - Alternative SavedModel converter

### 📊 [Analysis](analysis/)
Data analysis and model comparison

- `baseline_comparator.py` - Compare against baseline models
- `comparative_analysis.py` - Comparative analysis across models
- `compare_age_baselines.py` - Compare age-specific baselines
- `compare_mat_vs_csv.py` - Compare MATLAB vs CSV data formats
- `pattern_recognition.py` - Pattern recognition analysis
- `cognitive_load.py` - Cognitive load analysis

### 🛠️ [Utilities](utilities/)
Helper scripts and tools

- `analyze_mat_structure.py` - Analyze MATLAB file structure
- `inspect_mat_file.py` - Inspect MATLAB file contents
- `check_weights.py` - Check model weights and structure

---

## 🚀 Quick Start

### Environment Setup

```bash
# Navigate to project root
cd "c:\Users\roger\Desktop\Roger\Projects\Software engineering\python\Pythondata visualizer"

# Install dependencies
pip install -r requirements.txt
```

### Common Workflows

#### 1. Train Optimized Model (Small Dataset)

```bash
cd scripts/training
python train_optimized_model.py
```

**Path Configuration:**
- Input data: `../data/processed/prepared/children_asd_43features.npy`
- Output model: `../models/training/children_asd_optimized/`
- Uses 20 selected features via mutual information

#### 2. Preprocess New Data

```bash
cd scripts/preprocessing
python prepare_training_data.py --input "../data/raw/autism/" --output "../data/processed/prepared/"
```

**Extracts 43 features:**
- 28 core features (spatial, temporal, movement, attention)
- 15 advanced features (entropy, autocorrelation, cluster density)

#### 3. Convert Model to TFJS

```bash
cd scripts/conversion
python export_optimized_model.py --input "../models/training/children_asd_optimized/" --output "../models/production/children_asd_optimized/"
```

**Outputs:**
- `model.json` - Model architecture
- `group1-shard1of1.bin` - Model weights
- `scaler.json` - Feature scaler
- `preprocessing.json` - Preprocessing config

#### 4. Compare Models

```bash
cd scripts/analysis
python compare_age_baselines.py
```

**Compares:**
- Children ASD baseline
- Adult ASD baseline
- Neurotypical baseline
- Outputs MAE, R², performance metrics

---

## 📋 Path Conventions

All scripts use **relative paths** from their subdirectories:

```python
# From scripts/training/
data_path = "../data/processed/prepared/children_asd_43features.npy"
model_save_path = "../models/training/children_asd_optimized/"

# From scripts/preprocessing/
input_path = "../data/raw/autism/"
output_path = "../data/processed/prepared/"

# From scripts/conversion/
source_model = "../models/training/children_asd_optimized/"
output_model = "../models/production/children_asd_optimized/"
```

---

## 🔧 Development Guidelines

### Adding New Scripts

1. Place in appropriate subdirectory (training/, preprocessing/, etc.)
2. Use relative paths from script location
3. Add command-line arguments with `argparse`
4. Include docstring with usage example
5. Update this README with script description

### Code Style

- **PEP 8** compliance
- **Type hints** for functions
- **Docstrings** for all modules/classes/functions
- **Error handling** with try-except
- **Logging** instead of print statements

### Testing Scripts

```bash
# Unit test
cd ../../tests/unit
python test_data_format.py

# Integration test
cd ../integration
python test_autism_integration.py
```

---

## 📊 Script Dependencies

### Core Dependencies

```
tensorflow==2.15.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.11.1
```

### Optional Dependencies

```
matplotlib==3.7.2  # For visualization scripts
tensorflowjs==4.11.0  # For conversion scripts
h5py==3.9.0  # For HDF5 data
scipy.io  # For MATLAB files
```

---

## 🎯 Script-Specific Documentation

### Training Scripts

**train_optimized_model.py**
- Uses feature selection (20 of 43 features)
- Optimized for small datasets (20-50 samples)
- 91% parameter reduction
- MAE: 0.4231 (only 4% worse than baseline)

**train_enhanced_model.py**
- Uses all 43 features
- Requires large datasets (1000+ samples)
- More complex architecture
- Better performance on large data

### Preprocessing Scripts

**prepare_training_data.py**
- Extracts 43 clinical features
- Validates data format
- Handles missing values
- Outputs NumPy arrays

**autism_data_loader.py**
- Loads CSV eye-tracking data
- Validates Point of Regard format
- Filters invalid fixations
- Returns pandas DataFrame

### Conversion Scripts

**export_optimized_model.py** (PRODUCTION)
- Custom exporter bypassing numpy version issues
- Exports to TensorFlow.js format
- Includes scaler and preprocessing config
- Optimized for browser deployment

**convert_models_to_tfjs.py** (BATCH)
- Converts multiple models at once
- Useful for updating all age groups
- Validates output format

---

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Ensure you're in the correct directory
cd scripts/training  # or preprocessing, conversion, analysis
python your_script.py
```

**2. File Not Found Errors**
```bash
# Scripts use relative paths from their subdirectory
# ../data/ refers to data/ at project root
# ../models/ refers to models/ at project root
```

**3. NumPy Version Conflicts**
```bash
# Use export_optimized_model.py instead of tensorflowjs_converter
cd scripts/conversion
python export_optimized_model.py
```

**4. MATLAB File Issues**
```bash
# Use scipy.io.loadmat
cd scripts/utilities
python inspect_mat_file.py --file "../../data/raw/autism/adult_data.mat"
```

---

## 📈 Performance Tips

1. **Use feature selection** for small datasets (<100 samples)
2. **Use GPU** for training large models (TensorFlow GPU)
3. **Batch conversion** for multiple models
4. **Cache preprocessed data** to avoid recomputation
5. **Use validation split** (80/20) for hyperparameter tuning

---

## 🔄 Script Updates

When updating scripts after reorganization:

```python
# OLD (from root directory)
data_path = "data/prepared/children_asd_43features.npy"

# NEW (from scripts/training/)
data_path = "../data/processed/prepared/children_asd_43features.npy"
```

---

## 📞 Support

For script-specific issues:

1. Check script docstring for usage
2. Run with `--help` flag for arguments
3. Check logs in `logs/` directory
4. Consult [documentation](../docs/)
5. Open GitHub issue with error message

---

**Maintained by**: Project team  
**Last Updated**: Current session (repository reorganization)
