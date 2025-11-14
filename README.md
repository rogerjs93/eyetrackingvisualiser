# Eye-Tracking Visualizer for ASD Research

**Advanced eye-tracking analysis and baseline comparison for Autism Spectrum Disorder (ASD) research**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.11-orange)](https://www.tensorflow.org/js)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

---

## 🎯 Project Overview

This project provides **browser-based eye-tracking analysis** with machine learning-powered baseline comparison for ASD research. It features:

- **43 clinically-validated eye-tracking features** for comprehensive gaze pattern analysis
- **Age-specific baseline models** (children, adult, neurotypical)
- **Optimized lightweight models** for browser deployment (<20KB)
- **Real-time visualization** of gaze patterns and metrics
- **Privacy-first design** - all processing happens in the browser

### Key Features

✅ **Browser-Based AI** - No server required, works offline  
✅ **Clinical Features** - 43 features including saccade entropy, spatial autocorrelation, cluster density  
✅ **Dual Strategy** - Lightweight models for small datasets, complex models for large datasets  
✅ **Age-Specific** - Separate baselines for children (2-12 years), adults, neurotypical  
✅ **Fast Inference** - <100ms prediction time, <2s model load  

---

## 📁 Repository Structure

```
eyetrackingvisualiser/
│
├── 📚 docs/                         # Documentation
│   ├── methodology/                 # Research & feature engineering
│   ├── deployment/                  # Deployment guides
│   ├── datasets/                    # Data documentation
│   ├── guides/                      # User guides & tutorials
│   └── history/                     # Historical reports
│
├── 🐍 scripts/                      # Python scripts
│   ├── training/                    # Model training
│   ├── preprocessing/               # Data preparation
│   ├── conversion/                  # Model format conversion
│   ├── analysis/                    # Data analysis
│   └── utilities/                   # Helper scripts
│
├── 🌐 web/                          # Browser application
│   ├── js/                          # JavaScript modules
│   ├── html/                        # HTML files
│   └── assets/                      # Images, icons
│
├── 🤖 models/                       # Trained models
│   ├── production/                  # Production TFJS models
│   │   ├── children_asd_optimized/  # Optimized children (20 features)
│   │   ├── adult_asd/               # Adult ASD baseline
│   │   └── neurotypical/            # Neurotypical baseline
│   ├── training/                    # Keras training models
│   └── archive/                     # Legacy models
│
├── 📊 data/                         # Datasets
│   ├── raw/                         # Original data
│   │   ├── autism/                  # ASD datasets
│   │   └── standard/                # Neurotypical datasets
│   └── processed/                   # Prepared training data
│
├── 🧪 tests/                        # Test files
│   ├── browser/                     # Browser tests
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
│
├── index.html                       # Main web application
├── baseline_model_web.js            # Model interface (legacy location)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Option 1: Use the Live Demo (Recommended)

Visit: **https://rogerjs93.github.io/eyetrackingvisualiser/**

1. Upload your eye-tracking CSV file (Point of Regard format)
2. View gaze pattern visualization
3. Compare against age-appropriate baseline
4. Get similarity score and clinical insights

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/rogerjs93/eyetrackingvisualiser.git
cd eyetrackingvisualiser

# Start a local web server
python -m http.server 8000

# Open in browser
# http://localhost:8000
```

### Option 3: Python Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train new models
cd scripts/training
python train_optimized_model.py

# Preprocess data
cd ../preprocessing
python prepare_training_data.py

# Run analysis
cd ../analysis
python baseline_comparator.py
```

---

## 📖 Documentation

### User Guides
- [**Quick Start Guide**](docs/guides/quickstart.md) - Get started in 5 minutes
- [**Setup Guide**](docs/guides/setup_guide.md) - Detailed installation instructions
- [**Autism Data Guide**](docs/guides/autism_quickstart.md) - Working with ASD datasets

### Methodology
- [**Research Methodology**](docs/methodology/research_methodology.md) - Feature engineering approach
- [**Feature Engineering**](docs/methodology/feature_engineering.md) - 43 features explained
- [**Optimization Guide**](docs/methodology/optimization_guide.md) - Model optimization for small datasets
- [**Phase 2 Validation**](docs/methodology/phase2_validation_results.md) - Feature validation results

### Deployment
- [**Optimized Model Deployment**](docs/deployment/optimized_model_deployment.md) - Production deployment guide
- [**Dual Strategy Plan**](docs/deployment/dual_strategy_plan.md) - Small vs large dataset approaches
- [**Age-Specific Deployment**](docs/deployment/age_specific_deployment.md) - Age group baselines
- [**Web AI Implementation**](docs/deployment/web_ai_implementation.md) - Browser ML integration

### Datasets
- [**Dataset Overview**](docs/datasets/dataset_overview.md) - Available datasets
- [**Autism Data README**](docs/datasets/autism_data_readme.md) - ASD dataset details
- [**Adult Dataset Instructions**](docs/datasets/adult_dataset_instructions.md) - Adult ASD data

---

## 🤖 Models

### Production Models (TensorFlow.js)

| Model | Features | Parameters | MAE | Size | Use Case |
|-------|----------|------------|-----|------|----------|
| **Children ASD (Optimized)** | 20 | 1,084 | 0.4231 | 18.6 KB | Default, ages 2-12 |
| **Adult ASD** | 28 | 3,500 | 0.6065 | 45 KB | Ages 13+ |
| **Neurotypical** | 28 | 3,500 | 0.3478 | 45 KB | Comparison baseline |

### Model Architecture

**Optimized Children Model** (20→16→8→16→20):
- Feature selection via mutual information
- L2 regularization (0.001)
- 30% dropout
- 91% fewer parameters than original
- Only 4% worse than baseline (vs 66% worse for unoptimized)

---

## 🔬 Features Extracted

### Core Features (28)
- Spatial: x/y mean, std, min, max, coverage, dispersion
- Temporal: fixation duration, count, temporal consistency
- Movement: saccade velocity/amplitude mean/std
- Attention: center bias, edge bias, ROI focus, attention switches
- Patterns: scan path length, gaze entropy, revisit rate

### Advanced Features (15)
1. **Saccade Directional Entropy** - Scanning pattern diversity
2. **Spatial Autocorrelation (X/Y)** - Gaze predictability
3. **Fixation Cluster Density** - Interest area concentration
4. **First Fixation Center Bias** - Initial attention allocation
5. **Spatial Revisitation Rate** - Repetitive viewing
6. **Velocity Skewness/Kurtosis** - Movement planning
7. **ISI Coefficient of Variation** - Timing consistency
8. **Ambient/Focal Attention Ratio** - Processing style
9. **Saccade Amplitude Entropy** - Movement diversity
10. **Scanpath Efficiency** - Visual search efficiency
11. **Fixation Duration Entropy** - Processing variability
12. **Cross-Correlation XY** - Coordinated movement
13. **Peak Velocity Ratio** - Ballistic vs corrective saccades

---

## 🎓 Research & Citations

This project implements features from peer-reviewed ASD eye-tracking research:

- Directional entropy for atypical scanning patterns (Clinical Psychology Review, 2019)
- Spatial autocorrelation for attention stability (Journal of Autism, 2020)
- Cluster density for social stimulus focus (Autism Research, 2018)
- ISI variability for attention shifts (J. Child Psychology, 2021)

### Data Sources

- **Children ASD Dataset**: 25 participants, ages 2.7-12.3, CARS scores 17-45 (Kaggle)
- **Neurotypical Dataset**: 1000+ participants, 2.4M+ fixations, 20+ studies (Dryad)
- **Adult ASD Dataset**: 24 participants (MATLAB format)

---

## 📊 Data Format

### Required CSV Format (Point of Regard)

```csv
GazePointX (ADCSpx),GazePointY (ADCSpx),Timestamp (ms),FixationDuration (ms)
500.5,400.2,1234567890,250
520.1,405.8,1234568140,180
...
```

### Supported Formats
- Point of Regard (default)
- Raw gaze coordinates
- Fixation data
- Custom CSV with x, y, timestamp columns

---

## 🛠️ Development

### Training New Models

```bash
cd scripts/training

# For small datasets (20-50 samples)
python train_optimized_model.py --strategy 5

# For large datasets (1000+ samples)
python train_enhanced_model.py --data neurotypical
```

### Adding New Features

1. Add feature calculation to `web/js/baseline_model_web.js` (extractFeatures)
2. Update Python preprocessing in `scripts/preprocessing/prepare_training_data.py`
3. Retrain models with new feature count
4. Update documentation in `docs/methodology/`

### Converting Models to TFJS

```bash
cd scripts/conversion
python export_optimized_model.py
```

---

## 🧪 Testing

### Browser Tests

```bash
# Start local server
python -m http.server 8000

# Open test suite
open http://localhost:8000/tests/browser/test_optimized_model.html
```

### Python Tests

```bash
# Unit tests
python tests/unit/test_data_format.py

# Integration tests
python tests/integration/test_autism_integration.py
```

---

## 📈 Performance

### Model Performance

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Load Time | <2s | 1.2s | ✅ |
| Inference Time | <100ms | 45ms | ✅ |
| Model Size | <20KB | 18.6KB | ✅ |
| MAE (Children) | <0.45 | 0.4231 | ✅ |
| Browser Support | Modern | Chrome/Firefox/Edge/Safari | ✅ |

### Optimization Results

- **Parameter Reduction**: 11,643 → 1,084 (91% reduction)
- **Feature Selection**: 43 → 20 features (53% reduction)
- **Performance**: 66% worse → 4% worse than baseline (62% improvement)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- [ ] Additional clinical features
- [ ] Support for more data formats
- [ ] Advanced visualizations
- [ ] Model interpretability
- [ ] Multi-language support
- [ ] Mobile optimization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- TensorFlow.js team for browser ML capabilities
- Eye-tracking research community for feature validation
- Kaggle & Dryad for public ASD datasets
- Open-source contributors

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/rogerjs93/eyetrackingvisualiser/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rogerjs93/eyetrackingvisualiser/discussions)
- **Email**: [Project maintainer]

---

## 🗺️ Roadmap

### Phase 1: Core Features ✅ Complete
- [x] 43-feature extraction
- [x] Optimized children model
- [x] Browser deployment
- [x] Repository reorganization

### Phase 2: Large Dataset Training 🔄 In Progress
- [ ] Process neurotypical dataset (1000+ participants)
- [ ] Train complex model (43 features, 25K parameters)
- [ ] Deploy neurotypical baseline
- [ ] Performance comparison

### Phase 3: Advanced Features 📋 Planned
- [ ] Adult ASD model optimization
- [ ] Multi-modal analysis (gaze + facial expressions)
- [ ] Temporal sequence analysis
- [ ] Clinical report generation
- [ ] API for programmatic access

### Phase 4: Research Tools 💡 Future
- [ ] Experiment designer
- [ ] Batch processing
- [ ] Statistical analysis tools
- [ ] Publication-ready visualizations
- [ ] Dataset contribution system

---

## 📚 Additional Resources

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [Eye-Tracking Analysis Best Practices](docs/methodology/)
- [ASD Research Guidelines](docs/datasets/)
- [Model Training Tutorials](docs/guides/)

---

**Built with ❤️ for the ASD research community**

*Last Updated: November 14, 2025*
