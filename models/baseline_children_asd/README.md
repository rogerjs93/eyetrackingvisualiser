# Children ASD Baseline Model (Ages 3-12)# 🧠 Autism Baseline Model



## 📚 Research FoundationThis directory contains a TensorFlow-based baseline model trained on 23 participants with autism spectrum disorder (ASD). The model learns typical autism gaze patterns and can be used to compare new eye-tracking data against this baseline.



**Primary Dataset:** Eye-tracking Dataset to Support the Research on Autism Spectrum Disorder  ## 📊 Model Information

**Source:** Cilia et al., ResearchGate  

**Link:** https://www.researchgate.net/publication/369708398_Eye-tracking_Dataset_to_Support_the_Research_on_Autism_Spectrum_Disorder  - **Model Type**: Autoencoder (unsupervised learning)

**Dataset:** https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism- **Architecture**: 28 → 64 → 32 → 16 (latent) → 32 → 64 → 28

- **Training Data**: 23 participants with ASD

## 👥 Participant Demographics- **Age Range**: 2.7 - 11.7 years

- **CARS Score Range**: 20.0 - 42.5

- **Age Range:** 3-12 years old- **Total Parameters**: 9,708

- **Total Participants:** 25 children with ASD- **Training Accuracy**: Validation MAE = 0.7196

- **Usable Data:** 23-24 participants (depending on format)

- **Data Quality:** Raw eye-tracking with fixations, saccades, gaze coordinates, pupil data## 📁 Files

- **Sampling Rate:** 60 Hz

- **Duration:** ~5 minutes per participant### `autism_baseline_model.keras` (13.6 MB)

- **Stimuli:** Simple, standardized (photos, short videos, cartoons)TensorFlow/Keras model file. This is the trained neural network that learns to reconstruct typical autism gaze patterns.



## 🎯 Clinical Relevance### `scaler.pkl` (2.3 KB)

StandardScaler for feature normalization. Ensures that new data is scaled the same way as the training data.

This baseline model is specifically designed for **early childhood ASD diagnosis** because:

### `baseline_statistics.json` (5.5 KB)

✅ **Age-appropriate:** Covers the critical early diagnosis window (3-12 years)  Statistical baseline including:

✅ **Balanced samples:** Includes both ASD and typically developing (TD) groups  - Mean, std, min, max, median for each of the 28 features

✅ **Clinical applicability:** Simple stimuli that can be reproduced in clinical settings  - Age and CARS score statistics

✅ **Standardized format:** Raw data ready for machine learning and diagnostic modeling  - Feature names and descriptions

✅ **Real-world use:** Designed explicitly for ASD research and diagnostic applications

### `model_metadata.json` (0.6 KB)

## 🤖 Model ArchitecturesModel metadata including:

- Training date

### Original Baseline Model- Number of participants

- **Architecture:** 28→64→32→16→32→64→28 (autoencoder)- Feature descriptions

- **Parameters:** 9,708- Usage instructions

- **Validation MAE:** 0.7196

- **File:** `autism_baseline_model.keras`## 🔬 How to Use



### Optimized Model (RECOMMENDED)### Basic Comparison

- **Architecture:** 28→32→48→24→48→32→28 (hyperparameter-tuned)

- **Parameters:** 8,020 (17% smaller)```python

- **Validation MAE:** 0.4069 (43.5% better than original)from baseline_comparator import BaselineComparator

- **Real-world improvement:** 30.7% better reconstruction on test data

- **Hyperparameters:**# Initialize comparator

  - Encoder 1: 32 neuronscomparator = BaselineComparator(model_dir='models/baseline')

  - Encoder 2: 48 neurons

  - Latent: 24 dimensions# Load your eye-tracking data

  - Dropout: 0.4# data should be a pandas DataFrame with columns: x, y, timestamp, duration

  - Learning Rate: 0.00652import pandas as pd

- **File:** `optimized_autoencoder.keras`data = pd.read_csv('your_data.csv')



## 📊 28 Features Extracted# Compare to baseline

results = comparator.compare_to_baseline(data)

Each participant's eye-tracking data is reduced to 28 features:

print(f"Similarity Score: {results['similarity_score']:.1f}/100")

1-4: X coordinate statistics (mean, std, min, max)  print(f"Deviation Level: {results['deviation_level']}")

5-8: Y coordinate statistics (mean, std, min, max)  print(f"Interpretation: {results['deviation_interpretation']}")

9-12: Velocity statistics (mean, std, max, median)  ```

13-15: Acceleration statistics (mean, std, max)  

16-17: Fixation metrics (ratio, mean velocity during fixations)  ### Generate Detailed Report

18-19: Saccade metrics (ratio, mean velocity during saccades)  

20-23: Gaze distribution (X/Y percentiles 25th, 75th)  ```python

24: Path length (total distance traveled)  # Generate markdown report

25: Area covered (gaze spatial coverage)  report = comparator.generate_comparison_report(

26-27: Velocity variability (90th, 10th percentiles)      data, 

28: Sample count (data richness indicator)    output_path='comparison_report.md'

)

## 🔬 Comparison to Adult Dataset```



### Dataset A (Ramot et al., Nature - NOT USED HERE)## 📈 Features Extracted

**Link:** https://nih.figshare.com/articles/dataset/Eye_tracking_data_for_participants_with_Autism_Spectrum_Disorders/10324877

The model analyzes 28 features across multiple dimensions:

**Why not used for children's baseline:**

- ❌ Age range: 15-30 years (adolescents/young adults)### Spatial Features (10)

- ❌ Different development stage: Neural mechanisms differ from children- X/Y position statistics (mean, std, min, max, range)

- ❌ Complex stimuli: Movie clips (harder to standardize clinically)

- ❌ Small ASD sample: 36 males only### Temporal Features (7)

- ✅ High sampling rate: 1000 Hz (good for research)- Duration statistics (mean, std, median, quartiles)

- ✅ fMRI integration: Links gaze to neural activity- Time span and sampling rate



**Use case for adult dataset:**### Movement Features (7)

- Research on neural mechanisms of social orienting- Path length, efficiency, velocities

- Studying gaze patterns in older individuals- Distance statistics

- High-precision scan-path analysis

- **Could be used to create a separate "Adult ASD Baseline Model"**### Distribution Features (4)

- Spatial entropy

## 🎓 Scientific Rationale- Concentration metrics (distance from center)



### Why Children's Baseline is Separate## 🎯 Interpretation Guide



1. **Developmental differences:** Children's eye movements differ significantly from adults### Similarity Score (0-100)

2. **Clinical window:** Early diagnosis (3-12 years) has the most intervention impact- **90-100**: Very similar to typical autism patterns

3. **Stimulus differences:** Children respond better to simple, engaging stimuli- **70-89**: Moderately similar

4. **Sampling requirements:** 60 Hz sufficient for clinical screening in children- **50-69**: Some similarities with notable differences

5. **Standardization:** Simple stimuli enable consistent clinical deployment- **Below 50**: Significantly different from baseline



### Model Performance### Deviation Level

- **Low** (Z-score < 1.0): Pattern falls within typical autism range

**Training Results:**- **Moderate** (Z-score 1.0-2.0): Some unusual characteristics

- Trained on 23 ASD children (ages 3-12)- **High** (Z-score > 2.0): Significantly different from typical autism patterns

- Learned to reconstruct autism-specific eye-tracking patterns

- Lower MAE = better learned the "baseline" autism patterns### Z-Scores

- Can detect when new data differs from typical ASD patternsIndividual feature Z-scores show how many standard deviations away from the baseline mean:

- **|Z| < 1**: Within normal variation (68%)

**Validation:**- **|Z| < 2**: Moderate deviation (95%)

- Original model: MAE 0.7578 on autism data- **|Z| > 2**: Significant deviation (outlier)

- Optimized model: MAE 0.5250 on autism data

- **30.7% improvement in pattern recognition**## 🔧 Updating the Model



## 📝 UsageTo retrain the model with new participants:



### For Clinical Screening (Ages 3-12)```bash

Use this baseline model to:python baseline_model_builder.py

1. Upload child's eye-tracking data (ages 3-12)```

2. Calculate reconstruction error

3. Compare to autism baselineThis will:

4. **Low error** = gaze patterns similar to ASD children1. Load all participants from `data/autism/`

5. **High error** = gaze patterns differ from ASD baseline2. Extract features from each participant

3. Train a new autoencoder model

### For Research4. Save updated model files

- Study autism-specific gaze patterns in children

- Compare typically developing vs ASD## 📚 Research Applications

- Track changes with intervention

- Identify subgroups within ASD population### Clinical Assessment

- Compare individual patients against population baseline

## 🔗 Related Files- Track changes over time

- Identify atypical patterns requiring attention

- `autism_baseline_model.keras` - Original model

- `optimized_autoencoder.keras` - Improved model (RECOMMENDED)### Intervention Evaluation

- `scaler.pkl` - Feature normalization parameters- Measure pre/post intervention changes

- `baseline_statistics.json` - Mean/std of all 28 features- Quantify treatment effectiveness

- `comparison_results.json` - Performance comparison- Monitor progress



## 📖 Citations### Comparative Studies

- Compare ASD vs. neurotypical populations

**Primary Dataset:**- Age-based comparisons

Cilia, F., et al. (2023). Eye-tracking Dataset to Support the Research on Autism Spectrum Disorder. ResearchGate.  - Severity correlation analysis

https://www.researchgate.net/publication/369708398

## ⚠️ Important Notes

**Adult Dataset (for comparison):**

Ramot, M., et al. (2019). Eye tracking data for participants with Autism Spectrum Disorders. Figshare/Nature.  1. **Dataset Limitation**: Model trained on 23 participants with specific age and CARS score ranges

https://nih.figshare.com/articles/dataset/103248772. **Generalization**: May not generalize well outside training data distribution

3. **Complementary Tool**: Should be used alongside clinical judgment, not as sole diagnostic tool

## ⚠️ Important Notes4. **Data Quality**: Requires properly formatted eye-tracking data with x, y, timestamp, duration columns



- This model is trained ONLY on children ages 3-12 with ASD## 📖 Citation

- Do not use for adults or older adolescents (use adult dataset for that)

- Results should be interpreted by qualified cliniciansIf you use this baseline model in research, please cite:

- Model is for research and screening support, not standalone diagnosis

- Always combine with clinical assessment and other diagnostic tools**Dataset Source**:

- Eye Tracking Autism Dataset by IMT Kaggle Team

## 🚀 Future Work- Available at: https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism



1. **Create Adult ASD Baseline Model** using Ramot et al. dataset (ages 15-30)**Model**:

2. **Age-specific models** for different developmental stages- Autism Baseline Gaze Pattern Model

3. **Cross-dataset validation** between children and adult patterns- Eye-Tracking Data Visualizer Project

4. **Longitudinal tracking** of how patterns change with age- GitHub: https://github.com/rogerjs93/eyetrackingvisualiser

5. **Intervention response** modeling

## 📧 Support

---

For questions or issues with the baseline model:

**Model Version:** Optimized v1.0  1. Check that your data format matches the expected structure

**Last Updated:** November 12, 2025  2. Ensure all 28 features can be extracted from your data

**Recommended for:** Clinical screening and research in children ages 3-123. Verify that timestamps are properly normalized (starting from 0)


## 🔄 Version History

- **v1.0** (November 2025): Initial release with 23 ASD participants
