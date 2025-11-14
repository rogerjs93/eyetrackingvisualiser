
# ASD Baseline Models: Children vs Adult Comparison

## Model Architecture
Both models use identical architecture for fair comparison:
- **Encoder**: 28 → 32 (ReLU+BN+Dropout) → 48 (ReLU+BN+Dropout) → 24 (latent)
- **Decoder**: 24 → 48 (ReLU+BN+Dropout) → 32 (ReLU+BN) → 28 (output)
- **Total Parameters**: 8,020 per model
- **Optimizer**: Adam (learning rate: 0.00652)
- **Loss Function**: Mean Absolute Error (MAE)

## Dataset Information

### Children Model (Ages 3-12)
- **Source**: Eye-tracking Output CSV files
- **Participants**: 23 children with ASD
- **Citation**: Cilia et al. (2023)
- **Validation MAE**: 0.4069

### Adult Model
- **Source**: RawEyetrackingASD.mat file
- **Participants**: 24 adults
- **Trials**: 36 trials per participant
- **Validation MAE**: 0.6065

## Performance Comparison
✅ **Children model performs 49.1% better** (lower MAE)
- This could indicate children's gaze patterns are more stereotyped

**Absolute Difference**: 0.1996 MAE units


## Key Feature Differences

Top 5 features showing largest differences between age groups:

1. **Fixation_ratio**: Δ=11637985.01
   - Children: 11637986.00
   - Adult: 1.00

2. **Saccade_ratio**: Δ=8723035.59
   - Children: 8723035.59
   - Adult: 0.00

3. **Coverage_area**: Δ=3661546.53
   - Children: 4.17
   - Adult: 3661550.69

4. **Path_length**: Δ=532679.08
   - Children: 188585.13
   - Adult: 721264.22

5. **Sample_count**: Δ=502098.48
   - Children: 137.07
   - Adult: 502235.54


## Clinical Implications

1. **Developmental Differences**: Feature variations suggest age-related changes in gaze behavior
2. **Model Selection**: Use age-appropriate baseline for accurate ASD screening
3. **Future Research**: Investigate specific features driving age-group differences

## Deployment Status

✅ **Children Model**: Deployed to GitHub Pages  
✅ **Adult Model**: Trained and ready for deployment  
🎯 **Next Step**: Add age selection to web interface

## Files Generated

- `models/baseline_children_asd/optimized_autoencoder.keras`
- `models/baseline_adult_asd/adult_asd_baseline.keras`
- Both models include scaler artifacts for normalization

---
**Generated**: Automated comparison report
**Purpose**: Document age-specific ASD baseline models for research and deployment
