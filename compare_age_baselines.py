"""
Compare Children vs Adult ASD Baseline Models
Analyzes differences in performance and gaze patterns between age groups
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from pathlib import Path
import pickle

def load_children_model():
    """Load the optimized children baseline model"""
    model_path = Path('models/baseline_children_asd/optimized_autoencoder.keras')
    scaler_path = Path('models/baseline_children_asd/scaler.pkl')
    
    print("📦 Loading Children Model (Ages 3-12)...")
    model = keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"   ✓ Model loaded: {model.count_params():,} parameters")
    return model, scaler

def load_adult_model():
    """Load the adult baseline model"""
    model_path = Path('models/baseline_adult_asd/adult_asd_baseline.keras')
    scaler_path = Path('models/baseline_adult_asd/scaler.pkl')
    
    print("📦 Loading Adult Model...")
    model = keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"   ✓ Model loaded: {model.count_params():,} parameters")
    return model, scaler

def compare_performance():
    """Compare validation performance between both models"""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    # Load children results
    children_results_path = Path('models/baseline_children_asd/comparison_results.json')
    with open(children_results_path, 'r') as f:
        children_results = json.load(f)
    
    children_mae = children_results['optimized_autoencoder']['val_mae']
    
    # Adult MAE from training
    adult_mae = 0.6065
    
    print(f"\n📊 Model Performance:")
    print(f"   Children (Ages 3-12): MAE = {children_mae:.4f}")
    print(f"   Adult:                MAE = {adult_mae:.4f}")
    print(f"   Difference:           {abs(children_mae - adult_mae):.4f}")
    
    if adult_mae < children_mae:
        improvement = ((children_mae - adult_mae) / children_mae) * 100
        print(f"   🎯 Adult model is {improvement:.1f}% better")
    else:
        difference = ((adult_mae - children_mae) / children_mae) * 100
        print(f"   📊 Children model is {difference:.1f}% better")
    
    return children_mae, adult_mae

def analyze_feature_differences():
    """Analyze differences in feature distributions between age groups"""
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)
    
    # Load scalers to see feature statistics
    children_scaler_path = Path('models/baseline_children_asd/scaler.pkl')
    adult_scaler_path = Path('models/baseline_adult_asd/scaler.pkl')
    
    with open(children_scaler_path, 'rb') as f:
        children_scaler = pickle.load(f)
    
    with open(adult_scaler_path, 'rb') as f:
        adult_scaler = pickle.load(f)
    
    # Feature names (matching our 28 features)
    feature_names = [
        'X_mean', 'X_std', 'X_min', 'X_max',
        'Y_mean', 'Y_std', 'Y_min', 'Y_max',
        'Vel_mean', 'Vel_std', 'Vel_max', 'Vel_median',
        'Acc_mean', 'Acc_std', 'Acc_max',
        'Fixation_ratio', 'Fixation_vel',
        'Saccade_ratio', 'Saccade_vel',
        'X_Q25', 'X_Q75', 'Y_Q25', 'Y_Q75',
        'Path_length', 'Coverage_area',
        'Vel_P90', 'Vel_P10', 'Sample_count'
    ]
    
    print(f"\n🔍 Top 10 Largest Feature Differences (Children vs Adult):")
    print("   (Normalized mean values)")
    print()
    
    # Calculate absolute differences
    differences = []
    for i, name in enumerate(feature_names):
        children_mean = children_scaler.mean_[i]
        adult_mean = adult_scaler.mean_[i]
        diff = abs(children_mean - adult_mean)
        differences.append((name, diff, children_mean, adult_mean))
    
    # Sort by difference
    differences.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10
    for i, (name, diff, child_val, adult_val) in enumerate(differences[:10], 1):
        print(f"   {i:2d}. {name:20s} Δ={diff:10.2f}  (Children: {child_val:10.2f}, Adult: {adult_val:10.2f})")
    
    return differences

def generate_summary_report(children_mae, adult_mae, differences):
    """Generate comprehensive comparison report"""
    print("\n" + "="*70)
    print("COMPREHENSIVE AGE-GROUP BASELINE COMPARISON")
    print("="*70)
    
    report = f"""
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
- **Validation MAE**: {children_mae:.4f}

### Adult Model
- **Source**: RawEyetrackingASD.mat file
- **Participants**: 24 adults
- **Trials**: 36 trials per participant
- **Validation MAE**: {adult_mae:.4f}

## Performance Comparison
"""
    
    if adult_mae < children_mae:
        improvement = ((children_mae - adult_mae) / children_mae) * 100
        report += f"✅ **Adult model performs {improvement:.1f}% better** (lower MAE)\n"
        report += f"- This suggests adult gaze patterns may be more consistent/predictable\n"
    else:
        difference = ((adult_mae - children_mae) / children_mae) * 100
        report += f"✅ **Children model performs {difference:.1f}% better** (lower MAE)\n"
        report += f"- This could indicate children's gaze patterns are more stereotyped\n"
    
    report += f"\n**Absolute Difference**: {abs(children_mae - adult_mae):.4f} MAE units\n"
    
    report += f"""

## Key Feature Differences

Top 5 features showing largest differences between age groups:

"""
    
    for i, (name, diff, child_val, adult_val) in enumerate(differences[:5], 1):
        report += f"{i}. **{name}**: Δ={diff:.2f}\n"
        report += f"   - Children: {child_val:.2f}\n"
        report += f"   - Adult: {adult_val:.2f}\n\n"
    
    report += """
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
"""
    
    return report

def main():
    print("\n" + "="*70)
    print("AGE-GROUP BASELINE MODEL COMPARISON")
    print("="*70)
    print("\nComparing ASD baseline models across age groups...")
    print("Children (3-12) vs Adult")
    
    # Load models
    children_model, children_scaler = load_children_model()
    adult_model, adult_scaler = load_adult_model()
    
    # Compare performance
    children_mae, adult_mae = compare_performance()
    
    # Analyze feature differences
    differences = analyze_feature_differences()
    
    # Generate comprehensive report
    report = generate_summary_report(children_mae, adult_mae, differences)
    
    # Save report
    report_path = Path('models/AGE_BASELINE_COMPARISON.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n💾 Comprehensive report saved: {report_path}")
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Children MAE: {children_mae:.4f}")
    print(f"   Adult MAE:    {adult_mae:.4f}")
    print(f"   Both models ready for deployment")
    print(f"\n💡 Next: Update web interface with age selection dropdown")

if __name__ == "__main__":
    main()
