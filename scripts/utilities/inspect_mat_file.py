"""
Inspect the RawEyetrackingASD.mat file to understand its structure
"""
import scipy.io
import numpy as np

print("="*70)
print("Inspecting RawEyetrackingASD.mat")
print("="*70)

# Load the .mat file
mat_file_path = r"data\autism\autismdata2\RawEyetrackingASD.mat"
print(f"\n📂 Loading: {mat_file_path}")

try:
    mat_data = scipy.io.loadmat(mat_file_path)
    
    print("\n✅ File loaded successfully!")
    print(f"\n📊 Top-level keys in the file:")
    print("-" * 70)
    
    for key in mat_data.keys():
        if not key.startswith('__'):  # Skip MATLAB metadata
            value = mat_data[key]
            print(f"\n  Key: '{key}'")
            print(f"    Type: {type(value)}")
            
            if isinstance(value, np.ndarray):
                print(f"    Shape: {value.shape}")
                print(f"    Dtype: {value.dtype}")
                
                # Show some sample data
                if value.size < 20 and value.ndim <= 2:
                    print(f"    Data preview:\n{value}")
                elif value.ndim == 1:
                    print(f"    First 5 values: {value[:5]}")
                elif value.ndim == 2:
                    print(f"    First few rows/cols:\n{value[:3, :min(5, value.shape[1])]}")
                else:
                    print(f"    Multi-dimensional array, showing first element shape: {value[0].shape if value.size > 0 else 'empty'}")
    
    # Check if it's structured array
    print("\n" + "="*70)
    print("Analysis:")
    print("="*70)
    
    non_meta_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    
    if len(non_meta_keys) == 1:
        main_key = non_meta_keys[0]
        main_data = mat_data[main_key]
        
        print(f"\n📌 Single main data structure: '{main_key}'")
        
        if isinstance(main_data, np.ndarray):
            if main_data.dtype.names:  # Structured array
                print(f"\n🔍 This is a STRUCTURED ARRAY with fields:")
                for field_name in main_data.dtype.names:
                    field_data = main_data[field_name]
                    print(f"    • {field_name}: shape={field_data.shape}, dtype={field_data.dtype}")
            else:
                print(f"\n🔍 This is a regular array")
                print(f"    Likely contains: {'Individual participant data' if main_data.shape[0] == 25 else 'Combined/aggregated data'}")
                
                if main_data.shape[0] == 25:
                    print(f"    → 25 rows = 25 participants (matches dataset size)")
                    print(f"    → This is likely INDIVIDUAL participant data, NOT a baseline")
                elif main_data.shape[0] == 1:
                    print(f"    → 1 row = Could be aggregated baseline statistics")
                else:
                    print(f"    → {main_data.shape[0]} rows")
    
    print("\n" + "="*70)
    print("Conclusion:")
    print("="*70)
    
    # Provide interpretation
    if 'RawData' in mat_data or 'rawData' in mat_data:
        print("\n✅ This file contains RAW eye-tracking data (not a baseline model)")
        print("   Each entry is likely individual participant measurements")
    elif any(k in mat_data for k in ['mean', 'std', 'baseline', 'average']):
        print("\n✅ This file contains BASELINE/AGGREGATED statistics")
        print("   Pre-calculated mean/std values from all participants")
    else:
        main_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
        main_shape = mat_data[main_key].shape
        if main_shape[0] == 25:
            print("\n✅ This appears to be INDIVIDUAL participant data (25 participants)")
            print("   NOT a baseline model - raw data from each participant")
        else:
            print("\n⚠️  Structure unclear - manual inspection needed")
    
except Exception as e:
    print(f"\n❌ Error loading file: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
