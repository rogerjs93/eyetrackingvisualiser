"""
Manual TensorFlow.js export script for the enhanced model.
This bypasses the tensorflowjs library bugs with numpy 2.x
"""
import tensorflow as tf
import json
import os
import numpy as np
from pathlib import Path

def export_model_manually():
    """Export the Keras model to TensorFlow.js format manually"""
    
    # Load the OPTIMIZED model
    model_path = 'models/children_asd_optimized/model.keras'
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create output directory
    output_dir = 'models/ACTIVE/children_asd_optimized_tfjs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as SavedModel format first
    saved_model_path = os.path.join(output_dir, 'saved_model')
    print(f"Saving as SavedModel to: {saved_model_path}")
    tf.saved_model.save(model, saved_model_path)
    
    # Now use tensorflowjs_converter command line tool
    import subprocess
    cmd = [
        'tensorflowjs_converter',
        '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model',
        '--signature_name', 'serving_default',
        '--saved_model_tags', 'serve',
        saved_model_path,
        output_dir
    ]
    
    print(f"\nRunning converter command:")
    print(' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✅ Model exported successfully!")
        print(f"📁 Output directory: {output_dir}")
        
        # List exported files
        print("\n📋 Exported files:")
        for file in sorted(Path(output_dir).glob('*')):
            if file.is_file():
                size = file.stat().st_size / 1024  # KB
                print(f"  - {file.name} ({size:.1f} KB)")
    else:
        print("\n❌ Export failed!")
        print(f"Error: {result.stderr}")
        return False
    
    # Load and export scaler parameters
    print("\n📊 Exporting scaler parameters...")
    scaler_path = 'models/children_asd_optimized/scaler.pkl'
    if os.path.exists(scaler_path):
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        scaler_json = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'feature_count': len(scaler.mean_)
        }
        
        scaler_json_path = os.path.join(output_dir, 'scaler.json')
        with open(scaler_json_path, 'w') as f:
            json.dump(scaler_json, f, indent=2)
        
        print(f"✅ Scaler exported to: {scaler_json_path}")
        print(f"📏 Features: {scaler_json['feature_count']}")
    
    # Load and export preprocessing metadata (feature selection)
    print("\n🔧 Exporting preprocessing metadata...")
    preprocessing_path = 'models/children_asd_optimized/preprocessing.json'
    if os.path.exists(preprocessing_path):
        with open(preprocessing_path, 'r') as f:
            preprocessing = json.load(f)
        
        preprocessing_out_path = os.path.join(output_dir, 'preprocessing.json')
        with open(preprocessing_out_path, 'w') as f:
            json.dump(preprocessing, f, indent=2)
        
        print(f"✅ Preprocessing metadata exported to: {preprocessing_out_path}")
        print(f"📊 Selected features: {len(preprocessing.get('selected_features', []))}")
        print(f"   Feature indices: {preprocessing.get('selected_feature_indices', [])}")
    
    return True

if __name__ == "__main__":
    success = export_model_manually()
    if success:
        print("\n" + "="*60)
        print("✅ EXPORT COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Update baseline_model_web.js to load from:")
        print("   models/ACTIVE/children_asd_v2_tfjs/model.json")
        print("2. Test in browser with sample CSV")
        print("3. Deploy to GitHub Pages")
    else:
        print("\n❌ Export failed - check errors above")
