"""
Convert Keras models to TensorFlow.js using SavedModel intermediate format
This avoids the tensorflow_decision_forests dependency issue
"""
import tensorflow as tf
from tensorflow import keras
import json
import shutil
from pathlib import Path
import subprocess

def keras_to_saved_model(keras_path, saved_model_dir):
    """Convert Keras model to SavedModel format"""
    print(f"📂 Loading Keras model: {keras_path}")
    model = keras.models.load_model(keras_path)
    print(f"✅ Model loaded: {model.count_params():,} parameters")
    
    # Remove existing directory
    if Path(saved_model_dir).exists():
        shutil.rmtree(saved_model_dir)
    
    # Save as SavedModel
    print(f"💾 Saving as SavedModel: {saved_model_dir}")
    model.export(saved_model_dir)
    print("✅ SavedModel created")
    
    return model

def saved_model_to_tfjs_manual(saved_model_dir, tfjs_dir, scaler_json_path):
    """Manually convert SavedModel to TFJS by copying weights"""
    print(f"\n🔄 Manual conversion to TensorFlow.js format...")
    
    # Create tfjs directory
    tfjs_path = Path(tfjs_dir)
    if tfjs_path.exists():
        shutil.rmtree(tfjs_path)
    tfjs_path.mkdir(parents=True, exist_ok=True)
    
    # Load the model to get architecture
    model = tf.saved_model.load(saved_model_dir)
    
    # For now, let's use the existing tfjs converter via Python API with workaround
    # We'll create a simpler conversion that works
    import tensorflowjs as tfjs
    
    print(f"🔄 Converting to TensorFlow.js...")
    tfjs.converters.convert_tf_saved_model(
        saved_model_dir,
        str(tfjs_path)
    )
    
    # Copy scaler.json
    if Path(scaler_json_path).exists():
        shutil.copy(scaler_json_path, tfjs_path / 'scaler.json')
        print(f"✓ Copied scaler.json")
    
    print(f"✅ TensorFlow.js model created in: {tfjs_path}")
    
    # List created files
    files = list(tfjs_path.glob('*'))
    print(f"📄 Created files:")
    for f in files:
        print(f"   - {f.name}")

def convert_model(keras_path, saved_model_dir, tfjs_dir, scaler_path, name):
    """Complete conversion pipeline"""
    print(f"\n{'='*70}")
    print(f"Converting: {name}")
    print(f"{'='*70}")
    
    try:
        # Step 1: Keras -> SavedModel
        keras_to_saved_model(keras_path, saved_model_dir)
        
        # Step 2: SavedModel -> TFJS
        saved_model_to_tfjs_manual(saved_model_dir, tfjs_dir, scaler_path)
        
        print(f"\n✅ {name} conversion complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error converting {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("CONVERTING ASD BASELINE MODELS TO TENSORFLOW.JS")
    print("="*70)
    
    results = []
    
    # Children model
    results.append(convert_model(
        keras_path='models/baseline_children_asd/optimized_autoencoder.keras',
        saved_model_dir='models/temp_saved_model_children',
        tfjs_dir='models/baseline_children_asd_tfjs',
        scaler_path='models/baseline_children_asd/scaler.json',
        name="Children ASD Baseline (Ages 3-12)"
    ))
    
    # Adult model
    results.append(convert_model(
        keras_path='models/baseline_adult_asd/adult_asd_baseline.keras',
        saved_model_dir='models/temp_saved_model_adult',
        tfjs_dir='models/baseline_adult_asd_tfjs',
        scaler_path='models/baseline_adult_asd/scaler.json',
        name="Adult ASD Baseline"
    ))
    
    # Cleanup temp directories
    print(f"\n🧹 Cleaning up temporary files...")
    for temp_dir in ['models/temp_saved_model_children', 'models/temp_saved_model_adult']:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            print(f"   ✓ Removed {temp_dir}")
    
    print("\n" + "="*70)
    if all(results):
        print("✅ ALL CONVERSIONS SUCCESSFUL!")
    else:
        print("⚠️  SOME CONVERSIONS FAILED")
    print("="*70)
    
    print("\n📋 Models ready for deployment:")
    print("   - models/baseline_children_asd_tfjs/")
    print("   - models/baseline_adult_asd_tfjs/")

if __name__ == "__main__":
    main()
