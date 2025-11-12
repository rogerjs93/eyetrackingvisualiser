"""
Convert both Children and Adult baseline models to TensorFlow.js format
This creates browser-compatible versions for GitHub Pages deployment
"""
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
from pathlib import Path
import shutil

def convert_model_to_tfjs(keras_model_path, output_dir, model_name):
    """Convert a Keras model to TensorFlow.js format"""
    print(f"\n{'='*70}")
    print(f"Converting {model_name} to TensorFlow.js")
    print(f"{'='*70}")
    
    # Load the Keras model
    print(f"📂 Loading: {keras_model_path}")
    model = keras.models.load_model(keras_model_path)
    print(f"✅ Model loaded: {model.count_params():,} parameters")
    
    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"🗑️  Removing existing directory: {output_path}")
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    
    # Convert to TensorFlow.js format
    print("🔄 Converting to TensorFlow.js...")
    tfjs.converters.save_keras_model(model, str(output_path))
    print(f"✅ Conversion complete!")
    
    # Verify files were created
    model_json = output_path / 'model.json'
    if model_json.exists():
        print(f"✓ model.json created")
        # Count weight files
        weight_files = list(output_path.glob('*.bin'))
        print(f"✓ {len(weight_files)} weight file(s) created")
    else:
        print(f"❌ Error: model.json not found!")
    
    return output_path

def main():
    print("\n" + "="*70)
    print("CONVERTING ASD BASELINE MODELS TO TENSORFLOW.JS")
    print("="*70)
    print("\n🎯 Purpose: Enable browser-based AI analysis on GitHub Pages")
    print("📦 Output: TensorFlow.js compatible model files\n")
    
    # Convert Children model
    children_keras = Path('models/baseline_children_asd/optimized_autoencoder.keras')
    children_tfjs = 'models/baseline_children_asd_tfjs'
    
    if children_keras.exists():
        convert_model_to_tfjs(
            children_keras,
            children_tfjs,
            "Children ASD Baseline (Ages 3-12)"
        )
        
        # Copy scaler.json to tfjs directory
        scaler_src = Path('models/baseline_children_asd/scaler.json')
        scaler_dst = Path(children_tfjs) / 'scaler.json'
        if scaler_src.exists():
            shutil.copy(scaler_src, scaler_dst)
            print(f"✓ Copied scaler.json")
    else:
        print(f"❌ Children model not found: {children_keras}")
    
    # Convert Adult model
    adult_keras = Path('models/baseline_adult_asd/adult_asd_baseline.keras')
    adult_tfjs = 'models/baseline_adult_asd_tfjs'
    
    if adult_keras.exists():
        convert_model_to_tfjs(
            adult_keras,
            adult_tfjs,
            "Adult ASD Baseline"
        )
        
        # Copy scaler.json to tfjs directory
        scaler_src = Path('models/baseline_adult_asd/scaler.json')
        scaler_dst = Path(adult_tfjs) / 'scaler.json'
        if scaler_src.exists():
            shutil.copy(scaler_src, scaler_dst)
            print(f"✓ Copied scaler.json")
    else:
        print(f"❌ Adult model not found: {adult_keras}")
    
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE!")
    print("="*70)
    print("\n📋 Summary:")
    print(f"   Children TF.js: {children_tfjs}/")
    print(f"   Adult TF.js:    {adult_tfjs}/")
    print("\n💡 Next steps:")
    print("   1. Update index.html to use *_tfjs directories")
    print("   2. Commit and push to GitHub")
    print("   3. Test on GitHub Pages")
    print("\n🌐 Models will be accessible at:")
    print(f"   https://rogerjs93.github.io/eyetrackingvisualiser/{children_tfjs}/model.json")
    print(f"   https://rogerjs93.github.io/eyetrackingvisualiser/{adult_tfjs}/model.json")

if __name__ == "__main__":
    main()
