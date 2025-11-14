"""
Download and prepare Adult ASD Dataset (Ramot et al., 2019)
Ages 15-30, for comparison with children baseline
"""
import os
from pathlib import Path
import requests
from tqdm import tqdm

print("="*70)
print("Adult ASD Dataset Downloader")
print("="*70)
print("\n📚 Dataset Information:")
print("   Title: Eye tracking data for participants with Autism Spectrum Disorders")
print("   Authors: Ramot, M., Walsh, C., Martin, A. (2019)")
print("   Source: Figshare/Nature Human Behaviour")
print("   Link: https://nih.figshare.com/articles/dataset/10324877")
print("   Age Range: 15-30 years")
print("   Participants: 36 males with ASD + controls")
print("   Sampling Rate: 1000 Hz (high precision)")

# Create directory
adult_data_dir = Path("data/autism/adult_asd")
adult_data_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Download directory: {adult_data_dir}")

# Dataset file info from Figshare
# Note: You need to manually download from Figshare due to authentication
# This script provides instructions

print("\n" + "="*70)
print("MANUAL DOWNLOAD REQUIRED")
print("="*70)

print("\n⚠️  This dataset requires manual download from Figshare:")
print("\n📋 Instructions:")
print("   1. Visit: https://nih.figshare.com/articles/dataset/10324877")
print("   2. Look for file: 'Eye_tracking_ASD.mat' or similar")
print("   3. Click 'Download' button")
print("   4. Save to:", adult_data_dir.absolute())
print("   5. Run this script again to process the data")

# Check if file already exists
possible_files = [
    "Eye_tracking_ASD.mat",
    "eye_tracking_asd.mat", 
    "eyetracking_asd.mat",
    "asd_eyetracking.mat",
    "ramot_asd_data.mat"
]

found_files = []
for filename in possible_files:
    filepath = adult_data_dir / filename
    if filepath.exists():
        found_files.append(filepath)

if found_files:
    print("\n✅ Found existing files:")
    for f in found_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   • {f.name} ({size_mb:.1f} MB)")
    print("\n➡️  Run 'python process_adult_dataset.py' to continue")
else:
    print("\n❌ No dataset files found yet.")
    print("   Please download manually as instructed above.")

print("\n" + "="*70)
print("Alternative: Automated Download Helper")
print("="*70)
print("\nIf the dataset has a direct download link, you can:")
print("1. Find the direct file URL on Figshare")
print("2. Update this script with the URL")
print("3. Uncomment the download code below")

print("\n💡 Tip: Check Figshare API documentation for programmatic access")
print("   https://docs.figshare.com/")

# Placeholder for automated download (when you get the direct URL)
"""
# Example code (update with actual URL):
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/FILEID"
output_file = adult_data_dir / "Eye_tracking_ASD.mat"

if not output_file.exists():
    print(f"\n📥 Downloading from Figshare...")
    response = requests.get(DOWNLOAD_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_file, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=output_file.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"✅ Downloaded: {output_file}")
else:
    print(f"✅ File already exists: {output_file}")
"""

print("\n" + "="*70)
