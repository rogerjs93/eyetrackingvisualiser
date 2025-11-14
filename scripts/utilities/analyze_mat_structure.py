"""
Deep analysis of RawEyetrackingASD.mat structure
"""
import scipy.io
import numpy as np

print("="*70)
print("Deep Analysis of RawEyetrackingASD.mat")
print("="*70)

mat_data = scipy.io.loadmat(r"data\autism\autismdata2\RawEyetrackingASD.mat")
eye_data = mat_data['eyeMovementsASD']

print(f"\n📊 Data Structure:")
print(f"   Shape: {eye_data.shape}")
print(f"   Dimensions: (36, 24, 2, 14000)")
print()
print("   Interpretation:")
print("   • Dimension 1: 36 = Likely 36 TRIALS or STIMULI")
print("   • Dimension 2: 24 = Likely 24 PARTICIPANTS (or time windows)")
print("   • Dimension 3: 2 = Likely X and Y coordinates")
print("   • Dimension 4: 14000 = Time samples per trial")

print("\n🔍 Sample Data Analysis:")
print("-" * 70)

# Check if data is empty or has values
non_zero = np.count_nonzero(eye_data)
total_elements = eye_data.size
print(f"   Non-zero values: {non_zero:,} / {total_elements:,} ({non_zero/total_elements*100:.1f}%)")

# Check first trial, first participant
first_trial = eye_data[0, 0, :, :]  # Shape: (2, 14000) = X and Y over time
print(f"\n   First trial, first participant shape: {first_trial.shape}")
print(f"   X coordinate range: [{first_trial[0].min():.2f}, {first_trial[0].max():.2f}]")
print(f"   Y coordinate range: [{first_trial[1].min():.2f}, {first_trial[1].max():.2f}]")

# Check if coordinates look like screen positions
x_values = first_trial[0]
y_values = first_trial[1]
x_non_zero = x_values[x_values != 0]
y_non_zero = y_values[y_values != 0]

if len(x_non_zero) > 0:
    print(f"\n   Non-zero X values: min={x_non_zero.min():.2f}, max={x_non_zero.max():.2f}")
    print(f"   Non-zero Y values: min={y_non_zero.min():.2f}, max={y_non_zero.max():.2f}")

# Check how many participants have data
participants_with_data = []
for p in range(eye_data.shape[1]):
    participant_data = eye_data[0, p, :, :]
    if np.count_nonzero(participant_data) > 0:
        participants_with_data.append(p + 1)

print(f"\n   Participants with data in first trial: {len(participants_with_data)}/24")
if len(participants_with_data) <= 25:
    print(f"   Participant IDs: {participants_with_data}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)

print("\n✅ This is RAW EYE-TRACKING DATA, not a baseline model")
print("\n   Structure breakdown:")
print("   • 36 trials/stimuli shown to participants")
print("   • 24 participants (though dataset has 25 - might be 24 ASD)")
print("   • X, Y coordinates tracked")
print("   • 14,000 time samples per trial (high temporal resolution)")

print("\n   This is INDIVIDUAL RAW DATA, containing:")
print("   ✓ Raw gaze coordinates over time")
print("   ✓ Multiple trials per participant")
print("   ✓ NOT aggregated baseline statistics")
print("   ✓ NOT a pre-trained model")

print("\n💡 Comparison to your current data:")
print("   • Your CSV files: Individual participant data (similar)")
print("   • This .mat file: All participants combined in one file")
print("   • Your baseline model: Trained on extracted features from CSVs")
print("   • This .mat could be: Alternative source for the same data")

print("\n   You could potentially:")
print("   1. Extract features from this .mat file")
print("   2. Compare if it matches your CSV data")
print("   3. Use it as additional training data (if different)")

print("\n" + "="*70)
