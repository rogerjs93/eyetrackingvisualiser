# Autism Eye-Tracking Dataset

## ⬇️ Download Instructions

This folder should contain the autism eye-tracking dataset, which is **not included in the repository** due to GitHub file size limits.

### How to Get the Dataset

1. **Download from Kaggle:**
   - Visit: [Eye Tracking Autism Dataset](https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism)
   - Download the dataset (requires Kaggle account)

2. **Extract Files:**
   Place the following files in this directory:
   ```
   data/autism/
   ├── Metadata_Participants.csv
   └── Eye-tracking Output/
       ├── 1.csv
       ├── 2.csv
       ├── ...
       └── 25.csv
   ```

3. **Run the Dashboard:**
   Once the files are in place, the interactive dashboard will automatically detect the 25 participants.

## Dataset Information

- **Participants:** 25 children with autism spectrum disorder (ASD)
- **Age Range:** 2.7 - 12.3 years
- **CARS Scores:** 27.0 - 36.5
- **Data Format:** Eye-tracking coordinates (x, y), timestamps, duration, event types

## Attribution

**Eye Tracking Autism Dataset**  
Published by: IMT Kaggle Team  
License: Check Kaggle dataset page for specific license terms  
Citation: Please cite the original dataset when using this data in research

## File Structure

### Metadata_Participants.csv
Contains participant demographic information:
- ParticipantID
- Age
- Gender
- CARS Score (Childhood Autism Rating Scale)

### Eye-tracking Output/*.csv
Individual CSV files (1.csv through 25.csv) containing eye-tracking data:
- x, y coordinates
- timestamps
- duration
- event_type (fixation, saccade, etc.)

## Support

If you encounter issues loading the dataset, ensure:
1. Files are in the correct directory structure
2. CSV files are not corrupted
3. File names match exactly (1.csv, 2.csv, etc.)
