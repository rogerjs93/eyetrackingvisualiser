# Adult ASD Dataset - Download Instructions

## ⚠️ IMPORTANT: This is a DIFFERENT dataset from your children data

**Your current children dataset:**
- Location: `data/autism/Eye-tracking Output/` (CSV files)
- Also: `data/autism/autismdata2/RawEyetrackingASD.mat` (same data, MATLAB format)
- Source: Cilia et al. (2023)
- Age range: 3-12 years
- Status: ✅ Already trained and deployed

**Adult dataset you need to download:**
- Source: Ramot et al. (2019), Nature Human Behaviour
- Age range: 15-30 years  
- Participants: 36 males with ASD + controls
- Sampling rate: 1000 Hz (vs 60 Hz for children)

---

## 📥 Download Instructions

### Step 1: Visit Figshare
Go to: https://nih.figshare.com/articles/dataset/10324877

### Step 2: Look for the data file
The dataset should be named something like:
- `Eye_tracking_ASD.mat`
- `ASD_eye_tracking_data.mat`
- Or similar MATLAB format

### Step 3: Download
1. Click the download button
2. Save to: `C:\Users\roger\Desktop\Roger\Projects\Software engineering\python\Pythondata visualizer\data\autism\adult_asd\`
3. Note the exact filename

### Step 4: Verify download
The file should be:
- MATLAB format (.mat)
- Several MB in size
- Contains eye-tracking data for adults ages 15-30

### Step 5: Process the data
Once downloaded, run:
```bash
python process_adult_dataset.py
```

This will:
- Load the adult data
- Extract the same 28 features as children model
- Train a separate adult baseline model
- Compare children vs adult patterns

---

## 🔍 What to expect

**Adult dataset differences:**
- Older participants (15-30 vs 3-12)
- Higher sampling rate (1000 Hz vs 60 Hz)
- Different stimuli (movie clips vs simple images)
- More complex gaze patterns
- Different neural mechanisms

**Why separate models matter:**
- Eye movements change with development
- Adults have different attention strategies
- Neural maturation affects gaze behavior
- Different clinical applications

---

## 📞 If you have trouble downloading

The Figshare link requires:
1. No authentication (should be public)
2. Direct download button on the page
3. May need to accept terms/conditions

If the file structure is different than expected, we can adapt the processing script.

---

**Once you've downloaded the adult dataset, let me know and I'll process it immediately!**
