# Eye-Tracking Analysis Methodology

## Mathematical Formulations and Interpretations

This document provides detailed explanations of all mathematical formulations, algorithms, and interpretation guidelines used in the eye-tracking analysis system.

---

## Table of Contents

1. [Pattern Recognition Metrics](#pattern-recognition-metrics)
2. [Cognitive Load Metrics](#cognitive-load-metrics)
3. [Visualization Methods](#visualization-methods)
4. [References](#references)

---

## Pattern Recognition Metrics

### 1. Reading Behavior Detection

**Purpose:** Automatically detect whether the user is reading, scanning, or exploring content based on gaze movement patterns.

**Mathematical Formulation:**

```
Reading Score = (LTR_ratio × 0.4) + (return_sweeps/n × 0.3) + (regularity × 0.3)

Where:
- LTR_ratio = count(Δx > 0) / n  [proportion of rightward movements]
- return_sweeps = count(Δx < -100)  [large leftward jumps indicating line returns]
- regularity = 1 / (var(angle_changes) + 0.01)  [consistency of movement angles]
- n = total number of fixations

Scanning Score = (entropy/4 × 0.5) + (short_fixations × 0.3) + ((1-regularity) × 0.2)

Exploration Score = (entropy/4 × 0.6) + ((1-regularity) × 0.4)
```

**Weight Justification:**
- Reading: LTR_ratio (40%) is most indicative, return_sweeps (30%) confirms line-by-line reading
- Scanning: Entropy (50%) best captures distributed attention
- Exploration: Entropy (60%) dominates as it measures unpredictability

**Interpretation:**
- **Reading** (score > 0.7): High left-to-right progression, regular return sweeps (>100px jumps), consistent line-by-line movement
- **Scanning** (score > 0.6): Quick fixations (<200ms), high spatial dispersion, less regular patterns
- **Exploring** (score > 0.5): High entropy (>2.5), irregular movements, wide spatial coverage

**References:** Rayner (1998), Holmqvist et al. (2011)

---

### 2. Expertise Level Classification

**Purpose:** Classify user expertise (novice/intermediate/expert) based on gaze efficiency metrics.

**Mathematical Formulation:**

```
Path Efficiency = straight_line_distance / actual_path_length

Where:
- straight_line_distance = √[(x_end - x_start)² + (y_end - y_start)²]
- actual_path_length = Σᵢ √[(xᵢ - xᵢ₋₁)² + (yᵢ - yᵢ₋₁)²]

Duration Efficiency = 1 / (mean_fixation_duration / 200ms)

Movement Efficiency = 1 / (direction_changes/n + 1)

Expertise Score = (path_eff × 0.3) + (duration_eff × 0.25) + 
                 (movement_eff × 0.25) + ((1 - entropy/4) × 0.2)
```

**Threshold Justification:**
- Path efficiency > 0.7: Direct, efficient paths typical of experts
- Duration < 200ms: Quick information extraction
- Low entropy: Systematic, structured viewing

**Classification:**
- **Expert** (score > 0.7): Efficient paths, shorter fixations (<250ms), minimal backtracking
- **Intermediate** (0.4 < score ≤ 0.7): Moderate efficiency, some exploration
- **Novice** (score ≤ 0.4): Inefficient paths, longer fixations (>400ms), frequent backtracking

**References:** Gegenfurtner et al. (2011), Jarodzka et al. (2010)

---

### 3. Areas of Interest (AOI) Detection

**Purpose:** Automatically identify regions of high attention using density-based clustering.

**Algorithm:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Parameters:**
```
eps = 100 pixels  [maximum distance between points in same cluster]
min_samples = 5   [minimum points to form a dense region]
```

**For each cluster:**
```
Center = (mean(x), mean(y))
Radius = max(√[(xᵢ - center_x)² + (yᵢ - center_y)²])
Dwell Time = Σ duration_i for all points in cluster
Importance = dwell_time × n_fixations
```

**Parameter Justification:**
- eps=100px: Based on typical foveal vision diameter (2-3°, ~100px at 60cm distance)
- min_samples=5: Minimum meaningful cluster size

**Interpretation:**
- **High Importance** (>5000ms×points): Primary areas of interest
- **Coverage**: (n_points_in_clusters / total_points) indicates focus distribution
- **Noise Points** (label = -1): Scattered, transitional fixations

**References:** Ester et al. (1996), Holmqvist et al. (2011)

---

### 4. Confusion Indicator Detection

**Purpose:** Identify signs of cognitive difficulty through revisits and erratic movements.

**Mathematical Formulation:**

```
Revisit Rate = n_revisits / total_fixations

Where revisit is defined as:
  distance(point_i, point_j) < 50px, where j > i + 10

Movement Erraticism = std(angle_changes)

Where:
  angle_changes = diff(arctan2(Δy, Δx))

Long Fixation Rate = count(duration > 500ms) / n

Confusion Score = (revisit_rate × 0.3) + (min(erraticism/2, 1) × 0.3) +
                 (long_fix_rate × 0.2) + (entropy/4 × 0.2)
```

**Threshold Justification:**
- Revisit distance < 50px: Within attention focus area
- Gap > 10 fixations: Excludes local saccades, identifies true returns
- Long fixation > 500ms: Indicates processing difficulty (normal: 200-300ms)

**Interpretation:**
- **High Confusion** (>0.7): Many revisits (>30%), erratic movements (std>1.5), long fixations (>30%)
- **Moderate** (0.4-0.7): Some difficulty indicators present
- **Low** (<0.4): Smooth, efficient navigation with minimal backtracking

**References:** Gwizdka (2014), cognitive load indicators

---

## Cognitive Load Metrics

### 5. Spatial Entropy

**Purpose:** Measure unpredictability and dispersion of gaze distribution across the screen.

**Mathematical Formulation:**

Shannon Entropy:
```
H = -Σᵢ p(i) × log₂(p(i))

Where:
- Screen divided into n×n grid (default 10×10)
- p(i) = count(fixations in bin i) / total_fixations
- Sum over all bins with p(i) > 0
```

**Grid Size Justification:**
- 10×10 (100 bins): Balance between resolution and statistical reliability
- Smaller bins: More detail but sparse data
- Larger bins: Less sensitive to dispersion

**Interpretation:**
```
Maximum Entropy = log₂(n²) = log₂(100) = 6.64

- High (>2.5): Dispersed attention, exploratory viewing, scanning
- Medium (1.5-2.5): Balanced distribution, typical reading
- Low (<1.5): Focused attention on specific regions
```

**References:** Shannon (1948), adapted for eye-tracking

---

### 6. Fixation Rate Analysis

**Purpose:** Measure visual processing speed through fixations per second.

**Mathematical Formulation:**

```
Fixation Rate = n_fixations / total_time (seconds)

Mean Fixation Duration = Σ durations / n
```

**Interpretation:**

**Processing Indicator:**
- **High Rate** (>3 fps): High cognitive load OR fast scanning/searching
- **Normal** (2-3 fps): Typical reading/processing
- **Low Rate** (<2 fps): Deep processing/careful examination

**Combined with Duration:**
- High rate + short durations (<200ms) = Scanning/searching behavior
- Low rate + long durations (>400ms) = Careful reading/analysis
- High rate + long durations = Difficulty/confusion

**Typical Values:**
- Reading: 3-4 fps, 200-250ms duration
- Scanning: 4-5 fps, 150-200ms duration
- Careful examination: 1-2 fps, 400-600ms duration

**References:** Just & Carpenter (1980), Rayner (1998)

---

### 7. Saccade Metrics

**Purpose:** Analyze eye movement characteristics between fixations.

**Mathematical Formulation:**

```
Saccade Length = √[(x₂-x₁)² + (y₂-y₁)²]

Mean Saccade Length = (Σ lengths) / (n-1)

Saccade Rate = n_saccades / total_time

Saccade Velocity = length / time_difference
```

**Interpretation:**

**Saccade Length:**
- **Long** (>150px): Searching behavior, overview scanning
- **Medium** (50-150px): Normal navigation, reading
- **Short** (<50px): Detailed inspection, careful reading

**Saccade Rate:**
- **High** (>4/s): Active visual search, information gathering
- **Normal** (2-4/s): Typical reading
- **Low** (<2/s): Prolonged fixations, deep processing

**Velocity Patterns:**
- Fast saccades (>500px/s): Rapid reorientation
- Slow saccades (<300px/s): Careful movement, uncertainty

**References:** Holmqvist et al. (2011)

---

### 8. Ambient vs Focal Attention

**Purpose:** Classify attention mode based on fixation characteristics.

**Mathematical Formulation:**

```
Classification Threshold = 250ms

Ambient Fixations: duration < 250ms
- Rapid information gathering
- Peripheral processing
- "What is where" information

Focal Fixations: duration ≥ 250ms
- Central foveal processing
- Detailed examination
- "What is it" information

Ambient Ratio = n_ambient / total
Focal Ratio = n_focal / total
```

**Threshold Justification:**
- 250ms: Empirically determined boundary between rapid sampling and detailed processing
- Based on Velichkovsky et al. (2002) two-level processing theory

**Interpretation:**

**Dominant Mode:**
- **Ambient** (>50% short fixations): Overview, navigation, spatial orientation
- **Focal** (>50% long fixations): Deep processing, analysis, content extraction
- **Balanced** (45-55%): Task-dependent switching

**References:** Velichkovsky et al. (2002)

---

### 9. Gaze Transition Entropy

**Purpose:** Measure predictability of gaze movement patterns.

**Mathematical Formulation:**

```
1. Divide screen into 5×5 grid (25 regions)
2. Create transition matrix T[i,j] = count(region_i → region_j)
3. Normalize: P[i,j] = T[i,j] / Σⱼ T[i,j]
4. Calculate entropy: H = -ΣᵢΣⱼ P[i,j] × log₂(P[i,j])
```

**Interpretation:**

**Entropy Levels:**
- **High** (>4.0): Unpredictable, random movements
- **Medium** (2.0-4.0): Semi-structured patterns
- **Low** (<2.0): Predictable, repetitive patterns

**Low Entropy Suggests:**
- Systematic scanning strategy
- Task-driven behavior
- Expertise in domain

**References:** Ellis & Stark (1986)

---

### 10. Task Difficulty Score

**Purpose:** Composite measure of cognitive load from multiple indicators.

**Mathematical Formulation:**

```
Difficulty Score (0-10) = weighted_sum of:

1. Spatial Entropy (30%): (entropy / 4.0) × 3
2. Fixation Rate (20%): (rate / 10) × 2
3. Mean Duration (20%): (duration / 500) × 2
4. Saccade Length (20%): (length / 200) × 2
5. Ambient Ratio (10%): ambient_ratio × 1

Final Score = min(sum, 10)
```

**Weight Justification:**
- Entropy (30%): Most comprehensive indicator of attention distribution
- Fixation & Duration (40% combined): Direct processing indicators
- Saccade Length (20%): Navigation efficiency
- Ambient Ratio (10%): Attention mode context

**Interpretation:**

**Difficulty Levels:**
- **Low** (<4): Easy task, efficient processing, clear interface
- **Moderate** (4-7): Normal cognitive load, appropriate complexity
- **High** (>7): Difficult task, high cognitive demand, potential usability issues

**Actionable Recommendations:**
- **High**: Simplify interface, add guidance, reduce complexity
- **Moderate**: Monitor user success, consider optimizations
- **Low**: Appropriate difficulty level, maintain design

**References:** Based on cognitive load theory (Sweller, 1988)

---

## Visualization Methods

### 11. Gaze Heatmap

**Purpose:** 2D density visualization of fixation distribution.

**Algorithm:**

```
1. Divide screen into n×n bins (typically 40×40 or 50×50)
2. Count fixations in each bin: H[i,j] = count((x,y) in bin[i,j])
3. Apply Gaussian smoothing: G[i,j] = Σₖₗ H[k,l] × gaussian(i-k, j-l, σ=2)
4. Normalize to [0,1]: H_norm = (H - min(H)) / (max(H) - min(H))
5. Map to color scale (Hot/Viridis): red=high, blue=low
```

**Interpretation:**
- **Hot colors** (red/yellow) = High attention density
- **Cool colors** (blue/black) = Low/no attention
- **Reveals**: Primary focus areas, visual attention patterns, interface effectiveness

**References:** Wooding (2002)

---

### 12. Scan Path Visualization

**Purpose:** Sequential trajectory of gaze movements over time.

**Rendering:**

```
Plot: (x₁,y₁) → (x₂,y₂) → ... → (xₙ,yₙ)

With:
- Line thickness ∝ fixation duration
- Color gradient: temporal progression (time-based colormap)
- Circle size ∝ duration at each point
- Start marker: Green star
- End marker: Red star
```

**Pattern Recognition:**
- **Reading**: Left-right progression with downward return sweeps
- **Scanning**: Distributed points with long jumps (>200px)
- **Focused**: Clustered points in small region (<300px diameter)
- **Random**: No clear structure, high entropy

**References:** Noton & Stark (1971)

---

### 13. Sankey Diagram (Gaze Flow)

**Purpose:** Show flow of attention between screen regions.

**Algorithm:**

```
1. Divide screen into grid (e.g., 4×4 or 6×6)
2. Assign each fixation to region based on (x,y) coordinates
3. Count transitions: T[i→j] for all consecutive pairs
4. Create Sankey flow where width ∝ transition count
5. Filter: Only show flows with count ≥ threshold (e.g., 3)
```

**Interpretation:**
- **Thick flows**: Frequent transitions, common navigation paths
- **Thin flows**: Rare transitions, unusual paths
- **Reveals**: Information architecture, region relationships, navigation patterns

**References:** Blascheck et al. (2014)

---

### 14. Velocity Heatmap

**Purpose:** Show eye movement speed across screen regions.

**Mathematical Formulation:**

```
Velocity[i] = √[(x[i+1]-x[i])² + (y[i+1]-y[i])²] / (t[i+1]-t[i])

Average velocity per region:
V_region = mean(velocities for points in region)

Units: pixels per millisecond
```

**Interpretation:**

**High Velocity (red):**
- Fast saccades
- Scanning behavior
- Low-interest regions

**Low Velocity (blue):**
- Careful examination
- High-interest areas
- Reading zones

**References:** Salvucci & Goldberg (2000)

---

## References

1. **Rayner, K. (1998).** Eye movements in reading and information processing: 20 years of research. *Psychological Bulletin*, 124(3), 372-422.

2. **Holmqvist, K., Nyström, M., Andersson, R., Dewhurst, R., Jarodzka, H., & Van de Weijer, J. (2011).** *Eye tracking: A comprehensive guide to methods and measures.* Oxford University Press.

3. **Gegenfurtner, A., Lehtinen, E., & Säljö, R. (2011).** Expertise differences in the comprehension of visualizations: A meta-analysis of eye-tracking research in professional domains. *Educational Psychology Review*, 23(4), 523-552.

4. **Jarodzka, H., Scheiter, K., Gerjets, P., & Van Gog, T. (2010).** In the eyes of the beholder: How experts and novices interpret dynamic stimuli. *Learning and Instruction*, 20(2), 146-154.

5. **Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996).** A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*, 96(34), 226-231.

6. **Gwizdka, J. (2014).** Characterizing relevance with eye-tracking measures. *Proceedings of the 5th Information Interaction in Context Symposium*, 58-67.

7. **Shannon, C. E. (1948).** A mathematical theory of communication. *The Bell System Technical Journal*, 27(3), 379-423.

8. **Just, M. A., & Carpenter, P. A. (1980).** A theory of reading: From eye fixations to comprehension. *Psychological Review*, 87(4), 329-354.

9. **Velichkovsky, B. M., Rothert, A., Kopf, M., Dornhöfer, S. M., & Joos, M. (2002).** Towards an express-diagnostics for level of processing and hazard perception. *Transportation Research Part F: Traffic Psychology and Behaviour*, 5(2), 145-156.

10. **Ellis, S. R., & Stark, L. (1986).** Statistical dependency in visual scanning. *Human Factors*, 28(4), 421-438.

11. **Sweller, J. (1988).** Cognitive load during problem solving: Effects on learning. *Cognitive Science*, 12(2), 257-285.

12. **Wooding, D. S. (2002).** Fixation maps: Quantifying eye-movement traces. *Proceedings of the 2002 Symposium on Eye Tracking Research & Applications*, 31-36.

13. **Noton, D., & Stark, L. (1971).** Scanpaths in eye movements during pattern perception. *Science*, 171(3968), 308-311.

14. **Blascheck, T., Kurzhals, K., Raschke, M., Burch, M., Weiskopf, D., & Ertl, T. (2014).** State-of-the-art of visualization for eye tracking data. *EuroVis (STARs)*, 1-20.

15. **Salvucci, D. D., & Goldberg, J. H. (2000).** Identifying fixations and saccades in eye-tracking protocols. *Proceedings of the 2000 Symposium on Eye Tracking Research & Applications*, 71-78.

---

## Additional Notes

### Calibration and Data Quality

All metrics assume properly calibrated eye-tracking data with:
- Accuracy: <1° visual angle (~30px at 60cm viewing distance)
- Sampling rate: ≥60Hz (preferably ≥120Hz)
- Valid fixation detection (velocity threshold, duration minimum)

### Contextual Factors

Interpretation should consider:
- Task type (reading, searching, decision-making)
- Content type (text, images, mixed media)
- User expertise level
- Display characteristics (size, resolution, viewing distance)

### Customization

Many thresholds and weights can be adjusted based on:
- Specific application domain
- Population characteristics
- Task requirements
- Empirical validation with your data

---

*Last Updated: 2024*
