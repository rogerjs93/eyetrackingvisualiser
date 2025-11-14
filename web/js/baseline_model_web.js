/**
 * Baseline Model Web Interface
 * Loads TensorFlow.js model and performs autism baseline comparison in browser
 */

class BaselineModelWeb {
    constructor() {
        this.model = null;
        this.scaler = null;
        this.preprocessing = null; // Feature selection metadata
        this.baselineStats = null;
        this.isModelLoaded = false;
        
        // Default paths (can be overridden for different age groups)
        this.modelPath = 'models/production/children_asd_optimized/model.json';
        this.scalerPath = 'models/production/children_asd_optimized/scaler.json';
        this.preprocessingPath = 'models/production/children_asd_optimized/preprocessing.json';
        this.threshold = 0.4231; // Default to optimized children threshold
    }

    /**
     * Configure model for children ASD (optimized lightweight model)
     */
    useOptimizedChildrenModel() {
        this.modelPath = 'models/production/children_asd_optimized/model.json';
        this.scalerPath = 'models/production/children_asd_optimized/scaler.json';
        this.preprocessingPath = 'models/production/children_asd_optimized/preprocessing.json';
        this.threshold = 0.4231; // Optimized model threshold
        console.log('📌 Configured for: Optimized Children ASD Model (20 features)');
    }

    /**
     * Configure model for legacy children ASD (43 features)
     */
    useLegacyChildrenModel() {
        this.modelPath = 'models/archive/baseline_children_asd_tfjs/model.json';
        this.scalerPath = 'models/archive/baseline_children_asd_tfjs/scaler.json';
        this.preprocessingPath = null;
        this.threshold = 0.4069; // Legacy baseline
        console.log('📌 Configured for: Legacy Children ASD Model (28 features)');
    }


    /**
     * Load the TensorFlow.js model and baseline statistics
     */
    async loadModel() {
        try {
            console.log('🔄 Loading TensorFlow.js model...');
            
            // Determine base URL (works for both local and GitHub Pages)
            const baseUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
                ? '' 
                : 'https://rogerjs93.github.io/eyetrackingvisualiser/';
            
            console.log(`📍 Base URL: ${baseUrl || 'relative path'}`);
            
            // Load the model with configurable path
            const fullModelPath = `${baseUrl}${this.modelPath}`;
            console.log(`📂 Loading model from: ${fullModelPath}`);
            this.model = await tf.loadLayersModel(fullModelPath);
            console.log('✅ Model loaded successfully');
            
            // Load scaler parameters (mean and scale from StandardScaler)
            const fullScalerPath = `${baseUrl}${this.scalerPath}`;
            console.log(`📂 Loading scaler from: ${fullScalerPath}`);
            const scalerResponse = await fetch(fullScalerPath);
            if (!scalerResponse.ok) {
                throw new Error(`Failed to load scaler: ${scalerResponse.status} ${scalerResponse.statusText}`);
            }
            this.scaler = await scalerResponse.json();
            console.log('✅ Scaler parameters loaded');
            
            // Load preprocessing metadata (optional - for feature selection models)
            if (this.preprocessingPath) {
                const fullPreprocessingPath = `${baseUrl}${this.preprocessingPath}`;
                console.log(`📂 Loading preprocessing from: ${fullPreprocessingPath}`);
                const preprocessingResponse = await fetch(fullPreprocessingPath);
                if (preprocessingResponse.ok) {
                    this.preprocessing = await preprocessingResponse.json();
                    console.log(`✅ Preprocessing metadata loaded`);
                    console.log(`   🎯 Feature selection: ${this.preprocessing.selected_feature_indices.length} features from 43`);
                } else {
                    console.log('ℹ️  No preprocessing metadata (using all features)');
                }
            }
            
            console.log(`🎯 Using threshold: ${this.threshold}`);
            
            this.isModelLoaded = true;
            return true;
        } catch (error) {
            console.error('❌ Error loading model:', error);
            console.error('Error details:', error.message);
            throw error;
        }
    }

    /**
     * Extract 43 features from eye-tracking data
     * Enhanced with clinically-validated metrics for ASD detection
     * 
     * Feature Categories:
     * - Original 28 features: Basic spatial, temporal, movement metrics
     * - New 15 features: Advanced eye-tracking patterns linked to ASD research
     * 
     * Note: This ALWAYS extracts all 43 features. Feature selection is applied later if needed.
     */
    extractFeatures(data) {
        // Helper to get min/max without stack overflow
        const safeMin = (arr) => arr.reduce((min, val) => val < min ? val : min, Infinity);
        const safeMax = (arr) => arr.reduce((max, val) => val > max ? val : max, -Infinity);
        
        const xValues = data.map(d => d.x);
        const yValues = data.map(d => d.y);
        
        // Original 28 features (indices 0-27)
        const stats = {
            x_mean: this.mean(xValues),            // 0
            x_std: this.std(xValues),              // 1 ✅
            x_min: safeMin(xValues),               // 2
            x_max: safeMax(xValues),               // 3
            y_mean: this.mean(yValues),            // 4
            y_std: this.std(yValues),              // 5
            y_min: safeMin(yValues),               // 6
            y_max: safeMax(yValues),               // 7
            fixation_duration_mean: this.mean(data.map(d => d.fixation_duration || 200)), // 8
            fixation_duration_std: this.std(data.map(d => d.fixation_duration || 200)),  // 9
            fixation_count: data.length,           // 10 ✅
            saccade_velocity_mean: this.mean(this.calculateVelocities(data)),  // 11 ✅
            saccade_velocity_std: this.std(this.calculateVelocities(data)),    // 12 ✅
            saccade_amplitude_mean: this.mean(this.calculateAmplitudes(data)), // 13 ✅
            saccade_amplitude_std: this.std(this.calculateAmplitudes(data)),   // 14 ✅
            pupil_size_mean: this.mean(data.map(d => d.pupil_size || 3.5)),   // 15
            pupil_size_std: this.std(data.map(d => d.pupil_size || 3.5)),     // 16
            spatial_coverage: this.calculateSpatialCoverage(data),    // 17
            fixation_dispersion: this.calculateDispersion(data),      // 18 ✅
            scan_path_length: this.calculateScanPathLength(data),     // 19
            gaze_entropy: this.calculateEntropy(data),                // 20 ✅
            center_bias: this.calculateCenterBias(data),              // 21 ✅
            edge_bias: this.calculateEdgeBias(data),                  // 22 ✅
            roi_focus: this.calculateROIFocus(data),                  // 23 ✅
            vertical_horizontal_ratio: this.calculateVHRatio(data),   // 24 ✅
            temporal_consistency: this.calculateTemporalConsistency(data),  // 25 ✅
            attention_switches: this.calculateAttentionSwitches(data),      // 26
            revisit_rate: this.calculateRevisitRate(data),                  // 27
            
            // ===== NEW ADVANCED FEATURES (15) - Indices 28-42 =====
            
            // 1. Saccade Directional Entropy (Clinical: Atypical scanning patterns in ASD)
            // Lower entropy = repetitive/stereotyped gaze patterns
            saccade_direction_entropy: this.calculateDirectionalEntropy(data),  // 28 ✅
            
            // 2-3. Spatial Autocorrelation (Clinical: Attention stability)
            // Higher values = more predictable gaze patterns
            spatial_autocorr_x: this.calculateAutocorrelation(xValues),  // 29 ✅
            spatial_autocorr_y: this.calculateAutocorrelation(yValues),  // 30 ✅
            
            // 4. Fixation Cluster Density (Clinical: Areas of Interest focus)
            // ASD often shows reduced clustering on social stimuli
            fixation_cluster_density: this.calculateClusterDensity(data),  // 31 ✅
            
            // 5. First Fixation Center Bias (Clinical: Initial attention allocation)
            // ASD may show peripheral bias on face stimuli
            first_fixation_center_dist: this.calculateFirstFixationBias(data),  // 32
            
            // 6. Spatial Revisitation Rate (Clinical: Repetitive behaviors)
            // Higher rates may indicate compulsive re-checking
            spatial_revisitation_rate: this.calculateSpatialRevisitation(data),  // 33 ✅
            
            // 7-8. Velocity Distribution Shape (Clinical: Movement planning)
            // Skewness/kurtosis reveal saccade planning differences
            velocity_skewness: this.calculateSkewness(this.calculateVelocities(data)),  // 34
            velocity_kurtosis: this.calculateKurtosis(this.calculateVelocities(data)),  // 35
            
            // 9. Inter-Saccadic Interval Variability (Clinical: Timing consistency)
            // Higher CV = irregular attention shifts (common in ADHD/ASD)
            isi_coefficient_variation: this.calculateISIVariability(data),  // 36
            
            // 10. Ambient vs Focal Attention Ratio (Clinical: Processing style)
            // Short fixations (ambient) vs long (focal) - cognitive strategy marker
            ambient_focal_ratio: this.calculateAmbientFocalRatio(data),  // 37
            
            // 11. Saccade Amplitude Entropy (Clinical: Movement diversity)
            // Reduced diversity may indicate restricted exploration
            saccade_amplitude_entropy: this.calculateAmplitudeEntropy(data),  // 38 ✅
            
            // 12. Normalized Scanpath Length (Clinical: Efficiency)
            // Ratio of actual path to minimum spanning tree
            scanpath_efficiency: this.calculateScanpathEfficiency(data),  // 39
            
            // 13. Fixation Duration Entropy (Clinical: Processing variability)
            // ASD may show more uniform fixation durations
            fixation_duration_entropy: this.calculateFixationDurationEntropy(data),  // 40 ✅
            
            // 14. Cross-Correlation XY (Clinical: Diagonal movement bias)
            // Measures coordinated horizontal-vertical movement
            cross_correlation_xy: this.calculateCrossCorrelation(xValues, yValues),  // 41
            
            // 15. Peak Velocity Ratio (Clinical: Ballistic vs corrective saccades)
            // Ratio of max to mean velocity
            peak_velocity_ratio: this.calculatePeakVelocityRatio(data)  // 42
        };

        // Convert to array (all 43 features)
        const allFeatures = Object.values(stats);
        
        // Apply feature selection if preprocessing metadata is loaded
        if (this.preprocessing && this.preprocessing.selected_feature_indices) {
            const selectedFeatures = this.preprocessing.selected_feature_indices.map(idx => allFeatures[idx]);
            console.log(`🎯 Feature selection applied: ${allFeatures.length} → ${selectedFeatures.length} features`);
            return selectedFeatures;
        }
        
        // Otherwise return all 43 features
        return allFeatures;
    }

    /**
     * Transform features using StandardScaler
     */
    transformFeatures(features) {
        if (!this.scaler) {
            throw new Error('Scaler not loaded');
        }

        // Fixed: scaler.json has 'std' not 'scale' (v2.0)
        const transformed = features.map((value, i) => {
            const std = this.scaler.std || this.scaler.scale;
            return (value - this.scaler.mean[i]) / std[i];
        });

        return transformed;
    }

    /**
     * Compare input data to baseline
     */
    async compareToBaseline(data) {
        if (!this.isModelLoaded) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        try {
            // Extract features
            const features = this.extractFeatures(data);
            console.log('📊 Extracted features:', features.length);

            // Transform with scaler
            const scaledFeatures = this.transformFeatures(features);

            // Run through autoencoder
            const inputTensor = tf.tensor2d([scaledFeatures], [1, 43]); // Updated to 43 features
            const reconstruction = await this.model.predict(inputTensor);
            const reconstructedFeatures = await reconstruction.data();
            
            // Calculate reconstruction error (MAE)
            let totalError = 0;
            for (let i = 0; i < scaledFeatures.length; i++) {
                totalError += Math.abs(scaledFeatures[i] - reconstructedFeatures[i]);
            }
            const reconstructionError = totalError / scaledFeatures.length;

            // Calculate similarity score (0-100) using age-specific threshold
            const baselineMeanError = this.threshold; // Use configurable threshold
            const errorDiff = Math.abs(reconstructionError - baselineMeanError);
            const similarityScore = Math.max(0, 100 * (1 - errorDiff / baselineMeanError));

            // Cleanup tensors
            inputTensor.dispose();
            reconstruction.dispose();

            return {
                reconstructionError,
                similarityScore: similarityScore.toFixed(1),
                interpretation: this.interpretScore(similarityScore),
                features,
                scaledFeatures
            };

        } catch (error) {
            console.error('❌ Error during comparison:', error);
            throw error;
        }
    }

    /**
     * Interpret similarity score
     */
    interpretScore(score) {
        if (score >= 85) return 'Very similar to baseline (high similarity)';
        if (score >= 70) return 'Similar to baseline (moderate-high similarity)';
        if (score >= 50) return 'Somewhat similar to baseline (moderate similarity)';
        if (score >= 30) return 'Moderately different from baseline';
        return 'Significantly different from baseline';
    }

    // Helper statistical functions
    mean(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }

    std(arr) {
        const avg = this.mean(arr);
        const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
        return Math.sqrt(this.mean(squareDiffs));
    }

    calculateVelocities(data) {
        const velocities = [];
        for (let i = 1; i < data.length; i++) {
            const dx = data[i].x - data[i-1].x;
            const dy = data[i].y - data[i-1].y;
            const dt = (data[i].timestamp - data[i-1].timestamp) / 1000;
            const distance = Math.sqrt(dx*dx + dy*dy);
            velocities.push(dt > 0 ? distance / dt : 0);
        }
        return velocities.length > 0 ? velocities : [0];
    }

    calculateAmplitudes(data) {
        const amplitudes = [];
        for (let i = 1; i < data.length; i++) {
            const dx = data[i].x - data[i-1].x;
            const dy = data[i].y - data[i-1].y;
            amplitudes.push(Math.sqrt(dx*dx + dy*dy));
        }
        return amplitudes.length > 0 ? amplitudes : [0];
    }

    calculateSpatialCoverage(data) {
        const gridSize = 5;
        const visited = new Set();
        data.forEach(d => {
            const gridX = Math.floor(d.x / (100 / gridSize));
            const gridY = Math.floor(d.y / (100 / gridSize));
            visited.add(`${gridX},${gridY}`);
        });
        return visited.size / (gridSize * gridSize);
    }

    calculateDispersion(data) {
        const xs = data.map(d => d.x);
        const ys = data.map(d => d.y);
        return Math.sqrt(Math.pow(this.std(xs), 2) + Math.pow(this.std(ys), 2));
    }

    calculateScanPathLength(data) {
        let length = 0;
        for (let i = 1; i < data.length; i++) {
            const dx = data[i].x - data[i-1].x;
            const dy = data[i].y - data[i-1].y;
            length += Math.sqrt(dx*dx + dy*dy);
        }
        return length;
    }

    calculateEntropy(data) {
        const gridSize = 5;
        const grid = {};
        data.forEach(d => {
            const key = `${Math.floor(d.x / (100/gridSize))},${Math.floor(d.y / (100/gridSize))}`;
            grid[key] = (grid[key] || 0) + 1;
        });
        
        let entropy = 0;
        const total = data.length;
        Object.values(grid).forEach(count => {
            const p = count / total;
            entropy -= p * Math.log2(p);
        });
        return entropy;
    }

    calculateCenterBias(data) {
        const centerX = 50, centerY = 50;
        const distances = data.map(d => {
            const dx = d.x - centerX;
            const dy = d.y - centerY;
            return Math.sqrt(dx*dx + dy*dy);
        });
        return 1 / (1 + this.mean(distances));
    }

    calculateEdgeBias(data) {
        const edgeCount = data.filter(d => 
            d.x < 10 || d.x > 90 || d.y < 10 || d.y > 90
        ).length;
        return edgeCount / data.length;
    }

    calculateROIFocus(data) {
        const roiCount = data.filter(d => 
            d.x >= 30 && d.x <= 70 && d.y >= 30 && d.y <= 70
        ).length;
        return roiCount / data.length;
    }

    calculateVHRatio(data) {
        const vMovement = data.reduce((sum, d, i) => {
            if (i === 0) return 0;
            return sum + Math.abs(d.y - data[i-1].y);
        }, 0);
        
        const hMovement = data.reduce((sum, d, i) => {
            if (i === 0) return 0;
            return sum + Math.abs(d.x - data[i-1].x);
        }, 0);
        
        return hMovement > 0 ? vMovement / hMovement : 1;
    }

    calculateTemporalConsistency(data) {
        if (data.length < 2) return 0;
        
        const distances = [];
        for (let i = 1; i < data.length; i++) {
            const dx = data[i].x - data[i-1].x;
            const dy = data[i].y - data[i-1].y;
            distances.push(Math.sqrt(dx*dx + dy*dy));
        }
        
        return 1 / (1 + this.std(distances));
    }

    calculateAttentionSwitches(data) {
        const gridSize = 3;
        let switches = 0;
        
        for (let i = 1; i < data.length; i++) {
            const prevGrid = `${Math.floor(data[i-1].x / (100/gridSize))},${Math.floor(data[i-1].y / (100/gridSize))}`;
            const currGrid = `${Math.floor(data[i].x / (100/gridSize))},${Math.floor(data[i].y / (100/gridSize))}`;
            if (prevGrid !== currGrid) switches++;
        }
        
        return switches;
    }

    calculateRevisitRate(data) {
        const gridSize = 5;
        const visits = {};
        
        data.forEach(d => {
            const key = `${Math.floor(d.x / (100/gridSize))},${Math.floor(d.y / (100/gridSize))}`;
            visits[key] = (visits[key] || 0) + 1;
        });
        
        const revisits = Object.values(visits).filter(count => count > 1).length;
        return revisits / Object.keys(visits).length;
    }

    // ===== NEW ADVANCED FEATURE CALCULATIONS =====

    /**
     * Calculate saccade directional entropy
     * Measures diversity of saccade directions (0-3 bits)
     * Clinical: Lower values may indicate stereotyped scanning patterns in ASD
     */
    calculateDirectionalEntropy(data) {
        if (data.length < 2) return 0;
        
        const angles = [];
        for (let i = 1; i < data.length; i++) {
            const dx = data[i].x - data[i-1].x;
            const dy = data[i].y - data[i-1].y;
            if (dx !== 0 || dy !== 0) {
                angles.push(Math.atan2(dy, dx));
            }
        }
        
        if (angles.length === 0) return 0;
        
        // Bin into 8 directions (N, NE, E, SE, S, SW, W, NW)
        const bins = new Array(8).fill(0);
        angles.forEach(angle => {
            const binIdx = Math.floor(((angle + Math.PI) / (2 * Math.PI)) * 8) % 8;
            bins[binIdx]++;
        });
        
        // Calculate entropy
        const total = bins.reduce((a, b) => a + b, 0);
        const probs = bins.map(count => (count + 1e-10) / total);
        const entropy = -probs.reduce((sum, p) => sum + p * Math.log2(p), 0);
        
        return entropy;
    }

    /**
     * Calculate lag-1 autocorrelation
     * Measures temporal consistency of gaze position
     * Clinical: Higher values indicate more predictable gaze patterns
     */
    calculateAutocorrelation(values) {
        if (values.length < 2) return 0;
        
        const x1 = values.slice(0, -1);
        const x2 = values.slice(1);
        
        return this.pearsonCorr(x1, x2);
    }

    /**
     * Calculate fixation cluster density using simplified DBSCAN
     * Measures how fixations group into regions of interest
     * Clinical: ASD may show reduced clustering on social stimuli
     */
    calculateClusterDensity(data) {
        if (data.length < 5) return 0;
        
        const eps = 10; // Cluster radius (10% of normalized space)
        const minPoints = 3;
        const visited = new Set();
        let clusters = 0;
        
        for (let i = 0; i < data.length; i++) {
            if (visited.has(i)) continue;
            
            // Find neighbors within eps distance
            const neighbors = [];
            for (let j = 0; j < data.length; j++) {
                if (i === j) continue;
                const dist = Math.sqrt(
                    Math.pow(data[i].x - data[j].x, 2) +
                    Math.pow(data[i].y - data[j].y, 2)
                );
                if (dist < eps) neighbors.push(j);
            }
            
            // If enough neighbors, mark as cluster
            if (neighbors.length >= minPoints) {
                clusters++;
                visited.add(i);
                neighbors.forEach(n => visited.add(n));
            }
        }
        
        return clusters / data.length;
    }

    /**
     * Calculate first fixation distance from center
     * Measures initial attention allocation
     * Clinical: ASD may show peripheral bias when viewing faces
     */
    calculateFirstFixationBias(data) {
        if (data.length === 0) return 50; // Default to center distance
        
        const centerX = 50, centerY = 50;
        const firstX = data[0].x;
        const firstY = data[0].y;
        
        const dist = Math.sqrt(
            Math.pow(firstX - centerX, 2) +
            Math.pow(firstY - centerY, 2)
        );
        
        // Normalize by max possible distance (corner to center)
        const maxDist = Math.sqrt(50*50 + 50*50);
        return dist / maxDist * 100;
    }

    /**
     * Calculate spatial revisitation rate
     * Measures how often gaze returns to previously visited regions
     * Clinical: May indicate repetitive/compulsive viewing behaviors
     */
    calculateSpatialRevisitation(data) {
        if (data.length < 2) return 0;
        
        const gridSize = 10; // 10x10 grid
        const visited = new Map();
        let revisits = 0;
        
        data.forEach(point => {
            const cellX = Math.floor(point.x / gridSize);
            const cellY = Math.floor(point.y / gridSize);
            const cellKey = `${cellX},${cellY}`;
            
            if (visited.has(cellKey)) {
                revisits++;
            }
            visited.set(cellKey, (visited.get(cellKey) || 0) + 1);
        });
        
        return revisits / data.length;
    }

    /**
     * Calculate skewness of velocity distribution
     * Measures asymmetry of velocity distribution
     * Clinical: Reveals saccade planning differences
     */
    calculateSkewness(arr) {
        if (arr.length < 3) return 0;
        
        const mean = this.mean(arr);
        const std = this.std(arr);
        
        if (std === 0) return 0;
        
        const skew = arr.reduce((sum, x) => {
            return sum + Math.pow((x - mean) / std, 3);
        }, 0) / arr.length;
        
        return skew;
    }

    /**
     * Calculate kurtosis of velocity distribution
     * Measures "tailedness" of velocity distribution
     * Clinical: Excess kurtosis indicates extreme velocity events
     */
    calculateKurtosis(arr) {
        if (arr.length < 4) return 0;
        
        const mean = this.mean(arr);
        const std = this.std(arr);
        
        if (std === 0) return 0;
        
        const kurt = arr.reduce((sum, x) => {
            return sum + Math.pow((x - mean) / std, 4);
        }, 0) / arr.length - 3; // Subtract 3 for excess kurtosis
        
        return kurt;
    }

    /**
     * Calculate Inter-Saccadic Interval coefficient of variation
     * Measures timing regularity of saccades
     * Clinical: Higher values indicate irregular attention shifts (ADHD/ASD)
     */
    calculateISIVariability(data) {
        if (data.length < 2) return 0;
        
        const intervals = [];
        for (let i = 1; i < data.length; i++) {
            const dt = Math.abs(data[i].timestamp - data[i-1].timestamp);
            if (dt > 0) intervals.push(dt);
        }
        
        if (intervals.length === 0) return 0;
        
        const mean = this.mean(intervals);
        const std = this.std(intervals);
        
        return mean > 0 ? std / mean : 0; // Coefficient of Variation
    }

    /**
     * Calculate Ambient vs Focal attention ratio
     * Short fixations (<200ms) vs long (>400ms)
     * Clinical: Cognitive processing strategy marker
     */
    calculateAmbientFocalRatio(data) {
        const durations = data.map(d => d.fixation_duration || 200);
        
        const shortFix = durations.filter(d => d < 200).length;
        const longFix = durations.filter(d => d > 400).length;
        
        return longFix > 0 ? shortFix / longFix : shortFix;
    }

    /**
     * Calculate saccade amplitude entropy
     * Measures diversity of saccade sizes
     * Clinical: Reduced diversity may indicate restricted exploration
     */
    calculateAmplitudeEntropy(data) {
        const amplitudes = this.calculateAmplitudes(data);
        
        if (amplitudes.length < 2) return 0;
        
        // Bin amplitudes into 5 categories
        const max = Math.max(...amplitudes);
        const binSize = max / 5;
        const bins = new Array(5).fill(0);
        
        amplitudes.forEach(amp => {
            const binIdx = Math.min(4, Math.floor(amp / binSize));
            bins[binIdx]++;
        });
        
        // Calculate entropy
        const total = bins.reduce((a, b) => a + b, 0);
        const probs = bins.map(count => (count + 1e-10) / total);
        const entropy = -probs.reduce((sum, p) => sum + p * Math.log2(p), 0);
        
        return entropy;
    }

    /**
     * Calculate scanpath efficiency
     * Ratio of straight-line distance to actual path length
     * Clinical: Lower values may indicate inefficient visual search
     */
    calculateScanpathEfficiency(data) {
        if (data.length < 2) return 1;
        
        // Straight-line distance from start to end
        const dx = data[data.length-1].x - data[0].x;
        const dy = data[data.length-1].y - data[0].y;
        const straightDist = Math.sqrt(dx*dx + dy*dy);
        
        // Actual path length
        const actualDist = this.calculateScanPathLength(data);
        
        return actualDist > 0 ? straightDist / actualDist : 0;
    }

    /**
     * Calculate fixation duration entropy
     * Measures variability in processing time
     * Clinical: ASD may show more uniform fixation durations
     */
    calculateFixationDurationEntropy(data) {
        const durations = data.map(d => d.fixation_duration || 200);
        
        if (durations.length < 2) return 0;
        
        // Bin into 5 categories
        const min = Math.min(...durations);
        const max = Math.max(...durations);
        const range = max - min;
        
        if (range === 0) return 0;
        
        const bins = new Array(5).fill(0);
        durations.forEach(dur => {
            const binIdx = Math.min(4, Math.floor((dur - min) / range * 5));
            bins[binIdx]++;
        });
        
        // Calculate entropy
        const total = bins.reduce((a, b) => a + b, 0);
        const probs = bins.map(count => (count + 1e-10) / total);
        const entropy = -probs.reduce((sum, p) => sum + p * Math.log2(p), 0);
        
        return entropy;
    }

    /**
     * Calculate cross-correlation between X and Y movements
     * Measures coordinated horizontal-vertical movement
     * Clinical: May reveal diagonal scanning biases
     */
    calculateCrossCorrelation(xValues, yValues) {
        if (xValues.length < 2) return 0;
        
        return this.pearsonCorr(xValues, yValues);
    }

    /**
     * Calculate peak velocity ratio
     * Max velocity / mean velocity
     * Clinical: High ratios indicate ballistic saccades vs corrective movements
     */
    calculatePeakVelocityRatio(data) {
        const velocities = this.calculateVelocities(data);
        
        if (velocities.length === 0) return 1;
        
        const maxVel = Math.max(...velocities);
        const meanVel = this.mean(velocities);
        
        return meanVel > 0 ? maxVel / meanVel : 1;
    }

    /**
     * Calculate Pearson correlation coefficient
     * Helper for autocorrelation and cross-correlation
     */
    pearsonCorr(x, y) {
        if (x.length !== y.length || x.length === 0) return 0;
        
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
        
        const num = n * sumXY - sumX * sumY;
        const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return den === 0 ? 0 : num / den;
    }
}

// Export for use in HTML
if (typeof window !== 'undefined') {
    window.BaselineModelWeb = BaselineModelWeb;
}
