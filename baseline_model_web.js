/**
 * Baseline Model Web Interface
 * Loads TensorFlow.js model and performs autism baseline comparison in browser
 */

class BaselineModelWeb {
    constructor() {
        this.model = null;
        this.scaler = null;
        this.baselineStats = null;
        this.isModelLoaded = false;
        
        // Default paths (can be overridden for different age groups)
        this.modelPath = 'models/baseline_children_asd_tfjs/model.json';
        this.scalerPath = 'models/baseline_children_asd_tfjs/scaler.json';
        this.threshold = 0.4069; // Default to children threshold
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
     * Extract 28 features from eye-tracking data
     */
    extractFeatures(data) {
        const stats = {
            x_mean: this.mean(data.map(d => d.x)),
            x_std: this.std(data.map(d => d.x)),
            x_min: Math.min(...data.map(d => d.x)),
            x_max: Math.max(...data.map(d => d.x)),
            y_mean: this.mean(data.map(d => d.y)),
            y_std: this.std(data.map(d => d.y)),
            y_min: Math.min(...data.map(d => d.y)),
            y_max: Math.max(...data.map(d => d.y)),
            fixation_duration_mean: this.mean(data.map(d => d.fixation_duration || 200)),
            fixation_duration_std: this.std(data.map(d => d.fixation_duration || 200)),
            fixation_count: data.length,
            saccade_velocity_mean: this.mean(this.calculateVelocities(data)),
            saccade_velocity_std: this.std(this.calculateVelocities(data)),
            saccade_amplitude_mean: this.mean(this.calculateAmplitudes(data)),
            saccade_amplitude_std: this.std(this.calculateAmplitudes(data)),
            pupil_size_mean: this.mean(data.map(d => d.pupil_size || 3.5)),
            pupil_size_std: this.std(data.map(d => d.pupil_size || 3.5)),
            spatial_coverage: this.calculateSpatialCoverage(data),
            fixation_dispersion: this.calculateDispersion(data),
            scan_path_length: this.calculateScanPathLength(data),
            gaze_entropy: this.calculateEntropy(data),
            center_bias: this.calculateCenterBias(data),
            edge_bias: this.calculateEdgeBias(data),
            roi_focus: this.calculateROIFocus(data),
            vertical_horizontal_ratio: this.calculateVHRatio(data),
            temporal_consistency: this.calculateTemporalConsistency(data),
            attention_switches: this.calculateAttentionSwitches(data),
            revisit_rate: this.calculateRevisitRate(data)
        };

        return Object.values(stats);
    }

    /**
     * Transform features using StandardScaler
     */
    transformFeatures(features) {
        if (!this.scaler) {
            throw new Error('Scaler not loaded');
        }

        const transformed = features.map((value, i) => {
            return (value - this.scaler.mean[i]) / this.scaler.scale[i];
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
            const inputTensor = tf.tensor2d([scaledFeatures], [1, 28]);
            const reconstruction = await this.model.predict(inputTensor);
            const reconstructedFeatures = await reconstruction.data();
            
            // Calculate reconstruction error (MAE)
            let totalError = 0;
            for (let i = 0; i < scaledFeatures.length; i++) {
                totalError += Math.abs(scaledFeatures[i] - reconstructedFeatures[i]);
            }
            const reconstructionError = totalError / scaledFeatures.length;

            // Calculate Z-scores relative to baseline
            const zScores = features.map((value, i) => {
                const featureName = Object.keys(this.baselineStats)[i];
                const baselineMean = this.baselineStats[featureName].mean;
                const baselineStd = this.baselineStats[featureName].std;
                return (value - baselineMean) / baselineStd;
            });

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
                zScores,
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
}

// Export for use in HTML
if (typeof window !== 'undefined') {
    window.BaselineModelWeb = BaselineModelWeb;
}
