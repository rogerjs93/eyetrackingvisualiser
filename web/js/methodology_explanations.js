/**
 * Methodology and Formulations Documentation for JavaScript
 * Eye-Tracking Analysis - Mathematical Foundations and Interpretations
 */

const METHODOLOGY_EXPLANATIONS = {
    reading_behavior: {
        title: "Reading Behavior Detection",
        description: "Detects whether the user is reading, scanning, or exploring based on gaze movement patterns.",
        formulation: `Reading Score = (LTR_ratio × 0.4) + (return_sweeps/n × 0.3) + (regularity × 0.3)

Where:
- LTR_ratio = proportion of rightward movements (left-to-right reading)
- return_sweeps = count of large leftward jumps (>100px)
- regularity = 1 / (variance of angle changes + 0.01)

Scanning Score = (entropy/4 × 0.5) + (short_fixations × 0.3) + (1-regularity × 0.2)

Exploration Score = (entropy/4 × 0.6) + (1-regularity × 0.4)`,
        interpretation: `The behavior with the highest score is selected:
- Reading: High left-to-right progression, regular return sweeps, consistent line-by-line movement
- Scanning: Quick fixations, high spatial dispersion, less regular patterns
- Exploring: High entropy (unpredictable), irregular movements, wide spatial coverage`,
        references: "Based on Rayner (1998) 'Eye movements in reading and information processing' and Holmqvist et al. (2011)"
    },
    
    spatial_entropy: {
        title: "Spatial Entropy",
        description: "Measures unpredictability and dispersion of gaze distribution across the screen.",
        formulation: `Shannon Entropy:
H = -Σ p(i) × log₂(p(i))

Where:
- Screen divided into n×n grid (default 10×10)
- p(i) = proportion of fixations in bin i
- Sum over all bins with p(i) > 0`,
        interpretation: `Entropy values:
- High (>2.5): Dispersed attention, exploratory viewing
- Medium (1.5-2.5): Balanced distribution
- Low (<1.5): Focused attention on specific regions

Maximum entropy = log₂(n²) for uniform distribution`,
        references: "Shannon (1948) 'A Mathematical Theory of Communication', adapted for eye-tracking"
    },
    
    fixation_rate: {
        title: "Fixation Rate Analysis",
        description: "Measures visual processing speed through fixations per second.",
        formulation: `Fixation Rate = n_fixations / total_time (in seconds)

Mean Fixation Duration = Σ durations / n

Processing Indicator:
- Rate > 3 fps: High cognitive load / fast scanning
- Rate 2-3 fps: Normal processing
- Rate < 2 fps: Deep processing / careful examination`,
        interpretation: `Combined with duration:
- High rate + short durations = scanning/searching
- Low rate + long durations = careful reading/analysis
- High rate + long durations = difficulty/confusion`,
        references: "Just & Carpenter (1980), Rayner (1998)"
    },
    
    task_difficulty: {
        title: "Task Difficulty Score",
        description: "Composite measure of cognitive load from multiple indicators.",
        formulation: `Difficulty Score (0-10) = weighted_sum of:

1. Spatial Entropy (30%): entropy/4 × 3
2. Fixation Rate (20%): (rate/10) × 2
3. Mean Duration (20%): (duration/500) × 2
4. Saccade Length (20%): (length/200) × 2
5. Ambient Ratio (10%): ambient_ratio × 1

Normalized: min(sum, 10)`,
        interpretation: `Difficulty levels:
- Low (<4): Easy task, efficient processing
- Moderate (4-7): Normal cognitive load
- High (>7): Difficult task, high cognitive demand

Recommendations:
- High: Simplify interface, add guidance
- Moderate: Monitor user success
- Low: Appropriate difficulty level`,
        references: "Composite metric based on cognitive load theory (Sweller, 1988)"
    },
    
    heatmap: {
        title: "Gaze Heatmap",
        description: "2D density visualization of fixation distribution.",
        formulation: `1. Divide screen into n×n bins (typically 40×40 or 50×50)
2. Count fixations in each bin: H[i,j] = count(x,y in bin[i,j])
3. Optional: Apply Gaussian smoothing with σ=2
4. Normalize to [0,1] or use raw counts
5. Map to color scale (e.g., 'Hot', 'Viridis')`,
        interpretation: `Hot colors (red/yellow) = high attention density
Cool colors (blue/black) = low/no attention

Reveals:
- Primary areas of focus
- Visual attention patterns
- Interface effectiveness`,
        references: "Wooding (2002) 'Fixation maps'"
    },
    
    scan_path: {
        title: "Scan Path Visualization",
        description: "Sequential trajectory of gaze movements over time.",
        formulation: `Plot: (x₁,y₁) → (x₂,y₂) → ... → (xₙ,yₙ)

With:
- Line thickness ∝ fixation duration
- Color gradient: temporal progression
- Circle size ∝ duration at each point`,
        interpretation: `Patterns reveal:
- Reading: Left-right with return sweeps
- Scanning: Distributed with long jumps
- Focused: Clustered in small region
- Random: No clear structure`,
        references: "Noton & Stark (1971) 'Scanpaths in eye movements'"
    }
};

function getExplanation(metricName) {
    return METHODOLOGY_EXPLANATIONS[metricName] || {
        title: "Unknown Metric",
        description: "No explanation available.",
        formulation: "N/A",
        interpretation: "N/A",
        references: "N/A"
    };
}

function formatExplanationHTML(metricName) {
    const exp = getExplanation(metricName);
    
    return `
    <div style="padding: 15px; background: #f8f9fa; border-left: 4px solid #667eea; margin: 15px 0; border-radius: 5px;">
        <h4 style="color: #667eea; margin-top: 0;">ℹ️ ${exp.title}</h4>
        <p><strong>Description:</strong> ${exp.description}</p>
        <details>
            <summary style="cursor: pointer; color: #667eea; font-weight: bold;">📐 Mathematical Formulation</summary>
            <pre style="background: white; padding: 10px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap;">${exp.formulation}</pre>
        </details>
        <details>
            <summary style="cursor: pointer; color: #667eea; font-weight: bold;">💡 Interpretation</summary>
            <div style="background: white; padding: 10px; border-radius: 5px; white-space: pre-line;">${exp.interpretation}</div>
        </details>
        <p style="font-size: 12px; color: #6c757d; margin-bottom: 0;"><strong>References:</strong> ${exp.references}</p>
    </div>
    `;
}

function addExplanationToElement(elementId, metricName) {
    const element = document.getElementById(elementId);
    if (element) {
        const explanationDiv = document.createElement('div');
        explanationDiv.innerHTML = formatExplanationHTML(metricName);
        element.insertBefore(explanationDiv, element.firstChild);
    }
}
