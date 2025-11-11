"""
Methodology and Formulations Documentation
Eye-Tracking Analysis - Mathematical Foundations and Interpretations
"""

METHODOLOGY_EXPLANATIONS = {
    "reading_behavior": {
        "title": "Reading Behavior Detection",
        "description": "Detects whether the user is reading, scanning, or exploring based on gaze movement patterns.",
        "formulation": """
        Reading Score = (LTR_ratio × 0.4) + (return_sweeps/n × 0.3) + (regularity × 0.3)
        
        Where:
        - LTR_ratio = proportion of rightward movements (left-to-right reading)
        - return_sweeps = count of large leftward jumps (>100px)
        - regularity = 1 / (variance of angle changes + 0.01)
        
        Scanning Score = (entropy/4 × 0.5) + (short_fixations × 0.3) + (1-regularity × 0.2)
        
        Exploration Score = (entropy/4 × 0.6) + (1-regularity × 0.4)
        """,
        "interpretation": """
        The behavior with the highest score is selected:
        - Reading: High left-to-right progression, regular return sweeps, consistent line-by-line movement
        - Scanning: Quick fixations, high spatial dispersion, less regular patterns
        - Exploring: High entropy (unpredictable), irregular movements, wide spatial coverage
        """,
        "references": "Based on Rayner (1998) 'Eye movements in reading and information processing' and Holmqvist et al. (2011)"
    },
    
    "expertise_classification": {
        "title": "Expertise Level Classification",
        "description": "Classifies user expertise (novice/intermediate/expert) based on gaze efficiency metrics.",
        "formulation": """
        Path Efficiency = straight_line_distance / actual_path_length
        
        Duration Efficiency = 1 / (mean_fixation_duration / 200ms)
        
        Movement Efficiency = 1 / (direction_changes/n + 1)
        
        Expertise Score = (path_eff × 0.3) + (duration_eff × 0.25) + 
                         (movement_eff × 0.25) + ((1 - entropy/4) × 0.2)
        """,
        "interpretation": """
        Classification thresholds:
        - Expert: Score > 0.7 (efficient paths, shorter fixations, minimal backtracking)
        - Intermediate: 0.4 < Score ≤ 0.7 (moderate efficiency)
        - Novice: Score ≤ 0.4 (inefficient paths, longer fixations, frequent backtracking)
        """,
        "references": "Based on expertise research by Gegenfurtner et al. (2011) and Jarodzka et al. (2010)"
    },
    
    "aoi_detection": {
        "title": "Areas of Interest (AOI) Detection",
        "description": "Automatically identifies regions of high attention using density-based clustering.",
        "formulation": """
        DBSCAN Algorithm:
        - eps = 100 (maximum distance between points in same cluster)
        - min_samples = 5 (minimum points to form cluster)
        
        For each cluster:
        - Center = mean(x, y) of all points
        - Radius = max distance from center
        - Dwell Time = sum of fixation durations
        - Importance = dwell_time × n_fixations
        """,
        "interpretation": """
        Clusters represent Areas of Interest where:
        - High importance = long dwell time + many fixations
        - Coverage = (n_points_in_clusters / total_points)
        - Noise points (label = -1) indicate scattered attention
        """,
        "references": "DBSCAN: Ester et al. (1996), AOI analysis: Holmqvist et al. (2011)"
    },
    
    "confusion_detection": {
        "title": "Confusion Indicator Detection",
        "description": "Identifies signs of cognitive difficulty through revisits and erratic movements.",
        "formulation": """
        Revisit Rate = n_revisits / total_fixations
        where revisit = distance(point_i, point_j) < 50px, j > i + 10
        
        Movement Erraticism = std(angle_changes)
        where angle_changes = diff(arctan2(dy, dx))
        
        Long Fixation Rate = count(duration > 500ms) / n
        
        Confusion Score = (revisit_rate × 0.3) + (min(erraticism/2, 1) × 0.3) +
                         (long_fix_rate × 0.2) + (entropy/4 × 0.2)
        """,
        "interpretation": """
        Confusion levels:
        - High (>0.7): Many revisits, erratic movements, long fixations
        - Moderate (0.4-0.7): Some difficulty indicators
        - Low (<0.4): Smooth, efficient navigation
        """,
        "references": "Based on confusion detection by Gwizdka (2014) and cognitive load indicators"
    },
    
    "spatial_entropy": {
        "title": "Spatial Entropy",
        "description": "Measures unpredictability and dispersion of gaze distribution across the screen.",
        "formulation": """
        Shannon Entropy:
        H = -Σ p(i) × log₂(p(i))
        
        Where:
        - Screen divided into n×n grid (default 10×10)
        - p(i) = proportion of fixations in bin i
        - Sum over all bins with p(i) > 0
        """,
        "interpretation": """
        Entropy values:
        - High (>2.5): Dispersed attention, exploratory viewing
        - Medium (1.5-2.5): Balanced distribution
        - Low (<1.5): Focused attention on specific regions
        
        Maximum entropy = log₂(n²) for uniform distribution
        """,
        "references": "Shannon (1948) 'A Mathematical Theory of Communication', adapted for eye-tracking"
    },
    
    "fixation_rate": {
        "title": "Fixation Rate Analysis",
        "description": "Measures visual processing speed through fixations per second.",
        "formulation": """
        Fixation Rate = n_fixations / total_time (in seconds)
        
        Mean Fixation Duration = Σ durations / n
        
        Processing Indicator:
        - Rate > 3 fps: High cognitive load / fast scanning
        - Rate 2-3 fps: Normal processing
        - Rate < 2 fps: Deep processing / careful examination
        """,
        "interpretation": """
        Combined with duration:
        - High rate + short durations = scanning/searching
        - Low rate + long durations = careful reading/analysis
        - High rate + long durations = difficulty/confusion
        """,
        "references": "Just & Carpenter (1980), Rayner (1998)"
    },
    
    "saccade_metrics": {
        "title": "Saccade Metrics",
        "description": "Analyzes eye movement characteristics between fixations.",
        "formulation": """
        Saccade Length = √((x₂-x₁)² + (y₂-y₁)²)
        
        Mean Saccade Length = Σ lengths / (n-1)
        
        Saccade Rate = n_saccades / total_time
        
        Saccade Velocity = length / time_difference
        """,
        "interpretation": """
        Saccade patterns indicate:
        - Long saccades (>150px): Searching, overview
        - Short saccades (<50px): Reading, detailed inspection
        - High rate: Active visual search
        - Low rate: Prolonged fixations
        """,
        "references": "Holmqvist et al. (2011) 'Eye Tracking: A Comprehensive Guide'"
    },
    
    "ambient_focal_attention": {
        "title": "Ambient vs Focal Attention",
        "description": "Classifies attention mode based on fixation characteristics.",
        "formulation": """
        Classification threshold = 250ms
        
        Ambient Fixations: duration < 250ms
        - Rapid information gathering
        - Peripheral processing
        
        Focal Fixations: duration ≥ 250ms
        - Central foveal processing
        - Detailed examination
        
        Ambient Ratio = n_ambient / total
        Focal Ratio = n_focal / total
        """,
        "interpretation": """
        Dominant mode:
        - Ambient (>50% short fixations): Overview, navigation
        - Focal (>50% long fixations): Deep processing, analysis
        - Balanced: Task-dependent switching
        """,
        "references": "Velichkovsky et al. (2002) 'New solution to the modus ponens selection task'"
    },
    
    "gaze_transition_entropy": {
        "title": "Gaze Transition Entropy",
        "description": "Measures predictability of gaze movement patterns.",
        "formulation": """
        Transition Matrix T[i,j] = count(region_i → region_j)
        
        Normalize: P[i,j] = T[i,j] / Σⱼ T[i,j]
        
        Entropy: H = -Σᵢ Σⱼ P[i,j] × log₂(P[i,j])
        
        Grid: Screen divided into 5×5 regions
        """,
        "interpretation": """
        Entropy levels:
        - High: Unpredictable, random movements
        - Medium: Semi-structured patterns
        - Low: Predictable, repetitive patterns
        
        Low entropy suggests:
        - Systematic scanning strategy
        - Task-driven behavior
        - Expertise in domain
        """,
        "references": "Ellis & Stark (1986) 'Statistical dependency in visual scanning'"
    },
    
    "task_difficulty": {
        "title": "Task Difficulty Score",
        "description": "Composite measure of cognitive load from multiple indicators.",
        "formulation": """
        Difficulty Score (0-10) = weighted_sum of:
        
        1. Spatial Entropy (30%): entropy/4 × 3
        2. Fixation Rate (20%): (rate/10) × 2
        3. Mean Duration (20%): (duration/500) × 2
        4. Saccade Length (20%): (length/200) × 2
        5. Ambient Ratio (10%): ambient_ratio × 1
        
        Normalized: min(sum, 10)
        """,
        "interpretation": """
        Difficulty levels:
        - Low (<4): Easy task, efficient processing
        - Moderate (4-7): Normal cognitive load
        - High (>7): Difficult task, high cognitive demand
        
        Recommendations:
        - High: Simplify interface, add guidance
        - Moderate: Monitor user success
        - Low: Appropriate difficulty level
        """,
        "references": "Composite metric based on cognitive load theory (Sweller, 1988)"
    },
    
    "heatmap": {
        "title": "Gaze Heatmap",
        "description": "2D density visualization of fixation distribution.",
        "formulation": """
        1. Divide screen into n×n bins (typically 40×40 or 50×50)
        2. Count fixations in each bin: H[i,j] = count(x,y in bin[i,j])
        3. Optional: Apply Gaussian smoothing with σ=2
        4. Normalize to [0,1] or use raw counts
        5. Map to color scale (e.g., 'Hot', 'Viridis')
        """,
        "interpretation": """
        Hot colors (red/yellow) = high attention density
        Cool colors (blue/black) = low/no attention
        
        Reveals:
        - Primary areas of focus
        - Visual attention patterns
        - Interface effectiveness
        """,
        "references": "Wooding (2002) 'Fixation maps'"
    },
    
    "scan_path": {
        "title": "Scan Path Visualization",
        "description": "Sequential trajectory of gaze movements over time.",
        "formulation": """
        Plot: (x₁,y₁) → (x₂,y₂) → ... → (xₙ,yₙ)
        
        With:
        - Line thickness ∝ fixation duration
        - Color gradient: temporal progression
        - Circle size ∝ duration at each point
        """,
        "interpretation": """
        Patterns reveal:
        - Reading: Left-right with return sweeps
        - Scanning: Distributed with long jumps
        - Focused: Clustered in small region
        - Random: No clear structure
        """,
        "references": "Noton & Stark (1971) 'Scanpaths in eye movements'"
    },
    
    "sankey_diagram": {
        "title": "Gaze Flow - Sankey Diagram",
        "description": "Shows flow of attention between screen regions.",
        "formulation": """
        1. Divide screen into grid (e.g., 4×4 or 6×6)
        2. Assign each fixation to region
        3. Count transitions: T[i→j] for all consecutive pairs
        4. Flow width ∝ transition count
        """,
        "interpretation": """
        Thick flows = frequent transitions
        Thin flows = rare transitions
        
        Reveals:
        - Navigation patterns
        - Information flow
        - Region relationships
        """,
        "references": "Adapted from Sankey diagrams for eye-tracking by Blascheck et al. (2014)"
    },
    
    "network_graph": {
        "title": "AOI Network Graph",
        "description": "Shows relationships between Areas of Interest.",
        "formulation": """
        Nodes: AOIs from DBSCAN clustering
        - Size ∝ number of fixations
        - Color ∝ total dwell time
        
        Edges: Transitions between AOIs
        - Width ∝ transition count
        - Only show if count ≥ threshold
        """,
        "interpretation": """
        Network structure reveals:
        - Central AOIs: High degree, many connections
        - Peripheral AOIs: Few connections
        - Paths: Common navigation sequences
        """,
        "references": "Graph-based scanpath analysis, Goldberg & Helfman (2010)"
    },
    
    "velocity_heatmap": {
        "title": "Velocity Heatmap",
        "description": "Shows eye movement speed across screen regions.",
        "formulation": """
        Velocity[i] = √((x[i+1]-x[i])² + (y[i+1]-y[i])²) / (t[i+1]-t[i])
        
        Average velocity per region:
        V_region = mean(velocities for points in region)
        
        Units: pixels per millisecond
        """,
        "interpretation": """
        High velocity (red):
        - Fast saccades
        - Scanning behavior
        - Low interest regions
        
        Low velocity (blue):
        - Careful examination
        - High interest areas
        - Reading zones
        """,
        "references": "Velocity-based analysis, Salvucci & Goldberg (2000)"
    },
    
    "4d_visualization": {
        "title": "4D Visualization (X, Y, Time, Duration)",
        "description": "Combines spatial position, temporal progression, and fixation duration.",
        "formulation": """
        3D scatter plot:
        - X-axis: horizontal position
        - Y-axis: vertical position  
        - Z-axis: time (normalized 0-1000)
        - Marker size: fixation duration
        - Marker color: time progression
        """,
        "interpretation": """
        Viewing angle reveals:
        - X-Y plane: Spatial distribution
        - X-Time or Y-Time: Movement over time
        - Rising trajectory: Temporal progression
        - Cluster heights: Temporal focus periods
        """,
        "references": "High-dimensional visualization techniques"
    }
}


def get_explanation(metric_name):
    """Get detailed explanation for a specific metric."""
    return METHODOLOGY_EXPLANATIONS.get(metric_name, {
        "title": "Unknown Metric",
        "description": "No explanation available.",
        "formulation": "N/A",
        "interpretation": "N/A",
        "references": "N/A"
    })


def format_explanation_html(metric_name):
    """Format explanation as HTML for display in UI."""
    exp = get_explanation(metric_name)
    
    html = f"""
    <div style="padding: 15px; background: #f8f9fa; border-left: 4px solid #667eea; margin: 15px 0; border-radius: 5px;">
        <h4 style="color: #667eea; margin-top: 0;">ℹ️ {exp['title']}</h4>
        <p><strong>Description:</strong> {exp['description']}</p>
        <details>
            <summary style="cursor: pointer; color: #667eea; font-weight: bold;">📐 Mathematical Formulation</summary>
            <pre style="background: white; padding: 10px; border-radius: 5px; overflow-x: auto;">{exp['formulation']}</pre>
        </details>
        <details>
            <summary style="cursor: pointer; color: #667eea; font-weight: bold;">💡 Interpretation</summary>
            <p style="background: white; padding: 10px; border-radius: 5px; white-space: pre-line;">{exp['interpretation']}</p>
        </details>
        <p style="font-size: 12px; color: #6c757d; margin-bottom: 0;"><strong>References:</strong> {exp['references']}</p>
    </div>
    """
    return html


def format_explanation_markdown(metric_name):
    """Format explanation as Markdown for documentation."""
    exp = get_explanation(metric_name)
    
    md = f"""
## {exp['title']}

**Description:** {exp['description']}

### Mathematical Formulation

```
{exp['formulation']}
```

### Interpretation

{exp['interpretation']}

**References:** {exp['references']}
"""
    return md
