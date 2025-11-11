"""
Demo script showcasing all advanced features:
- AI Pattern Recognition
- Comparative Analysis
- Cognitive Load Analysis
- Advanced Visualizations
"""

import numpy as np
import pandas as pd
from sample_data_generator import generate_sample_eyetracking_data
from pattern_recognition import GazePatternRecognizer
from comparative_analysis import ComparativeAnalyzer
from cognitive_load import CognitiveLoadAnalyzer
from advanced_visualizations import AdvancedVisualizer


def demo_pattern_recognition():
    """Demonstrate AI-powered pattern recognition."""
    print("\n" + "="*70)
    print("[AI] PATTERN RECOGNITION DEMO")
    print("="*70)
    
    # Generate reading pattern data
    data = generate_sample_eyetracking_data(pattern='reading', n_points=500)
    
    recognizer = GazePatternRecognizer(data)
    
    # Detect reading behavior
    print("\n1. Reading Behavior Detection:")
    reading = recognizer.detect_reading_behavior()
    print(f"   Behavior: {reading['behavior']}")
    print(f"   Confidence: {reading['confidence']:.2%}")
    print(f"   Left-to-right ratio: {reading['metrics']['left_to_right_ratio']:.2%}")
    print(f"   Return sweeps detected: {reading['metrics']['return_sweeps']}")
    
    # Classify expertise
    print("\n2. Expertise Classification:")
    expertise = recognizer.classify_expertise_level()
    print(f"   Level: {expertise['expertise']}")
    print(f"   Confidence: {expertise['confidence']:.2%}")
    print(f"   Path efficiency: {expertise['metrics']['path_efficiency']:.2%}")
    
    # Detect AOIs
    print("\n3. Areas of Interest Detection:")
    aois = recognizer.detect_areas_of_interest()
    print(f"   Number of AOIs found: {aois['n_aois']}")
    for i, aoi in enumerate(aois['aois'][:3]):  # Show first 3
        print(f"   AOI {i+1}: Center {aoi['center']}, "
              f"Size: {aoi['n_fixations']} fixations")
    
    # Detect confusion
    print("\n4. Confusion Indicators:")
    confusion = recognizer.detect_confusion_indicators()
    print(f"   Confusion level: {confusion['confusion_level']}")
    print(f"   Confusion score: {confusion['confusion_score']:.2%}")
    print(f"   Revisit rate: {confusion['indicators']['revisit_rate']:.2%}")
    print(f"   Movement erraticism: {confusion['indicators']['movement_erraticism']:.3f}")
    
    # Generate narrative
    print("\n5. Narrative Insights:")
    insights = recognizer.get_narrative_insights()
    print(f"\n{insights}")


def demo_comparative_analysis():
    """Demonstrate statistical comparison of sessions."""
    print("\n" + "="*70)
    print("[STATS] COMPARATIVE ANALYSIS DEMO")
    print("="*70)
    
    # Generate two different sessions
    session1 = generate_sample_eyetracking_data(pattern='reading', n_points=400)
    session2 = generate_sample_eyetracking_data(pattern='f_pattern', n_points=400)
    
    analyzer = ComparativeAnalyzer()
    
    # Add sessions
    analyzer.add_session('reading_session', session1, {'condition': 'reading'})
    analyzer.add_session('fpattern_session', session2, {'condition': 'f_pattern'})
    
    # Two-sample comparison
    print("\n1. Two-Sample Comparison (Reading vs F-Pattern):")
    comparison = analyzer.compare_two_sessions('reading_session', 'fpattern_session')
    
    if 'fixation_duration' in comparison['statistical_tests']:
        dur_test = comparison['statistical_tests']['fixation_duration']
        print(f"\n   Fixation Duration:")
        print(f"   - Session 1 mean: {dur_test['mean_1']:.2f}ms")
        print(f"   - Session 2 mean: {dur_test['mean_2']:.2f}ms")
        print(f"   - T-test p-value: {dur_test['p_value']:.4f}")
        print(f"   - Significant: {dur_test['significant']}")
    
    print(f"\n   Spatial Distribution:")
    print(f"   - X position p-value: {comparison['statistical_tests']['x_position']['p_value']:.4f}")
    print(f"   - Y position p-value: {comparison['statistical_tests']['y_position']['p_value']:.4f}")
    
    # Group analysis
    print("\n2. Group Analysis (4 Sessions):")
    sessions = [
        generate_sample_eyetracking_data(pattern='reading', n_points=300),
        generate_sample_eyetracking_data(pattern='reading', n_points=300),
        generate_sample_eyetracking_data(pattern='f_pattern', n_points=300),
        generate_sample_eyetracking_data(pattern='f_pattern', n_points=300)
    ]
    
    # Add all sessions
    for i, session in enumerate(sessions):
        analyzer.add_session(f'session_{i}', session, {'group': 'A' if i < 2 else 'B'})
    
    group_results = analyzer.group_analysis([f'session_{i}' for i in range(4)])
    
    print(f"   Number of sessions: {group_results['n_sessions']}")
    print(f"   ANOVA Results:")
    anova = group_results['group_statistics']['anova']
    print(f"   - X position: F={anova['x_position']['f_statistic']:.2f}, p={anova['x_position']['p_value']:.4f}")
    print(f"   - Y position: F={anova['y_position']['f_statistic']:.2f}, p={anova['y_position']['p_value']:.4f}")
    
    # A/B testing
    print("\n3. A/B Testing:")
    control = ['session_0', 'session_1']
    treatment = ['session_2', 'session_3']
    
    ab_results = analyzer.ab_testing(control, treatment)
    
    # Get a key metric
    if 'mean_saccade_length' in ab_results['comparisons']:
        metric = ab_results['comparisons']['mean_saccade_length']
        print(f"   Mean saccade length:")
        print(f"   - Control: {metric['mean_a']:.2f}")
        print(f"   - Treatment: {metric['mean_b']:.2f}")
        print(f"   - Effect size (Cohen's d): {metric['cohens_d']:.3f} ({metric['effect_size']})")
        print(f"   - Significant: {metric['significant']}")
    print(f"   Recommendation: {ab_results['recommendation']}")


def demo_cognitive_load():
    """Demonstrate cognitive load analysis."""
    print("\n" + "="*70)
    print("[COGNITIVE] LOAD ANALYSIS DEMO")
    print("="*70)
    
    # Generate data (scattered pattern = higher cognitive load)
    data = generate_sample_eyetracking_data(pattern='scattered', n_points=500)
    
    analyzer = CognitiveLoadAnalyzer(data)
    
    # Spatial entropy
    print("\n1. Spatial Entropy (Attention Distribution):")
    entropy = analyzer.calculate_spatial_entropy()
    print(f"   Entropy: {entropy['entropy']:.3f}")
    print(f"   Interpretation: {entropy['interpretation']}")
    
    # Fixation rate
    print("\n2. Fixation Rate Analysis:")
    fixation = analyzer.calculate_fixation_rate()
    print(f"   Fixations/second: {fixation['fixations_per_second']:.2f}")
    print(f"   Mean duration: {fixation['mean_fixation_duration']:.0f}ms")
    print(f"   Interpretation: {fixation['interpretation']}")
    
    # Saccade metrics
    print("\n3. Saccade Metrics (Visual Search):")
    saccades = analyzer.calculate_saccade_metrics()
    print(f"   Mean saccade length: {saccades['mean_saccade_length']:.1f}px")
    print(f"   Saccade rate: {saccades['saccade_rate']:.2f}/sec")
    print(f"   Interpretation: {saccades['interpretation']}")
    
    # Ambient/Focal attention
    print("\n4. Attention Mode Classification:")
    attention = analyzer.calculate_ambient_focal_attention()
    print(f"   Ambient ratio: {attention['ambient_ratio']:.2%}")
    print(f"   Focal ratio: {attention['focal_ratio']:.2%}")
    print(f"   Dominant mode: {attention['dominant_mode']}")
    
    # Gaze transition entropy
    print("\n5. Gaze Transition Entropy:")
    transition = analyzer.calculate_gaze_transition_entropy()
    print(f"   Entropy: {transition['entropy']:.3f}")
    print(f"   Interpretation: {transition['interpretation']}")
    
    # Overall task difficulty
    print("\n6. Task Difficulty Assessment:")
    difficulty = analyzer.measure_task_difficulty()
    print(f"   Overall score: {difficulty['overall_score']:.2f}/10")
    print(f"   Level: {difficulty['difficulty_level']}")
    print(f"   Recommendation: {difficulty['recommendation']}")


def demo_advanced_visualizations():
    """Demonstrate advanced visualizations."""
    print("\n" + "="*70)
    print("[VIZ] ADVANCED VISUALIZATIONS DEMO")
    print("="*70)
    
    # Generate data
    data = generate_sample_eyetracking_data(pattern='f_pattern', n_points=500)
    
    viz = AdvancedVisualizer(data)
    
    print("\n1. Creating Sankey Diagram (gaze flow)...")
    fig1 = viz.create_sankey_diagram(grid_size=4)
    fig1.write_html('demo_sankey.html')
    print("   ✓ Saved to demo_sankey.html")
    
    print("\n2. Creating Network Graph (AOI relationships)...")
    fig2 = viz.create_network_graph(threshold=3)
    fig2.write_html('demo_network.html')
    print("   ✓ Saved to demo_network.html")
    
    print("\n3. Creating 4D Visualization (X, Y, Time, Duration)...")
    fig3 = viz.create_4d_visualization()
    fig3.write_html('demo_4d.html')
    print("   ✓ Saved to demo_4d.html")
    
    print("\n4. Creating Animated Scan Path...")
    fig4 = viz.create_animated_scan_path(fps=10)
    fig4.write_html('demo_animated.html')
    print("   ✓ Saved to demo_animated.html")
    
    print("\n5. Creating Velocity Heatmap...")
    fig5 = viz.create_velocity_heatmap()
    fig5.write_html('demo_velocity.html')
    print("   ✓ Saved to demo_velocity.html")
    
    print("\n6. Creating Attention Timeline...")
    fig6 = viz.create_attention_timeline()
    fig6.write_html('demo_timeline.html')
    print("   ✓ Saved to demo_timeline.html")
    
    print("\n7. Creating Comparison Dashboard...")
    sessions = [
        generate_sample_eyetracking_data(pattern='reading', n_points=300),
        generate_sample_eyetracking_data(pattern='f_pattern', n_points=300),
        generate_sample_eyetracking_data(pattern='centered', n_points=300)
    ]
    fig7 = viz.create_comparison_dashboard(sessions, ['Reading', 'F-Pattern', 'Centered'])
    fig7.write_html('demo_comparison.html')
    print("   ✓ Saved to demo_comparison.html")
    
    print("\n8. Generating comprehensive HTML report...")
    viz.export_interactive_html('demo_full_report.html')
    print("   ✓ Full report saved to demo_full_report.html")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("EYE-TRACKING ADVANCED FEATURES DEMO")
    print("="*70)
    print("\nThis demo showcases the sophisticated features that fill gaps")
    print("in the eye-tracking analysis field:\n")
    print("1. AI-powered automatic pattern recognition")
    print("2. Rigorous statistical comparison methods")
    print("3. Cognitive load and task difficulty analysis")
    print("4. Publication-ready advanced visualizations")
    
    try:
        demo_pattern_recognition()
        demo_comparative_analysis()
        demo_cognitive_load()
        demo_advanced_visualizations()
        
        print("\n" + "="*70)
        print("[SUCCESS] ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("   • demo_sankey.html - Sankey diagram")
        print("   • demo_network.html - Network graph")
        print("   • demo_4d.html - 4D visualization")
        print("   • demo_animated.html - Animated scan path")
        print("   • demo_velocity.html - Velocity heatmap")
        print("   • demo_timeline.html - Attention timeline")
        print("   • demo_comparison.html - Multi-session comparison")
        print("   • demo_full_report.html - Comprehensive report")
        print("\nOpen these HTML files in your browser to explore!")
        
    except Exception as e:
        print(f"\n[ERROR] Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
