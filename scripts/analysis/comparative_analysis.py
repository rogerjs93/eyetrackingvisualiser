"""
Comparative Analysis for Eye-Tracking Data
Compare multiple sessions, participants, or conditions with statistical testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import directed_hausdorff, euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple


class ComparativeAnalyzer:
    """
    Statistical comparison and analysis of multiple eye-tracking sessions.
    Supports participant comparison, A/B testing, and group analysis.
    """
    
    def __init__(self):
        """Initialize the comparative analyzer."""
        self.sessions = {}
        self.results = {}
        
    def add_session(self, session_id: str, data: pd.DataFrame, metadata: dict = None):
        """
        Add a session for comparison.
        
        Parameters:
        -----------
        session_id : str
            Unique identifier for the session
        data : pandas.DataFrame
            Eye-tracking data
        metadata : dict, optional
            Additional information (participant_id, condition, etc.)
        """
        self.sessions[session_id] = {
            'data': data,
            'metadata': metadata or {},
            'features': self._extract_features(data)
        }
    
    def _extract_features(self, data: pd.DataFrame) -> dict:
        """Extract comparable features from a session."""
        x = data['x'].values
        y = data['y'].values
        
        features = {
            # Spatial features
            'mean_x': np.mean(x),
            'mean_y': np.mean(y),
            'std_x': np.std(x),
            'std_y': np.std(y),
            'range_x': np.ptp(x),
            'range_y': np.ptp(y),
            
            # Count features
            'n_fixations': len(data),
            
            # Movement features
            'mean_saccade_length': np.mean(np.sqrt(np.diff(x)**2 + np.diff(y)**2)),
            'total_path_length': np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)),
        }
        
        # Duration features
        if 'duration' in data.columns:
            features['mean_duration'] = np.mean(data['duration'])
            features['total_duration'] = np.sum(data['duration'])
            features['std_duration'] = np.std(data['duration'])
        
        # Temporal features
        if 'timestamp' in data.columns:
            features['session_length'] = data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]
            dt = np.diff(data['timestamp'].values)
            dt[dt == 0] = 1
            distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            features['mean_velocity'] = np.mean(distances / dt)
        
        return features
    
    def compare_two_sessions(self, session_id1: str, session_id2: str) -> dict:
        """
        Compare two sessions with statistical tests.
        
        Parameters:
        -----------
        session_id1, session_id2 : str
            Session identifiers to compare
            
        Returns:
        --------
        dict with comparison results
        """
        if session_id1 not in self.sessions or session_id2 not in self.sessions:
            raise ValueError("Session IDs not found")
        
        data1 = self.sessions[session_id1]['data']
        data2 = self.sessions[session_id2]['data']
        features1 = self.sessions[session_id1]['features']
        features2 = self.sessions[session_id2]['features']
        
        comparison = {
            'session_1': session_id1,
            'session_2': session_id2,
            'feature_comparison': {},
            'statistical_tests': {},
            'spatial_similarity': {},
            'behavioral_differences': {}
        }
        
        # Compare features
        for feature in features1:
            if feature in features2:
                diff = features2[feature] - features1[feature]
                pct_change = (diff / features1[feature] * 100) if features1[feature] != 0 else 0
                
                comparison['feature_comparison'][feature] = {
                    'session_1': features1[feature],
                    'session_2': features2[feature],
                    'difference': diff,
                    'percent_change': pct_change
                }
        
        # Statistical tests on distributions
        x1, y1 = data1['x'].values, data1['y'].values
        x2, y2 = data2['x'].values, data2['y'].values
        
        # T-tests for spatial distributions
        t_stat_x, p_val_x = stats.ttest_ind(x1, x2)
        t_stat_y, p_val_y = stats.ttest_ind(y1, y2)
        
        comparison['statistical_tests']['x_position'] = {
            't_statistic': float(t_stat_x),
            'p_value': float(p_val_x),
            'significant': p_val_x < 0.05
        }
        
        comparison['statistical_tests']['y_position'] = {
            't_statistic': float(t_stat_y),
            'p_value': float(p_val_y),
            'significant': p_val_y < 0.05
        }
        
        # Kolmogorov-Smirnov test for distribution similarity
        ks_stat_x, ks_p_x = stats.ks_2samp(x1, x2)
        ks_stat_y, ks_p_y = stats.ks_2samp(y1, y2)
        
        comparison['statistical_tests']['distribution_similarity_x'] = {
            'ks_statistic': float(ks_stat_x),
            'p_value': float(ks_p_x),
            'similar': ks_p_x > 0.05
        }
        
        comparison['statistical_tests']['distribution_similarity_y'] = {
            'ks_statistic': float(ks_stat_y),
            'p_value': float(ks_p_y),
            'similar': ks_p_y > 0.05
        }
        
        # Duration comparison if available
        if 'duration' in data1.columns and 'duration' in data2.columns:
            dur1 = data1['duration'].values
            dur2 = data2['duration'].values
            
            t_stat_dur, p_val_dur = stats.ttest_ind(dur1, dur2)
            
            comparison['statistical_tests']['fixation_duration'] = {
                't_statistic': float(t_stat_dur),
                'p_value': float(p_val_dur),
                'significant': p_val_dur < 0.05,
                'mean_1': float(np.mean(dur1)),
                'mean_2': float(np.mean(dur2))
            }
        
        # Spatial similarity (Hausdorff distance)
        coords1 = np.column_stack([x1, y1])
        coords2 = np.column_stack([x2, y2])
        
        hausdorff_dist = max(
            directed_hausdorff(coords1, coords2)[0],
            directed_hausdorff(coords2, coords1)[0]
        )
        
        comparison['spatial_similarity'] = {
            'hausdorff_distance': float(hausdorff_dist),
            'similarity_score': float(1.0 / (1.0 + hausdorff_dist / 1000))
        }
        
        # Overall similarity score
        feature_similarity = []
        for feature in ['mean_x', 'mean_y', 'std_x', 'std_y']:
            if feature in features1 and feature in features2:
                diff = abs(features2[feature] - features1[feature])
                max_val = max(features1[feature], features2[feature])
                if max_val > 0:
                    similarity = 1.0 - (diff / max_val)
                    feature_similarity.append(max(0, similarity))
        
        comparison['overall_similarity'] = float(np.mean(feature_similarity))
        
        return comparison
    
    def group_analysis(self, group_sessions: List[str]) -> dict:
        """
        Analyze multiple sessions as a group.
        
        Parameters:
        -----------
        group_sessions : list of str
            Session IDs to include in group analysis
            
        Returns:
        --------
        dict with group statistics
        """
        if not all(sid in self.sessions for sid in group_sessions):
            raise ValueError("Some session IDs not found")
        
        # Collect all features
        feature_matrix = {}
        for feature_name in self.sessions[group_sessions[0]]['features']:
            feature_matrix[feature_name] = []
            
            for session_id in group_sessions:
                if feature_name in self.sessions[session_id]['features']:
                    feature_matrix[feature_name].append(
                        self.sessions[session_id]['features'][feature_name]
                    )
        
        # Calculate group statistics
        group_stats = {}
        for feature_name, values in feature_matrix.items():
            group_stats[feature_name] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0,
                'values': values
            }
        
        # ANOVA for spatial features
        x_positions = [self.sessions[sid]['data']['x'].values for sid in group_sessions]
        y_positions = [self.sessions[sid]['data']['y'].values for sid in group_sessions]
        
        f_stat_x, p_val_x = stats.f_oneway(*x_positions)
        f_stat_y, p_val_y = stats.f_oneway(*y_positions)
        
        group_stats['anova'] = {
            'x_position': {
                'f_statistic': float(f_stat_x),
                'p_value': float(p_val_x),
                'significant_difference': p_val_x < 0.05
            },
            'y_position': {
                'f_statistic': float(f_stat_y),
                'p_value': float(p_val_y),
                'significant_difference': p_val_y < 0.05
            }
        }
        
        return {
            'n_sessions': len(group_sessions),
            'group_statistics': group_stats,
            'session_ids': group_sessions
        }
    
    def ab_testing(self, condition_a: List[str], condition_b: List[str]) -> dict:
        """
        A/B testing between two conditions.
        
        Parameters:
        -----------
        condition_a : list of str
            Session IDs for condition A
        condition_b : list of str
            Session IDs for condition B
            
        Returns:
        --------
        dict with A/B test results
        """
        group_a = self.group_analysis(condition_a)
        group_b = self.group_analysis(condition_b)
        
        results = {
            'condition_a': group_a,
            'condition_b': group_b,
            'comparisons': {}
        }
        
        # Compare each feature
        for feature in group_a['group_statistics']:
            if feature in group_b['group_statistics']:
                values_a = group_a['group_statistics'][feature]['values']
                values_b = group_b['group_statistics'][feature]['values']
                
                # T-test
                t_stat, p_val = stats.ttest_ind(values_a, values_b)
                
                # Effect size (Cohen's d)
                mean_a = np.mean(values_a)
                mean_b = np.mean(values_b)
                std_pooled = np.sqrt((np.var(values_a) + np.var(values_b)) / 2)
                cohens_d = (mean_b - mean_a) / std_pooled if std_pooled != 0 else 0
                
                results['comparisons'][feature] = {
                    'mean_a': float(mean_a),
                    'mean_b': float(mean_b),
                    'difference': float(mean_b - mean_a),
                    'percent_change': float((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0,
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05,
                    'cohens_d': float(cohens_d),
                    'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
                }
        
        # Overall recommendation
        significant_features = [f for f, v in results['comparisons'].items() if v['significant']]
        
        if len(significant_features) > 0:
            results['recommendation'] = f"Significant differences found in {len(significant_features)} features: {', '.join(significant_features[:3])}"
        else:
            results['recommendation'] = "No significant differences detected between conditions"
        
        return results
    
    def generate_heatmap_comparison(self, session_ids: List[str], screen_width: int = 1920, 
                                   screen_height: int = 1080) -> np.ndarray:
        """
        Generate aggregated heatmap across multiple sessions.
        
        Parameters:
        -----------
        session_ids : list of str
            Sessions to aggregate
        screen_width, screen_height : int
            Screen dimensions
            
        Returns:
        --------
        numpy.ndarray: Combined heatmap
        """
        combined_heatmap = np.zeros((50, 50))
        
        for session_id in session_ids:
            if session_id not in self.sessions:
                continue
            
            data = self.sessions[session_id]['data']
            x = data['x'].values
            y = data['y'].values
            
            heatmap, _, _ = np.histogram2d(
                x, y, bins=[50, 50],
                range=[[0, screen_width], [0, screen_height]]
            )
            
            combined_heatmap += heatmap
        
        # Normalize
        combined_heatmap = combined_heatmap / len(session_ids)
        
        return combined_heatmap
    
    def calculate_consistency_score(self, session_ids: List[str]) -> dict:
        """
        Calculate how consistent gaze patterns are across sessions.
        
        Parameters:
        -----------
        session_ids : list of str
            Sessions to analyze for consistency
            
        Returns:
        --------
        dict with consistency metrics
        """
        if len(session_ids) < 2:
            return {'error': 'Need at least 2 sessions for consistency analysis'}
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(session_ids)):
            for j in range(i + 1, len(session_ids)):
                comparison = self.compare_two_sessions(session_ids[i], session_ids[j])
                similarities.append(comparison['overall_similarity'])
        
        return {
            'mean_consistency': float(np.mean(similarities)),
            'std_consistency': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'n_comparisons': len(similarities),
            'interpretation': self._interpret_consistency(np.mean(similarities))
        }
    
    def _interpret_consistency(self, score: float) -> str:
        """Interpret consistency score."""
        if score > 0.8:
            return "Very high consistency - participants show very similar gaze patterns"
        elif score > 0.6:
            return "High consistency - participants generally follow similar patterns"
        elif score > 0.4:
            return "Moderate consistency - some common patterns with individual variation"
        elif score > 0.2:
            return "Low consistency - significant individual differences in gaze behavior"
        else:
            return "Very low consistency - highly diverse gaze patterns"
    
    def identify_outliers(self, group_sessions: List[str], method: str = 'iqr') -> dict:
        """
        Identify outlier sessions that differ significantly from the group.
        
        Parameters:
        -----------
        group_sessions : list of str
            Sessions in the group
        method : str
            'iqr' for interquartile range or 'zscore'
            
        Returns:
        --------
        dict with outlier information
        """
        # Calculate similarity of each session to group mean
        similarities = []
        for session_id in group_sessions:
            other_sessions = [s for s in group_sessions if s != session_id]
            if len(other_sessions) == 0:
                continue
            
            # Compare to others
            session_similarities = []
            for other_id in other_sessions:
                comp = self.compare_two_sessions(session_id, other_id)
                session_similarities.append(comp['overall_similarity'])
            
            similarities.append({
                'session_id': session_id,
                'mean_similarity': np.mean(session_similarities)
            })
        
        sim_scores = [s['mean_similarity'] for s in similarities]
        
        if method == 'iqr':
            q1 = np.percentile(sim_scores, 25)
            q3 = np.percentile(sim_scores, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [s for s in similarities if s['mean_similarity'] < lower_bound]
        else:  # zscore
            z_scores = stats.zscore(sim_scores)
            outliers = [s for i, s in enumerate(similarities) if abs(z_scores[i]) > 2]
        
        return {
            'n_outliers': len(outliers),
            'outlier_sessions': outliers,
            'method': method,
            'group_mean_similarity': float(np.mean(sim_scores)),
            'group_std_similarity': float(np.std(sim_scores))
        }
    
    def export_comparison_report(self, filepath: str, comparison_type: str = 'all'):
        """
        Export comprehensive comparison report.
        
        Parameters:
        -----------
        filepath : str
            Path to save the report
        comparison_type : str
            'all', 'pairwise', or 'group'
        """
        report = []
        report.append("=" * 80)
        report.append("COMPARATIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal sessions: {len(self.sessions)}")
        report.append(f"Session IDs: {', '.join(self.sessions.keys())}")
        report.append("\n" + "=" * 80)
        
        # Add session summaries
        report.append("\nSESSION SUMMARIES")
        report.append("-" * 80)
        for session_id, session_data in self.sessions.items():
            report.append(f"\nSession: {session_id}")
            features = session_data['features']
            report.append(f"  Fixations: {features['n_fixations']}")
            report.append(f"  Mean position: ({features['mean_x']:.1f}, {features['mean_y']:.1f})")
            if 'mean_duration' in features:
                report.append(f"  Mean fixation duration: {features['mean_duration']:.1f}ms")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to: {filepath}")
