"""
Cognitive Load Analysis for Eye-Tracking Data
Measure cognitive effort, task difficulty, and mental workload from gaze patterns.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class CognitiveLoadAnalyzer:
    """
    Analyze cognitive load and mental effort from eye-tracking data.
    Implements multiple validated metrics from eye-tracking research.
    """
    
    def __init__(self, data: pd.DataFrame, screen_width: int = 1920, screen_height: int = 1080):
        """
        Initialize cognitive load analyzer.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Eye-tracking data with x, y, timestamp, duration
        screen_width, screen_height : int
            Screen dimensions
        """
        self.data = data
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.metrics = {}
        self._calculate_all_metrics()
    
    def _calculate_all_metrics(self):
        """Calculate all cognitive load metrics."""
        self.metrics['spatial_entropy'] = self.calculate_spatial_entropy()
        self.metrics['fixation_rate'] = self.calculate_fixation_rate()
        self.metrics['saccade_metrics'] = self.calculate_saccade_metrics()
        self.metrics['task_evoked_pupillary_response'] = self.calculate_tepr()
        self.metrics['ambient_focal_ratio'] = self.calculate_ambient_focal_attention()
        self.metrics['gaze_transition_entropy'] = self.calculate_gaze_transition_entropy()
    
    def calculate_spatial_entropy(self, bins: int = 20) -> Dict:
        """
        Calculate spatial entropy - measure of gaze distribution unpredictability.
        Higher entropy = more scattered attention = higher cognitive load.
        
        Returns:
        --------
        dict with entropy metrics
        """
        x = self.data['x'].values
        y = self.data['y'].values
        
        # 2D histogram
        H, xedges, yedges = np.histogram2d(
            x, y, bins=bins,
            range=[[0, self.screen_width], [0, self.screen_height]]
        )
        
        # Flatten and normalize
        H_flat = H.flatten()
        H_flat = H_flat[H_flat > 0]
        H_flat = H_flat / np.sum(H_flat)
        
        # Calculate entropy
        spatial_ent = entropy(H_flat)
        max_entropy = np.log2(bins * bins)
        normalized_entropy = spatial_ent / max_entropy
        
        # Marginal entropies
        x_hist, _ = np.histogram(x, bins=bins, range=[0, self.screen_width])
        x_hist = x_hist[x_hist > 0] / np.sum(x_hist)
        x_entropy = entropy(x_hist)
        
        y_hist, _ = np.histogram(y, bins=bins, range=[0, self.screen_height])
        y_hist = y_hist[y_hist > 0] / np.sum(y_hist)
        y_entropy = entropy(y_hist)
        
        # Interpretation
        if normalized_entropy > 0.8:
            interpretation = "Very high - highly dispersed attention, high cognitive load"
        elif normalized_entropy > 0.6:
            interpretation = "High - scattered attention pattern"
        elif normalized_entropy > 0.4:
            interpretation = "Moderate - balanced exploration and focus"
        elif normalized_entropy > 0.2:
            interpretation = "Low - concentrated attention"
        else:
            interpretation = "Very low - highly focused attention, low cognitive load"
        
        return {
            'spatial_entropy': float(spatial_ent),
            'normalized_entropy': float(normalized_entropy),
            'max_entropy': float(max_entropy),
            'x_entropy': float(x_entropy),
            'y_entropy': float(y_entropy),
            'interpretation': interpretation,
            'cognitive_load_indicator': 'high' if normalized_entropy > 0.6 else 'moderate' if normalized_entropy > 0.3 else 'low'
        }
    
    def calculate_fixation_rate(self) -> Dict:
        """
        Calculate fixation rate and related metrics.
        Lower fixation rate (longer fixations) = higher cognitive processing.
        
        Returns:
        --------
        dict with fixation rate metrics
        """
        if 'duration' not in self.data.columns:
            return {'error': 'Duration data required'}
        
        durations = self.data['duration'].values
        
        if 'timestamp' in self.data.columns:
            total_time = self.data['timestamp'].iloc[-1] - self.data['timestamp'].iloc[0]
            fixation_rate = len(durations) / (total_time / 1000)  # fixations per second
        else:
            fixation_rate = len(durations) / (np.sum(durations) / 1000)
        
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        cv_duration = std_duration / mean_duration if mean_duration > 0 else 0
        
        # Long fixations (> 500ms) indicate deep processing
        long_fixations = np.sum(durations > 500) / len(durations)
        
        # Short fixations (< 150ms) indicate search/scanning
        short_fixations = np.sum(durations < 150) / len(durations)
        
        # Cognitive load interpretation
        if mean_duration > 400:
            load = "High - extended processing time"
        elif mean_duration > 250:
            load = "Moderate - normal processing"
        else:
            load = "Low - rapid scanning"
        
        return {
            'fixation_rate': float(fixation_rate),
            'mean_duration': float(mean_duration),
            'std_duration': float(std_duration),
            'cv_duration': float(cv_duration),
            'long_fixation_ratio': float(long_fixations),
            'short_fixation_ratio': float(short_fixations),
            'cognitive_load_indicator': load
        }
    
    def calculate_saccade_metrics(self) -> Dict:
        """
        Calculate saccade-related cognitive load metrics.
        Longer saccades and higher velocity = higher cognitive load.
        
        Returns:
        --------
        dict with saccade metrics
        """
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Calculate saccade amplitudes
        dx = np.diff(x)
        dy = np.diff(y)
        amplitudes = np.sqrt(dx**2 + dy**2)
        
        # Saccade statistics
        mean_amplitude = np.mean(amplitudes)
        std_amplitude = np.std(amplitudes)
        max_amplitude = np.max(amplitudes)
        
        # Calculate velocities if timestamp available
        if 'timestamp' in self.data.columns:
            dt = np.diff(self.data['timestamp'].values)
            dt[dt == 0] = 1
            velocities = amplitudes / dt
            
            mean_velocity = np.mean(velocities)
            peak_velocity = np.max(velocities)
        else:
            mean_velocity = None
            peak_velocity = None
        
        # Long saccades (> 100px) indicate visual search
        long_saccades = np.sum(amplitudes > 100) / len(amplitudes)
        
        # Saccade frequency
        if 'timestamp' in self.data.columns:
            total_time = self.data['timestamp'].iloc[-1] - self.data['timestamp'].iloc[0]
            saccade_rate = len(amplitudes) / (total_time / 1000)
        else:
            saccade_rate = None
        
        # Cognitive load interpretation
        if mean_amplitude > 200:
            load = "High - extensive visual search"
        elif mean_amplitude > 100:
            load = "Moderate - active exploration"
        else:
            load = "Low - localized attention"
        
        metrics = {
            'mean_amplitude': float(mean_amplitude),
            'std_amplitude': float(std_amplitude),
            'max_amplitude': float(max_amplitude),
            'long_saccade_ratio': float(long_saccades),
            'cognitive_load_indicator': load
        }
        
        if mean_velocity is not None:
            metrics['mean_velocity'] = float(mean_velocity)
            metrics['peak_velocity'] = float(peak_velocity)
            metrics['saccade_rate'] = float(saccade_rate)
        
        return metrics
    
    def calculate_tepr(self) -> Dict:
        """
        Calculate Task-Evoked Pupillary Response proxy.
        Note: Requires pupil diameter data, uses fixation duration as proxy if not available.
        
        Returns:
        --------
        dict with TEPR metrics
        """
        if 'pupil_diameter' in self.data.columns:
            pupil = self.data['pupil_diameter'].values
            
            # Baseline (first 10% of data)
            baseline_length = max(1, len(pupil) // 10)
            baseline = np.mean(pupil[:baseline_length])
            
            # Calculate changes from baseline
            pupil_changes = pupil - baseline
            
            # TEPR metrics
            mean_change = np.mean(pupil_changes)
            max_dilation = np.max(pupil_changes)
            std_change = np.std(pupil_changes)
            
            # Cognitive load interpretation
            if mean_change > baseline * 0.1:
                load = "High - sustained pupil dilation indicates high cognitive effort"
            elif mean_change > baseline * 0.05:
                load = "Moderate - some cognitive effort detected"
            else:
                load = "Low - minimal pupil dilation"
            
            return {
                'baseline_pupil': float(baseline),
                'mean_change': float(mean_change),
                'max_dilation': float(max_dilation),
                'std_change': float(std_change),
                'cognitive_load_indicator': load,
                'data_available': True
            }
        else:
            # Use fixation duration as proxy
            if 'duration' in self.data.columns:
                durations = self.data['duration'].values
                mean_duration = np.mean(durations)
                
                return {
                    'note': 'Pupil data not available, using fixation duration proxy',
                    'mean_fixation_duration': float(mean_duration),
                    'cognitive_load_indicator': 'high' if mean_duration > 400 else 'moderate' if mean_duration > 250 else 'low',
                    'data_available': False
                }
            else:
                return {'error': 'Neither pupil nor duration data available'}
    
    def calculate_ambient_focal_attention(self) -> Dict:
        """
        Calculate ratio of ambient (global) vs focal (local) attention.
        Based on fixation duration and saccade amplitude.
        
        Ambient mode: Short fixations, long saccades (information gathering)
        Focal mode: Long fixations, short saccades (detailed processing)
        
        Returns:
        --------
        dict with ambient/focal metrics
        """
        if 'duration' not in self.data.columns:
            return {'error': 'Duration data required'}
        
        x = self.data['x'].values
        y = self.data['y'].values
        durations = self.data['duration'].values
        
        # Calculate saccade amplitudes
        amplitudes = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        
        # Classify each fixation
        # Ambient: duration < 200ms AND next saccade > 100px
        # Focal: duration > 300ms AND next saccade < 50px
        
        ambient_count = 0
        focal_count = 0
        
        for i in range(min(len(durations) - 1, len(amplitudes))):
            if durations[i] < 200 and amplitudes[i] > 100:
                ambient_count += 1
            elif durations[i] > 300 and amplitudes[i] < 50:
                focal_count += 1
        
        total_classified = ambient_count + focal_count
        
        if total_classified > 0:
            ambient_ratio = ambient_count / total_classified
            focal_ratio = focal_count / total_classified
        else:
            ambient_ratio = 0.5
            focal_ratio = 0.5
        
        # Interpretation
        if ambient_ratio > 0.7:
            mode = "Predominantly ambient - global information gathering, lower cognitive load"
        elif focal_ratio > 0.7:
            mode = "Predominantly focal - detailed processing, higher cognitive load"
        else:
            mode = "Mixed - balanced ambient and focal attention"
        
        return {
            'ambient_count': int(ambient_count),
            'focal_count': int(focal_count),
            'ambient_ratio': float(ambient_ratio),
            'focal_ratio': float(focal_ratio),
            'attention_mode': mode,
            'cognitive_load_indicator': 'high' if focal_ratio > 0.6 else 'moderate'
        }
    
    def calculate_gaze_transition_entropy(self) -> Dict:
        """
        Calculate entropy of gaze transitions between areas.
        Higher entropy = more unpredictable transitions = higher cognitive load.
        
        Returns:
        --------
        dict with transition entropy metrics
        """
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Divide screen into grid
        grid_size = 5
        x_bins = np.linspace(0, self.screen_width, grid_size + 1)
        y_bins = np.linspace(0, self.screen_height, grid_size + 1)
        
        # Assign each point to a cell
        x_indices = np.digitize(x, x_bins) - 1
        y_indices = np.digitize(y, y_bins) - 1
        
        # Create cell IDs
        cell_ids = x_indices * grid_size + y_indices
        
        # Calculate transition matrix
        n_cells = grid_size * grid_size
        transition_matrix = np.zeros((n_cells, n_cells))
        
        for i in range(len(cell_ids) - 1):
            from_cell = cell_ids[i]
            to_cell = cell_ids[i + 1]
            if 0 <= from_cell < n_cells and 0 <= to_cell < n_cells:
                transition_matrix[from_cell, to_cell] += 1
        
        # Calculate entropy for each source cell
        entropies = []
        for i in range(n_cells):
            row = transition_matrix[i]
            if np.sum(row) > 0:
                row_prob = row / np.sum(row)
                row_prob = row_prob[row_prob > 0]
                entropies.append(entropy(row_prob))
        
        if len(entropies) > 0:
            mean_entropy = np.mean(entropies)
            max_entropy = np.log2(n_cells)
            normalized_entropy = mean_entropy / max_entropy
        else:
            mean_entropy = 0
            normalized_entropy = 0
            max_entropy = np.log2(n_cells)
        
        # Interpretation
        if normalized_entropy > 0.7:
            interpretation = "High - unpredictable transitions, exploratory behavior"
        elif normalized_entropy > 0.4:
            interpretation = "Moderate - some structure with variability"
        else:
            interpretation = "Low - predictable transitions, structured viewing"
        
        return {
            'transition_entropy': float(mean_entropy),
            'normalized_entropy': float(normalized_entropy),
            'max_entropy': float(max_entropy),
            'interpretation': interpretation,
            'cognitive_load_indicator': 'high' if normalized_entropy > 0.6 else 'moderate' if normalized_entropy > 0.3 else 'low'
        }
    
    def calculate_index_of_cognitive_activity(self) -> Dict:
        """
        Calculate Index of Cognitive Activity (ICA).
        Detects small, rapid pupil dilations that correlate with cognitive processing.
        Note: Requires high-frequency pupil data (not typically available).
        
        Returns:
        --------
        dict with ICA metrics
        """
        if 'pupil_diameter' not in self.data.columns:
            return {
                'error': 'Pupil diameter data required for ICA',
                'alternative': 'Using fixation-based cognitive load proxy'
            }
        
        # This is a simplified proxy - real ICA requires high-frequency data
        pupil = self.data['pupil_diameter'].values
        
        # Detect rapid changes
        pupil_velocity = np.abs(np.diff(pupil))
        rapid_changes = np.sum(pupil_velocity > np.std(pupil_velocity))
        
        ica_proxy = rapid_changes / len(pupil)
        
        return {
            'ica_proxy': float(ica_proxy),
            'rapid_changes': int(rapid_changes),
            'note': 'Simplified ICA proxy - high-frequency data recommended for accurate ICA'
        }
    
    def measure_task_difficulty(self) -> Dict:
        """
        Estimate task difficulty based on multiple cognitive load indicators.
        
        Returns:
        --------
        dict with overall task difficulty assessment
        """
        # Collect indicators
        indicators = []
        
        # Spatial entropy indicator
        spatial = self.metrics['spatial_entropy']
        if spatial['cognitive_load_indicator'] == 'high':
            indicators.append(1.0)
        elif spatial['cognitive_load_indicator'] == 'moderate':
            indicators.append(0.5)
        else:
            indicators.append(0.0)
        
        # Fixation rate indicator
        fixation = self.metrics['fixation_rate']
        if 'cognitive_load_indicator' in fixation:
            if 'High' in fixation['cognitive_load_indicator']:
                indicators.append(1.0)
            elif 'Moderate' in fixation['cognitive_load_indicator']:
                indicators.append(0.5)
            else:
                indicators.append(0.0)
        
        # Saccade indicator
        saccade = self.metrics['saccade_metrics']
        if 'High' in saccade['cognitive_load_indicator']:
            indicators.append(1.0)
        elif 'Moderate' in saccade['cognitive_load_indicator']:
            indicators.append(0.5)
        else:
            indicators.append(0.0)
        
        # Ambient/focal indicator
        ambient_focal = self.metrics['ambient_focal_ratio']
        if ambient_focal['cognitive_load_indicator'] == 'high':
            indicators.append(1.0)
        else:
            indicators.append(0.5)
        
        # Calculate overall difficulty
        difficulty_score = np.mean(indicators)
        
        if difficulty_score > 0.7:
            difficulty = "High"
            recommendation = "Task appears cognitively demanding. Consider: simplifying content, improving layout, or providing additional guidance."
        elif difficulty_score > 0.4:
            difficulty = "Moderate"
            recommendation = "Task has moderate cognitive demands. Current design seems appropriate."
        else:
            difficulty = "Low"
            recommendation = "Task appears easy to process. Users navigate efficiently with low cognitive load."
        
        return {
            'difficulty_score': float(difficulty_score),
            'difficulty_level': difficulty,
            'contributing_factors': {
                'spatial_dispersion': spatial['cognitive_load_indicator'],
                'processing_time': fixation.get('cognitive_load_indicator', 'unknown'),
                'visual_search': saccade['cognitive_load_indicator'],
                'attention_mode': ambient_focal['cognitive_load_indicator']
            },
            'recommendation': recommendation
        }
    
    def generate_cognitive_load_report(self) -> str:
        """
        Generate comprehensive cognitive load analysis report.
        
        Returns:
        --------
        str: Formatted report
        """
        report = []
        report.append("=" * 80)
        report.append("COGNITIVE LOAD ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall task difficulty
        difficulty = self.measure_task_difficulty()
        report.append(f"Overall Task Difficulty: {difficulty['difficulty_level']}")
        report.append(f"Difficulty Score: {difficulty['difficulty_score']:.2f}")
        report.append(f"\n{difficulty['recommendation']}")
        report.append("")
        
        report.append("-" * 80)
        report.append("DETAILED METRICS")
        report.append("-" * 80)
        
        # Spatial entropy
        spatial = self.metrics['spatial_entropy']
        report.append(f"\n1. Spatial Entropy: {spatial['normalized_entropy']:.2f}")
        report.append(f"   {spatial['interpretation']}")
        
        # Fixation metrics
        fixation = self.metrics['fixation_rate']
        if 'mean_duration' in fixation:
            report.append(f"\n2. Fixation Analysis:")
            report.append(f"   Mean duration: {fixation['mean_duration']:.0f}ms")
            report.append(f"   Fixation rate: {fixation['fixation_rate']:.2f} per second")
            report.append(f"   {fixation['cognitive_load_indicator']}")
        
        # Saccade metrics
        saccade = self.metrics['saccade_metrics']
        report.append(f"\n3. Saccade Analysis:")
        report.append(f"   Mean amplitude: {saccade['mean_amplitude']:.1f} pixels")
        report.append(f"   {saccade['cognitive_load_indicator']}")
        
        # Ambient/Focal
        af = self.metrics['ambient_focal_ratio']
        if 'ambient_ratio' in af:
            report.append(f"\n4. Attention Mode:")
            report.append(f"   Ambient: {af['ambient_ratio']:.1%}, Focal: {af['focal_ratio']:.1%}")
            report.append(f"   {af['attention_mode']}")
        
        # Transition entropy
        transition = self.metrics['gaze_transition_entropy']
        report.append(f"\n5. Gaze Transition Entropy: {transition['normalized_entropy']:.2f}")
        report.append(f"   {transition['interpretation']}")
        
        report.append("\n" + "=" * 80)
        
        return '\n'.join(report)
    
    def get_all_metrics(self) -> Dict:
        """Return all calculated metrics."""
        return {
            **self.metrics,
            'task_difficulty': self.measure_task_difficulty()
        }
