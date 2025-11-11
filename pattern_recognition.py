"""
AI-Powered Pattern Recognition for Eye-Tracking Data
Automatically detect reading behavior, expertise levels, and areas of interest.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy, iqr
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')


class GazePatternRecognizer:
    """
    Machine learning-based pattern recognition for eye-tracking data.
    Automatically classifies viewing behavior and user characteristics.
    """
    
    def __init__(self, data):
        """
        Initialize with eye-tracking data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Eye-tracking data with columns: x, y, timestamp, duration
        """
        self.data = data
        self.features = {}
        self._extract_features()
        
    def _extract_features(self):
        """Extract comprehensive features for pattern recognition."""
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Spatial features
        self.features['spatial_variance_x'] = np.var(x)
        self.features['spatial_variance_y'] = np.var(y)
        self.features['spatial_range_x'] = np.ptp(x)
        self.features['spatial_range_y'] = np.ptp(y)
        
        # Movement features
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        
        self.features['mean_saccade_length'] = np.mean(distances)
        self.features['std_saccade_length'] = np.std(distances)
        self.features['max_saccade_length'] = np.max(distances)
        
        # Temporal features
        if 'timestamp' in self.data.columns:
            dt = np.diff(self.data['timestamp'].values)
            dt[dt == 0] = 1
            self.features['mean_saccade_duration'] = np.mean(dt)
            self.features['velocity_mean'] = np.mean(distances / dt)
            self.features['velocity_std'] = np.std(distances / dt)
        
        # Fixation features
        if 'duration' in self.data.columns:
            durations = self.data['duration'].values
            self.features['mean_fixation_duration'] = np.mean(durations)
            self.features['std_fixation_duration'] = np.std(durations)
            self.features['fixation_rate'] = len(durations) / (np.sum(durations) / 1000)
        
        # Entropy (unpredictability)
        self.features['spatial_entropy'] = self._calculate_spatial_entropy(x, y)
        self.features['scanpath_regularity'] = self._calculate_scanpath_regularity(x, y)
        
    def _calculate_spatial_entropy(self, x, y, bins=20):
        """Calculate spatial entropy of gaze distribution."""
        H, _, _ = np.histogram2d(x, y, bins=bins)
        H = H.flatten()
        H = H[H > 0]  # Remove zeros
        H = H / np.sum(H)  # Normalize
        return entropy(H)
    
    def _calculate_scanpath_regularity(self, x, y):
        """Calculate regularity of scan path (lower = more regular)."""
        if len(x) < 3:
            return 0.0
        
        # Calculate angles between consecutive movements
        dx = np.diff(x)
        dy = np.diff(y)
        angles = np.arctan2(dy, dx)
        angle_changes = np.abs(np.diff(angles))
        
        # Regularity is inverse of angle change variance
        return 1.0 / (np.var(angle_changes) + 0.01)
    
    def detect_reading_behavior(self):
        """
        Detect if user is reading vs scanning/exploring.
        
        Returns:
        --------
        dict with 'behavior' (reading/scanning/exploring) and confidence score
        """
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Reading indicators
        horizontal_movement = np.var(np.diff(x))
        vertical_progression = np.diff(y)
        
        # Calculate left-to-right progression with downward jumps
        left_to_right_ratio = np.sum(np.diff(x) > 0) / len(x) if len(x) > 1 else 0
        
        # Detect return sweeps (large leftward movements)
        large_leftward = np.sum(np.diff(x) < -100)
        
        # Line-by-line progression
        y_sorted_indices = np.argsort(y)
        y_sorted = y[y_sorted_indices]
        line_jumps = np.sum(np.abs(np.diff(y_sorted)) > 30)
        
        # Reading score
        reading_score = (
            (left_to_right_ratio * 0.4) +
            (large_leftward / max(len(x), 1) * 0.3) +
            (self.features['scanpath_regularity'] * 0.3)
        )
        
        # Scanning score (high spatial variance, short fixations)
        if 'duration' in self.data.columns:
            short_fixations = np.sum(self.data['duration'] < 200) / len(self.data)
        else:
            short_fixations = 0.5
        
        scanning_score = (
            (self.features['spatial_entropy'] / 4.0) * 0.5 +
            short_fixations * 0.3 +
            (1.0 - self.features['scanpath_regularity']) * 0.2
        )
        
        # Exploration score (random movements, high entropy)
        exploration_score = (
            self.features['spatial_entropy'] / 4.0 * 0.6 +
            (1.0 - self.features['scanpath_regularity']) * 0.4
        )
        
        scores = {
            'reading': reading_score,
            'scanning': scanning_score,
            'exploring': exploration_score
        }
        
        behavior = max(scores, key=scores.get)
        confidence = scores[behavior]
        
        return {
            'behavior': behavior,
            'confidence': min(confidence, 1.0),
            'scores': scores,
            'metrics': {
                'left_to_right_ratio': left_to_right_ratio,
                'return_sweeps': large_leftward,
                'line_jumps': line_jumps
            }
        }
    
    def classify_expertise_level(self):
        """
        Classify user as novice, intermediate, or expert based on gaze efficiency.
        
        Expert indicators:
        - Lower fixation count for same task
        - More direct scan paths
        - Shorter fixation durations
        - Higher scan path efficiency
        
        Returns:
        --------
        dict with 'expertise' level and confidence score
        """
        # Efficiency metrics
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Calculate path efficiency (straight line vs actual path)
        if len(x) > 1:
            straight_line_dist = euclidean([x[0], y[0]], [x[-1], y[-1]])
            actual_path = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            path_efficiency = straight_line_dist / (actual_path + 1)
        else:
            path_efficiency = 0.5
        
        # Fixation efficiency
        fixation_density = len(self.data) / (self.features['spatial_range_x'] * 
                                             self.features['spatial_range_y'] + 1)
        
        # Duration efficiency
        if 'duration' in self.data.columns:
            duration_efficiency = 1.0 / (self.features['mean_fixation_duration'] / 200)
        else:
            duration_efficiency = 0.5
        
        # Movement efficiency (less backtracking)
        direction_changes = np.sum(np.abs(np.diff(np.sign(np.diff(x)))))
        movement_efficiency = 1.0 / (direction_changes / len(x) + 1)
        
        # Calculate expertise score
        expertise_score = (
            path_efficiency * 0.3 +
            duration_efficiency * 0.25 +
            movement_efficiency * 0.25 +
            (1.0 - self.features['spatial_entropy'] / 4.0) * 0.2
        )
        
        # Classify
        if expertise_score > 0.7:
            level = 'expert'
            confidence = expertise_score
        elif expertise_score > 0.4:
            level = 'intermediate'
            confidence = 0.7
        else:
            level = 'novice'
            confidence = 1.0 - expertise_score
        
        return {
            'expertise': level,
            'confidence': min(confidence, 1.0),
            'score': expertise_score,
            'metrics': {
                'path_efficiency': path_efficiency,
                'fixation_density': fixation_density,
                'duration_efficiency': duration_efficiency,
                'movement_efficiency': movement_efficiency
            }
        }
    
    def detect_areas_of_interest(self, min_samples=5, eps=100):
        """
        Automatically detect Areas of Interest using DBSCAN clustering.
        
        Parameters:
        -----------
        min_samples : int
            Minimum number of points to form a cluster
        eps : float
            Maximum distance between two points to be in same cluster
            
        Returns:
        --------
        dict with AOI information
        """
        x = self.data['x'].values.reshape(-1, 1)
        y = self.data['y'].values.reshape(-1, 1)
        
        # Combine coordinates
        coords = np.hstack([x, y])
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_
        
        # Calculate AOI statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        aois = []
        for label in set(labels):
            if label == -1:  # Skip noise
                continue
            
            cluster_points = coords[labels == label]
            
            # Calculate AOI properties
            center_x = np.mean(cluster_points[:, 0])
            center_y = np.mean(cluster_points[:, 1])
            radius = np.max(np.sqrt(
                (cluster_points[:, 0] - center_x)**2 + 
                (cluster_points[:, 1] - center_y)**2
            ))
            
            # Dwell time
            if 'duration' in self.data.columns:
                cluster_durations = self.data['duration'].values[labels == label]
                dwell_time = np.sum(cluster_durations)
            else:
                dwell_time = len(cluster_points) * 200  # Estimate
            
            aois.append({
                'id': int(label),
                'center': (float(center_x), float(center_y)),
                'radius': float(radius),
                'n_fixations': int(len(cluster_points)),
                'dwell_time': float(dwell_time),
                'importance': float(dwell_time * len(cluster_points))
            })
        
        # Sort by importance
        aois.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'n_aois': n_clusters,
            'n_noise_points': n_noise,
            'aois': aois,
            'coverage': 1.0 - (n_noise / len(labels))
        }
    
    def detect_confusion_indicators(self):
        """
        Detect signs of user confusion or difficulty.
        
        Confusion indicators:
        - High number of revisits
        - Erratic movements
        - Long fixation durations
        - High spatial entropy
        
        Returns:
        --------
        dict with confusion score and indicators
        """
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Revisit detection (returning to same areas)
        revisits = 0
        for i in range(len(x) - 10):
            for j in range(i + 10, len(x)):
                if euclidean([x[i], y[i]], [x[j], y[j]]) < 50:
                    revisits += 1
        
        revisit_rate = revisits / len(x) if len(x) > 0 else 0
        
        # Movement erraticism (direction changes)
        if len(x) > 2:
            angles = np.arctan2(np.diff(y), np.diff(x))
            angle_changes = np.abs(np.diff(angles))
            erraticism = np.std(angle_changes)
        else:
            erraticism = 0
        
        # Long fixations indicator
        if 'duration' in self.data.columns:
            long_fixations = np.sum(self.data['duration'] > 500) / len(self.data)
        else:
            long_fixations = 0.3
        
        # High entropy (unpredictable)
        entropy_score = self.features['spatial_entropy'] / 4.0
        
        # Calculate confusion score
        confusion_score = (
            revisit_rate * 0.3 +
            min(erraticism / 2.0, 1.0) * 0.3 +
            long_fixations * 0.2 +
            entropy_score * 0.2
        )
        
        # Classify confusion level
        if confusion_score > 0.7:
            level = 'high'
        elif confusion_score > 0.4:
            level = 'moderate'
        else:
            level = 'low'
        
        return {
            'confusion_level': level,
            'confusion_score': min(confusion_score, 1.0),
            'indicators': {
                'revisit_rate': revisit_rate,
                'movement_erraticism': erraticism,
                'long_fixation_rate': long_fixations,
                'spatial_entropy': entropy_score
            },
            'recommendation': self._get_confusion_recommendation(level)
        }
    
    def _get_confusion_recommendation(self, level):
        """Get recommendation based on confusion level."""
        recommendations = {
            'high': 'User appears confused. Consider: simplifying interface, adding visual cues, or providing contextual help.',
            'moderate': 'Some difficulty detected. User may benefit from clearer navigation or information hierarchy.',
            'low': 'User appears to navigate efficiently with minimal confusion.'
        }
        return recommendations[level]
    
    def generate_pattern_summary(self):
        """
        Generate comprehensive summary of all detected patterns.
        
        Returns:
        --------
        dict with all pattern recognition results
        """
        summary = {
            'reading_behavior': self.detect_reading_behavior(),
            'expertise_level': self.classify_expertise_level(),
            'areas_of_interest': self.detect_areas_of_interest(),
            'confusion_indicators': self.detect_confusion_indicators(),
            'features': self.features
        }
        
        return summary
    
    def get_narrative_insights(self):
        """
        Generate human-readable narrative from pattern analysis.
        
        Returns:
        --------
        str: Natural language description of gaze patterns
        """
        reading = self.detect_reading_behavior()
        expertise = self.classify_expertise_level()
        confusion = self.detect_confusion_indicators()
        aois = self.detect_areas_of_interest()
        
        narrative = []
        
        # Behavior description
        behavior = reading['behavior']
        confidence = reading['confidence']
        narrative.append(
            f"The user exhibited {behavior} behavior with {confidence:.0%} confidence. "
        )
        
        # Expertise description
        exp_level = expertise['expertise']
        narrative.append(
            f"Based on gaze efficiency metrics, the user appears to be at an "
            f"{exp_level} level (efficiency score: {expertise['score']:.2f}). "
        )
        
        # AOI description
        if aois['n_aois'] > 0:
            top_aoi = aois['aois'][0]
            narrative.append(
                f"Analysis identified {aois['n_aois']} distinct areas of interest. "
                f"The primary focus area received {top_aoi['n_fixations']} fixations "
                f"with a total dwell time of {top_aoi['dwell_time']:.0f}ms. "
            )
        
        # Confusion description
        confusion_level = confusion['confusion_level']
        narrative.append(
            f"Confusion indicators show {confusion_level} difficulty. "
            f"{confusion['recommendation']}"
        )
        
        # Movement characteristics
        if 'mean_saccade_length' in self.features:
            narrative.append(
                f"\n\nMovement characteristics: Average saccade length of "
                f"{self.features['mean_saccade_length']:.1f} pixels with "
                f"{'high' if self.features['spatial_entropy'] > 2.5 else 'moderate'} "
                f"spatial entropy ({self.features['spatial_entropy']:.2f}), "
                f"indicating {'exploratory' if self.features['spatial_entropy'] > 2.5 else 'focused'} "
                f"viewing patterns."
            )
        
        return ''.join(narrative)


def compare_patterns(data1, data2):
    """
    Compare pattern recognition results between two datasets.
    
    Parameters:
    -----------
    data1, data2 : pandas.DataFrame
        Two eye-tracking datasets to compare
        
    Returns:
    --------
    dict with comparison results
    """
    recognizer1 = GazePatternRecognizer(data1)
    recognizer2 = GazePatternRecognizer(data2)
    
    summary1 = recognizer1.generate_pattern_summary()
    summary2 = recognizer2.generate_pattern_summary()
    
    comparison = {
        'behavior_match': summary1['reading_behavior']['behavior'] == 
                         summary2['reading_behavior']['behavior'],
        'expertise_difference': abs(
            summary1['expertise_level']['score'] - 
            summary2['expertise_level']['score']
        ),
        'confusion_difference': abs(
            summary1['confusion_indicators']['confusion_score'] - 
            summary2['confusion_indicators']['confusion_score']
        ),
        'aoi_count_difference': abs(
            summary1['areas_of_interest']['n_aois'] - 
            summary2['areas_of_interest']['n_aois']
        ),
        'summary1': summary1,
        'summary2': summary2
    }
    
    return comparison
