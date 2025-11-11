"""
Autism Eye-Tracking Data Loader
Loads and processes real autism eye-tracking data from CSV files.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


class AutismDataLoader:
    """Load and process autism eye-tracking data."""
    
    def __init__(self, data_dir='data/autism'):
        """Initialize the data loader.
        
        Args:
            data_dir: Path to the directory containing autism data
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / 'Metadata_Participants.csv'
        self.eyetracking_dir = self.data_dir / 'Eye-tracking Output'
        
        # Load metadata
        self.metadata = None
        if self.metadata_path.exists():
            self.metadata = pd.read_csv(self.metadata_path)
    
    def get_available_participants(self):
        """Get list of available participant IDs.
        
        Returns:
            List of participant IDs with data files
        """
        if not self.eyetracking_dir.exists():
            return []
        
        csv_files = list(self.eyetracking_dir.glob('*.csv'))
        participant_ids = [int(f.stem) for f in csv_files if f.stem.isdigit()]
        return sorted(participant_ids)
    
    def get_participant_info(self, participant_id):
        """Get metadata for a specific participant.
        
        Args:
            participant_id: Participant ID number
            
        Returns:
            Dictionary with participant information
        """
        if self.metadata is None:
            return None
        
        participant_row = self.metadata[self.metadata['ParticipantID'] == participant_id]
        if participant_row.empty:
            return None
        
        info = participant_row.iloc[0].to_dict()
        return info
    
    def load_participant_data(self, participant_id, screen_width=1920, screen_height=1080):
        """Load eye-tracking data for a specific participant.
        
        Args:
            participant_id: Participant ID number
            screen_width: Screen width in pixels (default: 1920)
            screen_height: Screen height in pixels (default: 1080)
            
        Returns:
            DataFrame with processed eye-tracking data in standard format
        """
        csv_path = self.eyetracking_dir / f'{participant_id}.csv'
        
        if not csv_path.exists():
            raise FileNotFoundError(f"No data file found for participant {participant_id}")
        
        # Load raw data
        raw_data = pd.read_csv(csv_path)
        
        # Process and convert to standard format
        processed_data = self._process_raw_data(raw_data, screen_width, screen_height)
        
        return processed_data
    
    def _process_raw_data(self, raw_data, screen_width, screen_height):
        """Process raw eye-tracking data into standard format.
        
        Args:
            raw_data: Raw DataFrame from CSV file
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            DataFrame with columns: x, y, timestamp, duration, event_type
        """
        # Extract relevant columns
        # Use Point of Regard (average of left and right eyes)
        x_right = raw_data['Point of Regard Right X [px]'].replace('-', np.nan).astype(float)
        y_right = raw_data['Point of Regard Right Y [px]'].replace('-', np.nan).astype(float)
        x_left = raw_data['Point of Regard Left X [px]'].replace('-', np.nan).astype(float)
        y_left = raw_data['Point of Regard Left Y [px]'].replace('-', np.nan).astype(float)
        
        # Average left and right eye positions (prefer available eye if one is missing)
        x = np.where(np.isnan(x_right), x_left, 
                    np.where(np.isnan(x_left), x_right, 
                            (x_right + x_left) / 2))
        y = np.where(np.isnan(y_right), y_left, 
                    np.where(np.isnan(y_left), y_right, 
                            (y_right + y_left) / 2))
        
        # Timestamps (in milliseconds)
        timestamp = raw_data['RecordingTime [ms]'].values
        
        # Event types (Fixation, Saccade, Blink)
        category_right = raw_data['Category Right'].values
        
        # Create processed dataframe
        processed = pd.DataFrame({
            'x': x,
            'y': y,
            'timestamp': timestamp,
            'event_type': category_right
        })
        
        # Remove invalid data points (blinks, missing data)
        processed = processed[
            (processed['x'].notna()) & 
            (processed['y'].notna()) &
            (processed['x'] > 0) &
            (processed['y'] > 0) &
            (processed['event_type'] != 'Blink')
        ].copy()
        
        # Sort by timestamp
        processed = processed.sort_values('timestamp').reset_index(drop=True)
        
        # Reset timestamp to start at 0
        if len(processed) > 0:
            min_timestamp = processed['timestamp'].min()
            processed['timestamp'] = processed['timestamp'] - min_timestamp
        
        # Calculate duration for each fixation
        processed['duration'] = self._calculate_durations(processed)
        
        # Ensure coordinates are within screen bounds
        processed['x'] = processed['x'].clip(0, screen_width)
        processed['y'] = processed['y'].clip(0, screen_height)
        
        return processed
    
    def _calculate_durations(self, data):
        """Calculate duration for each data point.
        
        Args:
            data: DataFrame with timestamp column
            
        Returns:
            Array of durations in milliseconds
        """
        if len(data) <= 1:
            return np.array([100])  # Default duration
        
        # Calculate time differences
        time_diffs = np.diff(data['timestamp'].values)
        
        # Use time diff as duration, with last point using median duration
        durations = np.append(time_diffs, np.median(time_diffs))
        
        # Clip to reasonable values (10ms to 5000ms)
        durations = np.clip(durations, 10, 5000)
        
        return durations
    
    def get_all_participants_summary(self):
        """Get summary information for all participants.
        
        Returns:
            DataFrame with participant summaries
        """
        participants = self.get_available_participants()
        summaries = []
        
        for pid in participants:
            info = self.get_participant_info(pid)
            try:
                data = self.load_participant_data(pid)
                n_points = len(data)
                duration = data['timestamp'].max() / 1000  # Convert to seconds
                
                summary = {
                    'ParticipantID': pid,
                    'N_Points': n_points,
                    'Duration_sec': duration,
                }
                
                if info:
                    summary.update(info)
                
                summaries.append(summary)
            except Exception as e:
                print(f"Error loading participant {pid}: {e}")
                continue
        
        return pd.DataFrame(summaries)
    
    def load_group_comparison_data(self, participant_ids):
        """Load data for multiple participants for group comparison.
        
        Args:
            participant_ids: List of participant IDs to load
            
        Returns:
            Dictionary mapping participant_id to DataFrame
        """
        data_dict = {}
        
        for pid in participant_ids:
            try:
                data = self.load_participant_data(pid)
                data_dict[pid] = data
            except Exception as e:
                print(f"Warning: Could not load participant {pid}: {e}")
        
        return data_dict


# Convenience function for quick loading
def load_autism_data(participant_id, data_dir='data/autism'):
    """Quick function to load autism eye-tracking data.
    
    Args:
        participant_id: Participant ID number
        data_dir: Path to data directory
        
    Returns:
        DataFrame with eye-tracking data
    """
    loader = AutismDataLoader(data_dir)
    return loader.load_participant_data(participant_id)


if __name__ == '__main__':
    # Demo usage
    print("=" * 70)
    print("Autism Eye-Tracking Data Loader - Demo")
    print("=" * 70)
    
    loader = AutismDataLoader()
    
    # Show available participants
    participants = loader.get_available_participants()
    print(f"\n📊 Found {len(participants)} participants with eye-tracking data")
    print(f"Participant IDs: {participants[:10]}..." if len(participants) > 10 else f"Participant IDs: {participants}")
    
    # Load first participant
    if participants:
        pid = participants[0]
        print(f"\n📈 Loading data for Participant {pid}...")
        
        # Get participant info
        info = loader.get_participant_info(pid)
        if info:
            print(f"\nParticipant Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        # Load data
        data = loader.load_participant_data(pid)
        print(f"\n✅ Loaded {len(data)} data points")
        print(f"Duration: {data['timestamp'].max()/1000:.1f} seconds")
        print(f"\nData sample:")
        print(data.head())
        
        # Show statistics
        print(f"\nStatistics:")
        print(f"  X range: {data['x'].min():.0f} - {data['x'].max():.0f} px")
        print(f"  Y range: {data['y'].min():.0f} - {data['y'].max():.0f} px")
        print(f"  Mean fixation duration: {data['duration'].mean():.1f} ms")
        print(f"  Event types: {data['event_type'].value_counts().to_dict()}")
    
    print("\n" + "=" * 70)
