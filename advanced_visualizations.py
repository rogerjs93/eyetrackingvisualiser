"""
Advanced Visualizations for Eye-Tracking Data
Includes Sankey diagrams, network graphs, 4D plots, and animated transitions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter
from typing import List, Dict, Tuple


class AdvancedVisualizer:
    """
    Advanced visualization techniques for eye-tracking data analysis.
    Provides unique, publication-ready visualizations.
    """
    
    def __init__(self, data: pd.DataFrame, screen_width: int = 1920, screen_height: int = 1080):
        """
        Initialize advanced visualizer.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Eye-tracking data
        screen_width, screen_height : int
            Screen dimensions
        """
        self.data = data
        self.screen_width = screen_width
        self.screen_height = screen_height
    
    def create_sankey_diagram(self, grid_size: int = 6) -> go.Figure:
        """
        Create Sankey diagram showing gaze flow between screen regions.
        Shows how attention moves between different areas.
        
        Parameters:
        -----------
        grid_size : int
            Divide screen into grid_size x grid_size regions
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Divide screen into grid
        x_bins = np.linspace(0, self.screen_width, grid_size + 1)
        y_bins = np.linspace(0, self.screen_height, grid_size + 1)
        
        # Assign each point to a region
        x_indices = np.digitize(x, x_bins) - 1
        y_indices = np.digitize(y, y_bins) - 1
        
        # Create region labels
        regions = []
        for i in range(len(x)):
            if 0 <= x_indices[i] < grid_size and 0 <= y_indices[i] < grid_size:
                region_id = y_indices[i] * grid_size + x_indices[i]
                regions.append(f"Region {region_id}")
            else:
                regions.append("Out of bounds")
        
        # Count transitions
        transitions = []
        for i in range(len(regions) - 1):
            if regions[i] != "Out of bounds" and regions[i+1] != "Out of bounds":
                transitions.append((regions[i], regions[i+1]))
        
        # Aggregate transition counts
        transition_counts = Counter(transitions)
        
        # Build Sankey data
        unique_regions = sorted(list(set([r for t in transitions for r in t])))
        region_to_idx = {region: i for i, region in enumerate(unique_regions)}
        
        source = []
        target = []
        values = []
        
        for (src, tgt), count in transition_counts.items():
            source.append(region_to_idx[src])
            target.append(region_to_idx[tgt])
            values.append(count)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=unique_regions,
                color="lightblue"
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                color="rgba(0,0,255,0.2)"
            )
        )])
        
        fig.update_layout(
            title="Gaze Flow Between Screen Regions",
            font=dict(size=12),
            height=600
        )
        
        return fig
    
    def create_network_graph(self, threshold: float = 3) -> go.Figure:
        """
        Create network graph showing relationships between fixation clusters.
        Nodes = Areas of Interest, Edges = Transitions between them.
        
        Parameters:
        -----------
        threshold : float
            Minimum transition count to show edge
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        from sklearn.cluster import DBSCAN
        
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Cluster fixations into AOIs
        coords = np.column_stack([x, y])
        clustering = DBSCAN(eps=100, min_samples=5).fit(coords)
        labels = clustering.labels_
        
        # Build transition graph
        G = nx.DiGraph()
        
        # Add nodes (clusters)
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise
        
        for label in unique_labels:
            cluster_points = coords[labels == label]
            center_x = np.mean(cluster_points[:, 0])
            center_y = np.mean(cluster_points[:, 1])
            size = len(cluster_points)
            
            G.add_node(label, pos=(center_x, center_y), size=size)
        
        # Add edges (transitions)
        transitions = Counter()
        for i in range(len(labels) - 1):
            if labels[i] != -1 and labels[i+1] != -1:
                if labels[i] != labels[i+1]:  # Only transitions between different clusters
                    transitions[(labels[i], labels[i+1])] += 1
        
        for (src, tgt), count in transitions.items():
            if count >= threshold:
                G.add_edge(src, tgt, weight=count)
        
        # Create plotly figure
        pos = nx.get_node_attributes(G, 'pos')
        
        # Edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight/2, color='rgba(125,125,125,0.5)'),
                hoverinfo='text',
                text=f'Transitions: {weight}',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Node trace
        node_x = []
        node_y = []
        node_size = []
        node_text = []
        
        for node in G.nodes(data=True):
            x, y = node[1]['pos']
            size = node[1]['size']
            node_x.append(x)
            node_y.append(y)
            node_size.append(size)
            node_text.append(f"AOI {node[0]}<br>Fixations: {size}")
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=[s/2 for s in node_size],
                color=node_size,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Fixations"),
                line=dict(width=2, color='white')
            ),
            text=[f"AOI {i}" for i in G.nodes()],
            textposition="top center",
            hoverinfo='text',
            hovertext=node_text,
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title="Gaze Network - AOI Relationships",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, self.screen_width]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[self.screen_height, 0]),
            plot_bgcolor='rgba(240,240,240,0.5)',
            height=600
        )
        
        return fig
    
    def create_4d_visualization(self) -> go.Figure:
        """
        Create 4D visualization: X, Y, Time, Duration.
        Uses color and size to represent 4th dimension.
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        x = self.data['x'].values
        y = self.data['y'].values
        
        if 'timestamp' in self.data.columns:
            time = self.data['timestamp'].values
            time_normalized = (time - time.min()) / (time.max() - time.min())
        else:
            time_normalized = np.linspace(0, 1, len(x))
        
        if 'duration' in self.data.columns:
            duration = self.data['duration'].values
            size = duration / 10
        else:
            size = [10] * len(x)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=time_normalized * 1000,  # Scale for visibility
            mode='markers',
            marker=dict(
                size=size,
                color=time_normalized,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time Progression"),
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=[f"Position: ({xi:.0f}, {yi:.0f})<br>Time: {ti:.0f}<br>Duration: {di:.0f}ms" 
                  for xi, yi, ti, di in zip(x, y, time_normalized*1000, 
                                            self.data.get('duration', [0]*len(x)))],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title="4D Gaze Visualization (X, Y, Time, Duration)",
            scene=dict(
                xaxis=dict(title='X Position (px)', range=[0, self.screen_width]),
                yaxis=dict(title='Y Position (px)', range=[0, self.screen_height]),
                zaxis=dict(title='Time Progression'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=700
        )
        
        return fig
    
    def create_animated_scan_path(self, fps: int = 10) -> go.Figure:
        """
        Create animated replay of scan path over time.
        
        Parameters:
        -----------
        fps : int
            Frames per second for animation
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        x = self.data['x'].values
        y = self.data['y'].values
        
        # Create frames
        frames = []
        n_points = len(x)
        step = max(1, n_points // (fps * 5))  # 5 seconds animation
        
        for i in range(0, n_points, step):
            end_idx = min(i + step, n_points)
            
            frame_data = [
                # Path trace
                go.Scatter(
                    x=x[:end_idx],
                    y=y[:end_idx],
                    mode='lines',
                    line=dict(color='rgba(100,100,255,0.5)', width=2),
                    showlegend=False
                ),
                # Current point
                go.Scatter(
                    x=[x[end_idx-1]],
                    y=[y[end_idx-1]],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='circle'),
                    showlegend=False
                )
            ]
            
            frames.append(go.Frame(data=frame_data, name=str(i)))
        
        # Initial frame
        fig = go.Figure(
            data=[
                go.Scatter(x=[x[0]], y=[y[0]], mode='lines'),
                go.Scatter(x=[x[0]], y=[y[0]], mode='markers', 
                          marker=dict(size=20, color='red'))
            ],
            frames=frames
        )
        
        fig.update_layout(
            title="Animated Scan Path Replay",
            xaxis=dict(range=[0, self.screen_width], title="X Position"),
            yaxis=dict(range=[self.screen_height, 0], title="Y Position"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, dict(frame=dict(duration=100, redraw=True),
                                        fromcurrent=True)]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                          mode="immediate")])
                ],
                x=0.1,
                y=1.15
            )],
            height=600
        )
        
        return fig
    
    def create_comparison_dashboard(self, data_list: List[pd.DataFrame], 
                                   labels: List[str]) -> go.Figure:
        """
        Create side-by-side comparison of multiple sessions.
        
        Parameters:
        -----------
        data_list : list of pandas.DataFrame
            List of eye-tracking datasets to compare
        labels : list of str
            Labels for each dataset
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        n_datasets = len(data_list)
        
        fig = make_subplots(
            rows=2, cols=n_datasets,
            subplot_titles=[f"{label} - Heatmap" for label in labels] + 
                          [f"{label} - Scan Path" for label in labels],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        for idx, (data, label) in enumerate(zip(data_list, labels)):
            x = data['x'].values
            y = data['y'].values
            
            # Heatmap
            heatmap, _, _ = np.histogram2d(
                x, y, bins=[40, 40],
                range=[[0, self.screen_width], [0, self.screen_height]]
            )
            
            fig.add_trace(
                go.Heatmap(z=heatmap.T, colorscale='Hot', showscale=(idx==0)),
                row=1, col=idx+1
            )
            
            # Scan path
            colors = np.linspace(0, 1, len(x))
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='lines+markers',
                    marker=dict(size=3, color=colors, colorscale='Viridis', showscale=False),
                    line=dict(width=1, color='rgba(100,100,100,0.3)'),
                    showlegend=False
                ),
                row=2, col=idx+1
            )
        
        fig.update_xaxes(range=[0, self.screen_width])
        fig.update_yaxes(range=[self.screen_height, 0])
        
        fig.update_layout(
            title="Multi-Session Comparison Dashboard",
            height=800
        )
        
        return fig
    
    def create_velocity_heatmap(self) -> go.Figure:
        """
        Create heatmap of gaze velocities across screen regions.
        Shows where users move their eyes quickly vs slowly.
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if 'timestamp' not in self.data.columns:
            raise ValueError("Timestamp data required for velocity calculation")
        
        x = self.data['x'].values
        y = self.data['y'].values
        t = self.data['timestamp'].values
        
        # Calculate velocities
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        dt[dt == 0] = 1
        velocities = np.sqrt(dx**2 + dy**2) / dt
        
        # Assign velocities to positions (use midpoint)
        x_mid = (x[:-1] + x[1:]) / 2
        y_mid = (y[:-1] + y[1:]) / 2
        
        # Create velocity map
        bins = 40
        velocity_map = np.zeros((bins, bins))
        count_map = np.zeros((bins, bins))
        
        x_bins = np.linspace(0, self.screen_width, bins + 1)
        y_bins = np.linspace(0, self.screen_height, bins + 1)
        
        for i in range(len(x_mid)):
            x_idx = min(bins - 1, int(x_mid[i] / self.screen_width * bins))
            y_idx = min(bins - 1, int(y_mid[i] / self.screen_height * bins))
            
            velocity_map[y_idx, x_idx] += velocities[i]
            count_map[y_idx, x_idx] += 1
        
        # Average velocities
        with np.errstate(divide='ignore', invalid='ignore'):
            velocity_map = np.where(count_map > 0, velocity_map / count_map, 0)
        
        fig = go.Figure(data=go.Heatmap(
            z=velocity_map,
            colorscale='RdYlGn_r',
            colorbar=dict(title="Velocity<br>(px/ms)"),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Velocity: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Gaze Velocity Heatmap",
            xaxis=dict(title="X Position", range=[0, bins]),
            yaxis=dict(title="Y Position", range=[bins, 0]),
            height=600
        )
        
        return fig
    
    def create_attention_timeline(self, window_size: int = 10) -> go.Figure:
        """
        Create timeline showing attention intensity over time.
        
        Parameters:
        -----------
        window_size : int
            Rolling window size for smoothing
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if 'timestamp' not in self.data.columns:
            raise ValueError("Timestamp data required")
        
        time = self.data['timestamp'].values
        
        if 'duration' in self.data.columns:
            durations = self.data['duration'].values
            # Smooth durations
            smoothed = pd.Series(durations).rolling(window=window_size, center=True).mean()
        else:
            smoothed = np.ones(len(time)) * 200
        
        fig = go.Figure()
        
        # Area plot
        fig.add_trace(go.Scatter(
            x=time,
            y=smoothed,
            fill='tozeroy',
            mode='lines',
            line=dict(color='rgba(0,100,255,0.7)', width=2),
            name='Attention Intensity'
        ))
        
        # Add threshold lines
        if 'duration' in self.data.columns:
            mean_dur = np.mean(durations)
            fig.add_hline(y=mean_dur, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_dur:.0f}ms")
        
        fig.update_layout(
            title="Attention Intensity Timeline",
            xaxis_title="Time (ms)",
            yaxis_title="Fixation Duration (ms)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def export_interactive_html(self, filepath: str, include_all: bool = True):
        """
        Export all visualizations to an interactive HTML report.
        
        Parameters:
        -----------
        filepath : str
            Path to save HTML file
        include_all : bool
            Include all visualizations
        """
        from plotly.subplots import make_subplots
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Eye-Tracking Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
                .plot {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Advanced Eye-Tracking Analysis Report</h1>
                <p>Generated with Advanced Visualizer</p>
        """
        
        # Add each visualization
        viz_functions = [
            ('Sankey Diagram', self.create_sankey_diagram),
            ('Network Graph', self.create_network_graph),
            ('4D Visualization', self.create_4d_visualization),
            ('Velocity Heatmap', self.create_velocity_heatmap),
            ('Attention Timeline', self.create_attention_timeline)
        ]
        
        for title, func in viz_functions:
            try:
                fig = func()
                html_content += f'<div class="plot"><h2>{title}</h2>'
                html_content += fig.to_html(full_html=False, include_plotlyjs=False)
                html_content += '</div>'
            except Exception as e:
                html_content += f'<div class="plot"><h2>{title}</h2><p>Error: {str(e)}</p></div>'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive report saved to: {filepath}")
