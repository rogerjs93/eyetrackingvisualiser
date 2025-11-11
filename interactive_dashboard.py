"""
Interactive Eye-Tracking Data Dashboard
A web-based GUI for visualizing and manipulating eye-tracking data in real-time.
Uses Plotly Dash for interactive visualizations.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sample_data_generator import generate_sample_eyetracking_data
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

# Import advanced analysis modules
from pattern_recognition import GazePatternRecognizer
from cognitive_load import CognitiveLoadAnalyzer
from advanced_visualizations import AdvancedVisualizer

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Eye-Tracking Data Visualizer"

# Global variables for data storage
current_data = None
screen_width = 1920
screen_height = 1080

# Define the layout
app.layout = html.Div([
    html.Div([
        html.H1("🎯 Eye-Tracking Data Visualizer", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Interactive dashboard for qualitative eye-tracking analysis",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
    ]),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Data Pattern:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='pattern-dropdown',
                options=[
                    {'label': '🔍 Natural Viewing', 'value': 'natural'},
                    {'label': '📖 Reading Pattern', 'value': 'reading'},
                    {'label': '🎯 Center-Focused', 'value': 'centered'},
                    {'label': '🎲 Scattered Random', 'value': 'scattered'},
                    {'label': '📱 F-Pattern (Web)', 'value': 'f_pattern'},
                    {'label': '⚡ Z-Pattern', 'value': 'z_pattern'}
                ],
                value='natural',
                style={'width': '100%'}
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Number of Points:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Slider(
                id='points-slider',
                min=100,
                max=1000,
                step=50,
                value=500,
                marks={100: '100', 250: '250', 500: '500', 750: '750', 1000: '1000'},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '25%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Screen Resolution:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='resolution-dropdown',
                options=[
                    {'label': '1920x1080 (Full HD)', 'value': '1920x1080'},
                    {'label': '2560x1440 (2K)', 'value': '2560x1440'},
                    {'label': '3840x2160 (4K)', 'value': '3840x2160'},
                    {'label': '1366x768 (Laptop)', 'value': '1366x768'}
                ],
                value='1920x1080',
                style={'width': '100%'}
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Button('🔄 Generate Data', id='generate-button', n_clicks=0,
                       style={
                           'backgroundColor': '#3498db',
                           'color': 'white',
                           'border': 'none',
                           'padding': '12px 30px',
                           'fontSize': '16px',
                           'cursor': 'pointer',
                           'borderRadius': '5px',
                           'marginTop': '25px'
                       }),
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Statistics Panel
    html.Div(id='stats-panel', style={'padding': '15px', 'backgroundColor': '#fff', 
                                       'borderRadius': '10px', 'marginBottom': '20px',
                                       'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Visualization Tabs
    dcc.Tabs(id='tabs', value='overview', children=[
        dcc.Tab(label='📊 Overview Dashboard', value='overview', 
                style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'color': '#3498db'}),
        dcc.Tab(label='🔥 Heatmap & Scan Path', value='heatmap',
                style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'color': '#3498db'}),
        dcc.Tab(label='📈 Distributions', value='distributions',
                style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'color': '#3498db'}),
        dcc.Tab(label='⏱️ Temporal Analysis', value='temporal',
                style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'color': '#3498db'}),
        dcc.Tab(label='🎯 Attention Zones', value='attention',
                style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'color': '#3498db'}),
        dcc.Tab(label='🤖 AI Pattern Recognition', value='ai_patterns',
                style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'color': '#e74c3c'}),
        dcc.Tab(label='🧠 Cognitive Load', value='cognitive',
                style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'color': '#9b59b6'}),
        dcc.Tab(label='🎨 Advanced Viz', value='advanced',
                style={'fontWeight': 'bold'}, selected_style={'fontWeight': 'bold', 'color': '#16a085'}),
    ], style={'marginBottom': '20px'}),
    
    # Content Area
    html.Div(id='tab-content', style={'padding': '20px'}),
    
    # Store component for data
    dcc.Store(id='data-store'),
    
], style={'maxWidth': '1800px', 'margin': '0 auto', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})


# Callback to generate data
@app.callback(
    [Output('data-store', 'data'),
     Output('stats-panel', 'children')],
    [Input('generate-button', 'n_clicks')],
    [State('pattern-dropdown', 'value'),
     State('points-slider', 'value'),
     State('resolution-dropdown', 'value')]
)
def generate_data(n_clicks, pattern, n_points, resolution):
    global current_data, screen_width, screen_height
    
    if n_clicks == 0:
        # Generate initial data
        pattern = 'natural'
        n_points = 500
        resolution = '1920x1080'
    
    # Parse resolution
    screen_width, screen_height = map(int, resolution.split('x'))
    
    # Generate data
    data = generate_sample_eyetracking_data(
        n_points=n_points,
        screen_width=screen_width,
        screen_height=screen_height,
        pattern=pattern,
        seed=42 + n_clicks
    )
    
    current_data = data
    
    # Create statistics panel
    stats = create_statistics_panel(data)
    
    return data.to_dict('records'), stats


def create_statistics_panel(data):
    """Create statistics display panel."""
    x = data['x'].values
    y = data['y'].values
    
    stats_style = {
        'display': 'inline-block',
        'padding': '15px',
        'margin': '10px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '8px',
        'minWidth': '150px',
        'textAlign': 'center'
    }
    
    stats = [
        html.Div([
            html.H4("📊 Total Points", style={'color': '#3498db', 'marginBottom': '5px'}),
            html.H2(str(len(data)), style={'color': '#2c3e50', 'margin': '0'})
        ], style=stats_style),
        
        html.Div([
            html.H4("📍 X Position", style={'color': '#e74c3c', 'marginBottom': '5px'}),
            html.P(f"Mean: {np.mean(x):.0f}px", style={'margin': '5px'}),
            html.P(f"Std: {np.std(x):.0f}px", style={'margin': '5px'})
        ], style=stats_style),
        
        html.Div([
            html.H4("📍 Y Position", style={'color': '#e74c3c', 'marginBottom': '5px'}),
            html.P(f"Mean: {np.mean(y):.0f}px", style={'margin': '5px'}),
            html.P(f"Std: {np.std(y):.0f}px", style={'margin': '5px'})
        ], style=stats_style),
    ]
    
    if 'duration' in data.columns:
        dur = data['duration'].values
        stats.append(
            html.Div([
                html.H4("⏱️ Duration", style={'color': '#9b59b6', 'marginBottom': '5px'}),
                html.P(f"Mean: {np.mean(dur):.0f}ms", style={'margin': '5px'}),
                html.P(f"Total: {np.sum(dur)/1000:.1f}s", style={'margin': '5px'})
            ], style=stats_style)
        )
    
    if 'timestamp' in data.columns:
        time = data['timestamp'].values
        stats.append(
            html.Div([
                html.H4("🕐 Time Span", style={'color': '#16a085', 'marginBottom': '5px'}),
                html.P(f"{(time[-1] - time[0])/1000:.1f}s", style={'margin': '5px', 'fontSize': '24px'})
            ], style=stats_style)
        )
    
    return html.Div(stats, style={'textAlign': 'center'})


# Callback to update visualizations
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'),
     Input('data-store', 'data')]
)
def update_tab_content(tab, data_dict):
    if data_dict is None:
        return html.Div("Click 'Generate Data' to start!", 
                       style={'textAlign': 'center', 'padding': '50px', 'fontSize': '20px'})
    
    data = pd.DataFrame(data_dict)
    
    if tab == 'overview':
        return create_overview_dashboard(data)
    elif tab == 'heatmap':
        return create_heatmap_view(data)
    elif tab == 'distributions':
        return create_distributions_view(data)
    elif tab == 'temporal':
        return create_temporal_view(data)
    elif tab == 'attention':
        return create_attention_view(data)
    elif tab == 'ai_patterns':
        return create_ai_patterns_view(data)
    elif tab == 'cognitive':
        return create_cognitive_load_view(data)
    elif tab == 'advanced':
        return create_advanced_viz_view(data)


def create_overview_dashboard(data):
    """Create overview dashboard with multiple plots."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Gaze Heatmap', 'Scan Path', 
                       'Fixation Duration', 'Spatial Distribution'),
        specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
               [{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    x = data['x'].values
    y = data['y'].values
    
    # 1. Heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[50, 50],
                                              range=[[0, screen_width], [0, screen_height]])
    heatmap = gaussian_filter(heatmap, sigma=2)
    
    fig.add_trace(
        go.Heatmap(z=heatmap.T, colorscale='Hot', showscale=False),
        row=1, col=1
    )
    
    # 2. Scan Path
    colors = np.linspace(0, 1, len(x))
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='lines+markers',
                  marker=dict(size=4, color=colors, colorscale='Viridis', showscale=False),
                  line=dict(width=1, color='rgba(100,100,100,0.3)'),
                  showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[x[0]], y=[y[0]], mode='markers',
                  marker=dict(size=15, color='green', symbol='star'),
                  name='Start', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[x[-1]], y=[y[-1]], mode='markers',
                  marker=dict(size=15, color='red', symbol='star'),
                  name='End', showlegend=False),
        row=1, col=2
    )
    
    # 3. Duration histogram
    if 'duration' in data.columns:
        fig.add_trace(
            go.Histogram(x=data['duration'], nbinsx=30, 
                        marker_color='steelblue', showlegend=False),
            row=2, col=1
        )
    
    # 4. Spatial distribution
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='markers',
                  marker=dict(size=5, color='purple', opacity=0.5),
                  showlegend=False),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="X Position", row=1, col=1)
    fig.update_yaxes(title_text="Y Position", row=1, col=1)
    fig.update_xaxes(title_text="X Position", range=[0, screen_width], row=1, col=2)
    fig.update_yaxes(title_text="Y Position", range=[screen_height, 0], row=1, col=2)
    fig.update_xaxes(title_text="Duration (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="X Position", range=[0, screen_width], row=2, col=2)
    fig.update_yaxes(title_text="Y Position", range=[screen_height, 0], row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False)
    
    return dcc.Graph(figure=fig)


def create_heatmap_view(data):
    """Create detailed heatmap and scan path view."""
    x = data['x'].values
    y = data['y'].values
    
    # Create heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[60, 60],
                                              range=[[0, screen_width], [0, screen_height]])
    heatmap = gaussian_filter(heatmap, sigma=2)
    
    fig1 = go.Figure(data=go.Heatmap(
        z=heatmap.T,
        colorscale='Hot',
        colorbar=dict(title="Density")
    ))
    
    fig1.update_layout(
        title="Gaze Heatmap",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        height=500
    )
    
    # Create scan path
    colors = np.linspace(0, 1, len(x))
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        marker=dict(size=6, color=colors, colorscale='Viridis', 
                   colorbar=dict(title="Time")),
        line=dict(width=2, color='rgba(100,100,100,0.3)'),
        name='Gaze Path'
    ))
    
    fig2.add_trace(go.Scatter(
        x=[x[0]], y=[y[0]],
        mode='markers',
        marker=dict(size=20, color='green', symbol='star'),
        name='Start'
    ))
    
    fig2.add_trace(go.Scatter(
        x=[x[-1]], y=[y[-1]],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        name='End'
    ))
    
    fig2.update_layout(
        title="Scan Path (Gaze Trajectory)",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        xaxis=dict(range=[0, screen_width]),
        yaxis=dict(range=[screen_height, 0]),
        height=500
    )
    
    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2)
    ])


def create_distributions_view(data):
    """Create distribution analysis view."""
    x = data['x'].values
    y = data['y'].values
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('X Position Distribution', 'Y Position Distribution',
                       'Fixation Duration Distribution', 'Density Scatter'),
        specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    # X distribution
    fig.add_trace(
        go.Histogram(x=x, nbinsx=40, marker_color='skyblue', name='X'),
        row=1, col=1
    )
    
    # Y distribution
    fig.add_trace(
        go.Histogram(x=y, nbinsx=40, marker_color='lightcoral', name='Y'),
        row=1, col=2
    )
    
    # Duration distribution
    if 'duration' in data.columns:
        dur = data['duration'].values
        fig.add_trace(
            go.Histogram(x=dur, nbinsx=40, marker_color='mediumpurple', name='Duration'),
            row=2, col=1
        )
    
    # Density scatter
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='markers',
                  marker=dict(size=8, color=z, colorscale='Plasma', 
                             showscale=True, colorbar=dict(x=1.15)),
                  name='Density'),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="X Position", row=1, col=1)
    fig.update_xaxes(title_text="Y Position", row=1, col=2)
    fig.update_xaxes(title_text="Duration (ms)", row=2, col=1)
    fig.update_xaxes(title_text="X Position", row=2, col=2)
    fig.update_yaxes(title_text="Y Position", row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False)
    
    return dcc.Graph(figure=fig)


def create_temporal_view(data):
    """Create temporal analysis view."""
    if 'timestamp' not in data.columns:
        return html.Div("No timestamp data available", 
                       style={'textAlign': 'center', 'padding': '50px'})
    
    time = data['timestamp'].values
    x = data['x'].values
    y = data['y'].values
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('X/Y Position Over Time', 'Movement Velocity'),
        shared_xaxes=True
    )
    
    # Position over time
    fig.add_trace(
        go.Scatter(x=time, y=x, mode='lines', name='X Position',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=time, y=y, mode='lines', name='Y Position',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Calculate velocity
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(time)
    dt[dt == 0] = 1  # Avoid division by zero
    velocity = np.sqrt(dx**2 + dy**2) / dt
    
    fig.add_trace(
        go.Scatter(x=time[1:], y=velocity, mode='lines',
                  name='Velocity', line=dict(color='green', width=2),
                  fill='tozeroy'),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Position (pixels)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (px/ms)", row=2, col=1)
    
    fig.update_layout(height=700)
    
    return dcc.Graph(figure=fig)


def create_attention_view(data):
    """Create attention zones view."""
    x = data['x'].values
    y = data['y'].values
    
    # Create smoothed heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[50, 50],
                                              range=[[0, screen_width], [0, screen_height]])
    heatmap_smooth = gaussian_filter(heatmap, sigma=2)
    
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=heatmap_smooth.T,
        colorscale='YlOrRd',
        opacity=0.7,
        colorbar=dict(title="Attention<br>Density")
    ))
    
    # Add scatter overlay
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.2),
        name='Fixations'
    ))
    
    # Add contours
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    fig.add_trace(go.Contour(
        z=heatmap_smooth.T,
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale='YlOrRd',
        showscale=False,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        line=dict(width=2),
        opacity=0.5
    ))
    
    fig.update_layout(
        title="Attention Zones with Density Contours",
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        height=700,
        showlegend=True
    )
    
    return dcc.Graph(figure=fig)


def create_ai_patterns_view(data):
    """Create AI Pattern Recognition view."""
    recognizer = GazePatternRecognizer(data)
    
    # Get all analyses
    reading = recognizer.detect_reading_behavior()
    expertise = recognizer.classify_expertise_level()
    aois = recognizer.detect_areas_of_interest()
    confusion = recognizer.detect_confusion_indicators()
    narrative = recognizer.get_narrative_insights()
    
    # Create layout
    return html.Div([
        html.H2("🤖 AI Pattern Recognition Analysis", style={'color': '#e74c3c', 'marginBottom': '20px'}),
        
        # Narrative Summary
        html.Div([
            html.H3("📝 Analysis Summary", style={'color': '#34495e'}),
            html.P(narrative, style={'fontSize': '16px', 'lineHeight': '1.8', 'padding': '15px',
                                     'backgroundColor': '#ecf0f1', 'borderRadius': '8px'})
        ], style={'marginBottom': '30px'}),
        
        # Metrics Grid
        html.Div([
            # Reading Behavior
            html.Div([
                html.H4("📖 Reading Behavior", style={'color': '#3498db', 'borderBottom': '2px solid #3498db'}),
                html.P(f"Behavior: {reading['behavior'].title()}", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                html.P(f"Confidence: {reading['confidence']:.1%}"),
                html.P(f"Left-to-right: {reading['metrics']['left_to_right_ratio']:.1%}"),
                html.P(f"Return sweeps: {reading['metrics']['return_sweeps']}")
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'margin': '10px', 'minWidth': '220px'}),
            
            # Expertise Level
            html.Div([
                html.H4("🎓 Expertise Level", style={'color': '#9b59b6', 'borderBottom': '2px solid #9b59b6'}),
                html.P(f"Level: {expertise['expertise'].title()}", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                html.P(f"Confidence: {expertise['confidence']:.1%}"),
                html.P(f"Path efficiency: {expertise['metrics']['path_efficiency']:.1%}"),
                html.P(f"Score: {expertise['score']:.2f}")
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'margin': '10px', 'minWidth': '220px'}),
            
            # Confusion Indicators
            html.Div([
                html.H4("⚠️ Confusion Level", style={'color': '#e74c3c', 'borderBottom': '2px solid #e74c3c'}),
                html.P(f"Level: {confusion['confusion_level'].title()}", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                html.P(f"Score: {confusion['confusion_score']:.1%}"),
                html.P(f"Revisit rate: {confusion['indicators']['revisit_rate']:.1%}"),
                html.P(f"Erraticism: {confusion['indicators']['movement_erraticism']:.3f}")
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'margin': '10px', 'minWidth': '220px'}),
            
            # Areas of Interest
            html.Div([
                html.H4("🎯 Areas of Interest", style={'color': '#16a085', 'borderBottom': '2px solid #16a085'}),
                html.P(f"AOIs Found: {aois['n_aois']}", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                html.P(f"Coverage: {aois['coverage']:.1%}"),
                html.P(f"Noise points: {aois['n_noise_points']}")
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'margin': '10px', 'minWidth': '220px'}),
            
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
        
        # Recommendation
        html.Div([
            html.H4("💡 Recommendation", style={'color': '#f39c12'}),
            html.P(confusion['recommendation'], style={'fontSize': '16px', 'padding': '15px',
                                                       'backgroundColor': '#fff3cd', 'borderRadius': '8px',
                                                       'border': '1px solid #f39c12'})
        ], style={'marginTop': '30px'})
    ])


def create_cognitive_load_view(data):
    """Create Cognitive Load Analysis view."""
    analyzer = CognitiveLoadAnalyzer(data)
    
    # Get all metrics
    entropy = analyzer.calculate_spatial_entropy()
    fixation = analyzer.calculate_fixation_rate()
    saccades = analyzer.calculate_saccade_metrics()
    attention = analyzer.calculate_ambient_focal_attention()
    transition = analyzer.calculate_gaze_transition_entropy()
    difficulty = analyzer.measure_task_difficulty()
    
    # Create visualizations
    timeline_fig = analyzer.create_attention_timeline()
    
    return html.Div([
        html.H2("🧠 Cognitive Load Analysis", style={'color': '#9b59b6', 'marginBottom': '20px'}),
        
        # Overall Difficulty Score
        html.Div([
            html.H3("Overall Task Difficulty", style={'textAlign': 'center', 'color': '#34495e'}),
            html.H1(f"{difficulty['overall_score']:.1f}/10", 
                   style={'textAlign': 'center', 'fontSize': '60px', 'color': '#e74c3c', 'margin': '20px'}),
            html.H4(f"Level: {difficulty['difficulty_level'].title()}", 
                   style={'textAlign': 'center', 'color': '#7f8c8d'}),
            html.P(difficulty['recommendation'], 
                  style={'textAlign': 'center', 'fontSize': '16px', 'padding': '15px',
                        'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'marginTop': '20px'})
        ], style={'padding': '30px', 'backgroundColor': '#fff', 'borderRadius': '15px',
                 'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'marginBottom': '30px'}),
        
        # Metrics Grid
        html.Div([
            # Spatial Entropy
            html.Div([
                html.H4("🌐 Spatial Entropy", style={'color': '#3498db'}),
                html.H2(f"{entropy['entropy']:.2f}", style={'color': '#2c3e50'}),
                html.P(entropy['interpretation'], style={'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'margin': '10px', 'minWidth': '200px'}),
            
            # Fixation Rate
            html.Div([
                html.H4("👁️ Fixation Rate", style={'color': '#e74c3c'}),
                html.H2(f"{fixation['fixations_per_second']:.1f}/s", style={'color': '#2c3e50'}),
                html.P(f"Mean: {fixation['mean_fixation_duration']:.0f}ms", style={'fontSize': '14px'}),
                html.P(fixation['interpretation'], style={'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'margin': '10px', 'minWidth': '200px'}),
            
            # Saccade Metrics
            html.Div([
                html.H4("⚡ Saccades", style={'color': '#f39c12'}),
                html.H2(f"{saccades['mean_saccade_length']:.0f}px", style={'color': '#2c3e50'}),
                html.P(f"Rate: {saccades['saccade_rate']:.1f}/s", style={'fontSize': '14px'}),
                html.P(saccades['interpretation'], style={'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'margin': '10px', 'minWidth': '200px'}),
            
            # Attention Mode
            html.Div([
                html.H4("🎯 Attention Mode", style={'color': '#9b59b6'}),
                html.H2(attention['dominant_mode'].title(), style={'color': '#2c3e50', 'fontSize': '24px'}),
                html.P(f"Ambient: {attention['ambient_ratio']:.1%}", style={'fontSize': '14px'}),
                html.P(f"Focal: {attention['focal_ratio']:.1%}", style={'fontSize': '14px'})
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'margin': '10px', 'minWidth': '200px'}),
            
            # Transition Entropy
            html.Div([
                html.H4("🔀 Predictability", style={'color': '#16a085'}),
                html.H2(f"{transition['entropy']:.2f}", style={'color': '#2c3e50'}),
                html.P(transition['interpretation'], style={'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '10px',
                     'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 'margin': '10px', 'minWidth': '200px'}),
            
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'marginBottom': '30px'}),
        
        # Attention Timeline
        dcc.Graph(figure=timeline_fig)
    ])


def create_advanced_viz_view(data):
    """Create Advanced Visualizations view."""
    viz = AdvancedVisualizer(data, screen_width, screen_height)
    
    # Create visualizations
    sankey_fig = viz.create_sankey_diagram(grid_size=4)
    network_fig = viz.create_network_graph(threshold=2)
    viz_4d_fig = viz.create_4d_visualization()
    velocity_fig = viz.create_velocity_heatmap()
    
    return html.Div([
        html.H2("🎨 Advanced Visualizations", style={'color': '#16a085', 'marginBottom': '20px'}),
        
        html.P("Unique, publication-ready visualizations for in-depth analysis", 
               style={'fontSize': '16px', 'color': '#7f8c8d', 'marginBottom': '30px'}),
        
        # Sankey Diagram
        html.Div([
            html.H3("🌊 Gaze Flow - Sankey Diagram", style={'color': '#3498db'}),
            html.P("Shows how attention flows between different screen regions", 
                   style={'color': '#7f8c8d', 'marginBottom': '15px'}),
            dcc.Graph(figure=sankey_fig)
        ], style={'marginBottom': '40px'}),
        
        # Network Graph
        html.Div([
            html.H3("🕸️ AOI Network Graph", style={'color': '#e74c3c'}),
            html.P("Network showing relationships and transitions between Areas of Interest", 
                   style={'color': '#7f8c8d', 'marginBottom': '15px'}),
            dcc.Graph(figure=network_fig)
        ], style={'marginBottom': '40px'}),
        
        # 4D Visualization
        html.Div([
            html.H3("🌌 4D Visualization", style={'color': '#9b59b6'}),
            html.P("Combines X, Y position, time progression, and fixation duration in a single plot", 
                   style={'color': '#7f8c8d', 'marginBottom': '15px'}),
            dcc.Graph(figure=viz_4d_fig)
        ], style={'marginBottom': '40px'}),
        
        # Velocity Heatmap
        html.Div([
            html.H3("⚡ Velocity Heatmap", style={'color': '#f39c12'}),
            html.P("Shows eye movement speed across different screen regions", 
                   style={'color': '#7f8c8d', 'marginBottom': '15px'}),
            dcc.Graph(figure=velocity_fig)
        ], style={'marginBottom': '40px'}),
    ])


if __name__ == '__main__':
    print("=" * 70)
    print("🎯 Eye-Tracking Data Visualizer - Interactive Dashboard")
    print("=" * 70)
    print("\n📡 Starting server...")
    print("🌐 Open your browser and go to: http://localhost:8050")
    print("\n✨ Features:")
    print("   • Generate synthetic eye-tracking data with different patterns")
    print("   • Interactive visualizations with zoom and pan")
    print("   • Real-time statistics")
    print("   • AI Pattern Recognition")
    print("   • Cognitive Load Analysis")
    print("   • Advanced Visualizations")
    print("   • Multiple visualization modes")
    print("⚠️  Press Ctrl+C to stop the server\n")
    print("=" * 70)
    
    app.run(debug=True, port=8050)
