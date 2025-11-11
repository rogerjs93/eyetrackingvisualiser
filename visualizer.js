// Eye-Tracking Data Visualizer - JavaScript
let currentData = null;
let screenWidth = 1920;
let screenHeight = 1080;

// Update points value display
document.getElementById('points').addEventListener('input', (e) => {
    document.getElementById('pointsValue').textContent = e.target.value;
});

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// Generate synthetic eye-tracking data
function generateData() {
    const pattern = document.getElementById('pattern').value;
    const nPoints = parseInt(document.getElementById('points').value);
    const resolution = document.getElementById('resolution').value;
    
    [screenWidth, screenHeight] = resolution.split('x').map(Number);
    
    // Generate data based on pattern
    let data;
    switch(pattern) {
        case 'natural':
            data = generateNaturalPattern(nPoints);
            break;
        case 'reading':
            data = generateReadingPattern(nPoints);
            break;
        case 'centered':
            data = generateCenteredPattern(nPoints);
            break;
        case 'scattered':
            data = generateScatteredPattern(nPoints);
            break;
        case 'f_pattern':
            data = generateFPattern(nPoints);
            break;
        case 'z_pattern':
            data = generateZPattern(nPoints);
            break;
    }
    
    currentData = data;
    updateStatistics(data);
    createVisualizations(data);
}

function generateNaturalPattern(n) {
    const aois = [
        [screenWidth * 0.25, screenHeight * 0.3],
        [screenWidth * 0.75, screenHeight * 0.3],
        [screenWidth * 0.5, screenHeight * 0.5],
        [screenWidth * 0.35, screenHeight * 0.7],
        [screenWidth * 0.65, screenHeight * 0.7]
    ];
    
    const x = [], y = [], timestamp = [], duration = [];
    let time = 0;
    
    for (let i = 0; i < n; i++) {
        const aoiIdx = Math.floor(Math.random() * aois.length);
        const [aoiX, aoiY] = aois[aoiIdx];
        
        x.push(Math.max(0, Math.min(screenWidth, aoiX + gaussianRandom() * screenWidth * 0.08)));
        y.push(Math.max(0, Math.min(screenHeight, aoiY + gaussianRandom() * screenHeight * 0.08)));
        
        const dur = Math.max(50, Math.min(800, gammaRandom(2, 100)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    return {x, y, timestamp, duration};
}

function generateReadingPattern(n) {
    const x = [], y = [], timestamp = [], duration = [];
    let time = 0;
    let currentY = screenHeight * 0.2;
    const lineHeight = screenHeight * 0.05;
    const startX = screenWidth * 0.1;
    const endX = screenWidth * 0.9;
    
    for (let i = 0; i < n; i++) {
        const progress = (i % 20) / 20;
        x.push(startX + progress * (endX - startX) + gaussianRandom() * 20);
        y.push(currentY + gaussianRandom() * 10);
        
        if (i % 20 === 19) {
            currentY += lineHeight;
            if (currentY > screenHeight * 0.8) currentY = screenHeight * 0.2;
        }
        
        const dur = Math.max(80, Math.min(500, gammaRandom(2, 120)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    return {x, y, timestamp, duration};
}

function generateCenteredPattern(n) {
    const centerX = screenWidth / 2;
    const centerY = screenHeight / 2;
    const x = [], y = [], timestamp = [], duration = [];
    let time = 0;
    
    for (let i = 0; i < n; i++) {
        const radius = rayleighRandom() * Math.min(screenWidth, screenHeight) * 0.15;
        const angle = Math.random() * 2 * Math.PI;
        
        x.push(Math.max(0, Math.min(screenWidth, centerX + radius * Math.cos(angle))));
        y.push(Math.max(0, Math.min(screenHeight, centerY + radius * Math.sin(angle))));
        
        const dur = Math.max(100, Math.min(600, gammaRandom(2, 150)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    return {x, y, timestamp, duration};
}

function generateScatteredPattern(n) {
    const x = [], y = [], timestamp = [], duration = [];
    let time = 0;
    
    for (let i = 0; i < n; i++) {
        x.push(screenWidth * 0.1 + Math.random() * screenWidth * 0.8);
        y.push(screenHeight * 0.1 + Math.random() * screenHeight * 0.8);
        
        const dur = Math.max(50, Math.min(700, gammaRandom(2, 120)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    return {x, y, timestamp, duration};
}

function generateFPattern(n) {
    const x = [], y = [], timestamp = [], duration = [];
    let time = 0;
    
    const nTop = Math.floor(n / 3);
    const nMid = Math.floor(n / 4);
    const nVert = n - nTop - nMid;
    
    // Top bar
    for (let i = 0; i < nTop; i++) {
        x.push(screenWidth * 0.1 + (screenWidth * 0.8) * (i / nTop) + gaussianRandom() * 30);
        y.push(screenHeight * 0.2 + gaussianRandom() * 20);
        const dur = Math.max(60, Math.min(400, gammaRandom(2, 100)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    // Middle bar
    for (let i = 0; i < nMid; i++) {
        x.push(screenWidth * 0.1 + (screenWidth * 0.5) * (i / nMid) + gaussianRandom() * 30);
        y.push(screenHeight * 0.45 + gaussianRandom() * 20);
        const dur = Math.max(60, Math.min(400, gammaRandom(2, 100)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    // Vertical bar
    for (let i = 0; i < nVert; i++) {
        x.push(screenWidth * 0.1 + gaussianRandom() * 30);
        y.push(screenHeight * 0.2 + (screenHeight * 0.6) * (i / nVert) + gaussianRandom() * 30);
        const dur = Math.max(60, Math.min(400, gammaRandom(2, 100)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    return {x, y, timestamp, duration};
}

function generateZPattern(n) {
    const x = [], y = [], timestamp = [], duration = [];
    let time = 0;
    const nPerSegment = Math.floor(n / 3);
    
    // Top horizontal
    for (let i = 0; i < nPerSegment; i++) {
        x.push(screenWidth * 0.1 + (screenWidth * 0.8) * (i / nPerSegment) + gaussianRandom() * 30);
        y.push(screenHeight * 0.2 + gaussianRandom() * 20);
        const dur = Math.max(70, Math.min(450, gammaRandom(2, 110)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    // Diagonal
    for (let i = 0; i < nPerSegment; i++) {
        const progress = i / nPerSegment;
        x.push(screenWidth * 0.9 - (screenWidth * 0.8) * progress + gaussianRandom() * 30);
        y.push(screenHeight * 0.2 + (screenHeight * 0.6) * progress + gaussianRandom() * 30);
        const dur = Math.max(70, Math.min(450, gammaRandom(2, 110)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    // Bottom horizontal
    for (let i = 0; i < (n - 2 * nPerSegment); i++) {
        x.push(screenWidth * 0.1 + (screenWidth * 0.8) * (i / nPerSegment) + gaussianRandom() * 30);
        y.push(screenHeight * 0.8 + gaussianRandom() * 20);
        const dur = Math.max(70, Math.min(450, gammaRandom(2, 110)));
        duration.push(dur);
        timestamp.push(time);
        time += dur;
    }
    
    return {x, y, timestamp, duration};
}

// Statistical helper functions
function gaussianRandom() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function gammaRandom(shape, scale) {
    // Simple gamma approximation
    let sum = 0;
    for (let i = 0; i < shape; i++) {
        sum += -Math.log(Math.random());
    }
    return sum * scale / shape;
}

function rayleighRandom() {
    return Math.sqrt(-2 * Math.log(Math.random()));
}

// Update statistics display
function updateStatistics(data) {
    const stats = document.getElementById('stats');
    const {x, y, duration, timestamp} = data;
    
    const meanX = x.reduce((a, b) => a + b, 0) / x.length;
    const meanY = y.reduce((a, b) => a + b, 0) / y.length;
    const stdX = Math.sqrt(x.reduce((a, b) => a + Math.pow(b - meanX, 2), 0) / x.length);
    const stdY = Math.sqrt(y.reduce((a, b) => a + Math.pow(b - meanY, 2), 0) / y.length);
    const meanDur = duration.reduce((a, b) => a + b, 0) / duration.length;
    const totalTime = timestamp[timestamp.length - 1] - timestamp[0];
    
    stats.innerHTML = `
        <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3>📊 Total Points</h3>
            <div class="value">${x.length}</div>
        </div>
        <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>📍 X Position</h3>
            <div class="value">${meanX.toFixed(0)}px</div>
            <small>±${stdX.toFixed(0)}px</small>
        </div>
        <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>📍 Y Position</h3>
            <div class="value">${meanY.toFixed(0)}px</div>
            <small>±${stdY.toFixed(0)}px</small>
        </div>
        <div class="stat-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>⏱️ Avg Duration</h3>
            <div class="value">${meanDur.toFixed(0)}ms</div>
        </div>
        <div class="stat-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <h3>🕐 Total Time</h3>
            <div class="value">${(totalTime/1000).toFixed(1)}s</div>
        </div>
    `;
}

// Create all visualizations
function createVisualizations(data) {
    createHeatmap(data);
    createScanPath(data);
    createDurationHist(data);
    createSpatialDist(data);
    createDistributions(data);
    createTemporal(data);
}

function createHeatmap(data) {
    const {x, y} = data;
    
    // Create 2D histogram
    const bins = 50;
    const heatmap = Array(bins).fill().map(() => Array(bins).fill(0));
    
    x.forEach((xi, i) => {
        const binX = Math.min(bins - 1, Math.floor((xi / screenWidth) * bins));
        const binY = Math.min(bins - 1, Math.floor((y[i] / screenHeight) * bins));
        heatmap[binY][binX]++;
    });
    
    const trace = {
        z: heatmap,
        type: 'heatmap',
        colorscale: 'Hot',
        showscale: true
    };
    
    const layout = {
        title: 'Gaze Heatmap',
        xaxis: {title: 'X Position'},
        yaxis: {title: 'Y Position'},
        height: 500
    };
    
    Plotly.newPlot('plot-heatmap', [trace], layout);
    Plotly.newPlot('plot-heatmap-overview', [trace], {...layout, height: 400});
}

function createScanPath(data) {
    const {x, y} = data;
    const colors = x.map((_, i) => i / x.length);
    
    const trace = {
        x: x,
        y: y,
        mode: 'lines+markers',
        marker: {
            size: 5,
            color: colors,
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {title: 'Time'}
        },
        line: {width: 1, color: 'rgba(100,100,100,0.3)'}
    };
    
    const start = {
        x: [x[0]],
        y: [y[0]],
        mode: 'markers',
        marker: {size: 15, color: 'green', symbol: 'star'},
        name: 'Start'
    };
    
    const end = {
        x: [x[x.length-1]],
        y: [y[y.length-1]],
        mode: 'markers',
        marker: {size: 15, color: 'red', symbol: 'star'},
        name: 'End'
    };
    
    const layout = {
        title: 'Scan Path (Gaze Trajectory)',
        xaxis: {title: 'X Position', range: [0, screenWidth]},
        yaxis: {title: 'Y Position', range: [screenHeight, 0]},
        height: 500
    };
    
    Plotly.newPlot('plot-scanpath', [trace, start, end], layout);
    Plotly.newPlot('plot-scanpath-overview', [trace, start, end], {...layout, height: 400});
}

function createDurationHist(data) {
    const trace = {
        x: data.duration,
        type: 'histogram',
        nbinsx: 30,
        marker: {color: 'steelblue'}
    };
    
    const layout = {
        title: 'Fixation Duration Distribution',
        xaxis: {title: 'Duration (ms)'},
        yaxis: {title: 'Count'},
        height: 400
    };
    
    Plotly.newPlot('plot-duration', [trace], layout);
    Plotly.newPlot('plot-duration-overview', [trace], {...layout, height: 400});
}

function createSpatialDist(data) {
    const {x, y} = data;
    
    const trace = {
        x: x,
        y: y,
        mode: 'markers',
        marker: {size: 5, color: 'purple', opacity: 0.5}
    };
    
    const layout = {
        title: 'Spatial Distribution',
        xaxis: {title: 'X Position', range: [0, screenWidth]},
        yaxis: {title: 'Y Position', range: [screenHeight, 0]},
        height: 400
    };
    
    Plotly.newPlot('plot-spatial-overview', [trace], layout);
}

function createDistributions(data) {
    const {x, y, duration} = data;
    
    const xTrace = {
        x: x,
        type: 'histogram',
        nbinsx: 40,
        marker: {color: 'skyblue'}
    };
    
    const yTrace = {
        x: y,
        type: 'histogram',
        nbinsx: 40,
        marker: {color: 'lightcoral'}
    };
    
    const layout1 = {
        title: 'X Position Distribution',
        xaxis: {title: 'X Position'},
        yaxis: {title: 'Count'},
        height: 400
    };
    
    const layout2 = {
        title: 'Y Position Distribution',
        xaxis: {title: 'Y Position'},
        yaxis: {title: 'Count'},
        height: 400
    };
    
    // Density scatter
    const densityTrace = {
        x: x,
        y: y,
        mode: 'markers',
        marker: {
            size: 8,
            color: x.map((xi, i) => Math.sqrt(Math.pow(xi - screenWidth/2, 2) + Math.pow(y[i] - screenHeight/2, 2))),
            colorscale: 'Plasma',
            showscale: true
        }
    };
    
    const densityLayout = {
        title: 'Density Scatter',
        xaxis: {title: 'X Position'},
        yaxis: {title: 'Y Position'},
        height: 400
    };
    
    Plotly.newPlot('plot-x-dist', [xTrace], layout1);
    Plotly.newPlot('plot-y-dist', [yTrace], layout2);
    Plotly.newPlot('plot-density', [densityTrace], densityLayout);
}

function createTemporal(data) {
    const {x, y, timestamp} = data;
    
    const xTrace = {
        x: timestamp,
        y: x,
        mode: 'lines',
        name: 'X Position',
        line: {color: 'blue', width: 2}
    };
    
    const yTrace = {
        x: timestamp,
        y: y,
        mode: 'lines',
        name: 'Y Position',
        line: {color: 'red', width: 2},
        yaxis: 'y2'
    };
    
    const layout = {
        title: 'Position Over Time',
        xaxis: {title: 'Time (ms)'},
        yaxis: {title: 'X Position (px)', titlefont: {color: 'blue'}},
        yaxis2: {
            title: 'Y Position (px)',
            titlefont: {color: 'red'},
            overlaying: 'y',
            side: 'right'
        },
        height: 400
    };
    
    // Calculate velocity
    const dx = x.slice(1).map((xi, i) => xi - x[i]);
    const dy = y.slice(1).map((yi, i) => yi - y[i]);
    const dt = timestamp.slice(1).map((ti, i) => Math.max(1, ti - timestamp[i]));
    const velocity = dx.map((dxi, i) => Math.sqrt(dxi*dxi + dy[i]*dy[i]) / dt[i]);
    
    const velTrace = {
        x: timestamp.slice(1),
        y: velocity,
        mode: 'lines',
        fill: 'tozeroy',
        line: {color: 'green', width: 2}
    };
    
    const velLayout = {
        title: 'Movement Velocity',
        xaxis: {title: 'Time (ms)'},
        yaxis: {title: 'Velocity (px/ms)'},
        height: 400
    };
    
    Plotly.newPlot('plot-temporal', [xTrace, yTrace], layout);
    Plotly.newPlot('plot-velocity', [velTrace], velLayout);
}

// Generate initial data on load
window.addEventListener('load', () => {
    generateData();
});
