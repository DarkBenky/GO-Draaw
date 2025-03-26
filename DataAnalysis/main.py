import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json 
import plotly.io as pio  # Add this import at the top
import plotly.colors as pc
import colorlover as cl  # Make sure this is installed

# Load the data
data = json.load(open('versionTimes.json'))
data = data['VersionTimes']
df = pd.DataFrame(data)

# Calculate statistics for each version
metrics = ['Mean Frame Time', 'Std Frame Time', 'Min Frame Time', 'Bottom Frame Time 10%', 'Top Frame Time 10%', 'Max Frame Time', 'Median Frame Time']  # Added missing comma
versions = {}
for column in df.columns:
    versions[column] = {
        'Mean Frame Time': df[column].mean(),
        'Std Frame Time': df[column].std(),
        'Min Frame Time': df[column].min(),
        'Bottom Frame Time 10%': df[column].quantile(0.1),
        'Top Frame Time 10%': df[column].quantile(0.9),
        'Max Frame Time': df[column].max(),
        'Median Frame Time': df[column].median()  # Added missing comma
    }

# Convert versions dictionary to DataFrame for easier plotting
stats_df = pd.DataFrame(versions).T


# Create a color palette for versions
num_versions = len(stats_df.index)

# Generate enough colors for all versions
if num_versions <= 10:
    # Use a qualitative color scale with enough colors
    colors = cl.scales['10']['qual']['Paired'][:num_versions]
elif num_versions <= 20:
    # For more versions, generate more colors
    colors = cl.scales['10']['qual']['Paired'] + cl.scales['10']['qual']['Set3'][:num_versions-10]
else:
    # For even more versions, create a custom color scale
    colorscale = cl.scales['10']['div']['Spectral']
    colors = cl.interp(colorscale, num_versions)

# Create and save individual plots for each metric
for metric in metrics:
    # Create individual figure for each metric
    single_fig = go.Figure(
        go.Bar(
            name=metric,
            x=stats_df.index,
            y=stats_df[metric],
            text=stats_df[metric].round(4),
            textposition='auto',
            marker_color=colors  # Add this line to set colors
        )
    )
    
    # Update layout for individual plot
    single_fig.update_layout(
        title=f'{metric} Comparison Across Versions',
        xaxis_title='Versions',
        yaxis_title='Time (microseconds)',
        height=600,
        width=1000,
        showlegend=False
    )
    
    # Save the figure as PNG
    pio.write_image(single_fig, f'performance_{metric.lower().replace(" ", "_")}.png')

# Create subplot figure with separate graph for each metric
fig = make_subplots(rows=len(metrics), cols=1,
                    subplot_titles=metrics,
                    vertical_spacing=0.1)

# Add bars for each metric in separate subplots
for i, metric in enumerate(metrics, 1):
    fig.add_trace(
        go.Bar(
            name=metric,
            x=stats_df.index,
            y=stats_df[metric],
            text=stats_df[metric].round(4),
            textposition='auto',
            marker_color=colors,  # Now colors has enough entries
            showlegend=False
        ),
        row=i, col=1
    )

# Update layout
fig.update_layout(
    title='Performance Metrics Comparison',
    height=1800,  # Increased height to accommodate all subplots
    width=1000,
    showlegend=False
)

# Update y-axes titles
for i in range(len(metrics)):
    fig.update_yaxes(title_text='Time (microseconds)', row=i+1, col=1)

# Update x-axes titles
for i in range(len(metrics)):
    fig.update_xaxes(title_text='Versions', row=i+1, col=1)

# Show the plot
fig.show()

# Create color scale for cells
def create_color_scale(values):
    normalized = (values - values.min()) / (values.max() - values.min())
    colors = [f'rgb({int(255*(x+0.3))},{int(255*(x+0.4))},{int(255*(x+0.2))})' for x in normalized]
    return colors

# Create color-coded cells for each metric
cell_colors = []
for metric in metrics:
    cell_colors.append(create_color_scale(stats_df[metric]))

# Create a table figure with color coding
table_fig = go.Figure(data=[go.Table(
    header=dict(
        values=['Version'] + metrics,
        fill_color='rgb(200, 220, 230)',
        align='left',
        font=dict(size=12, color='black')
    ),
    cells=dict(
        values=[stats_df.index] + [stats_df[metric].round(4) for metric in metrics],
        fill_color=[['white'] * len(stats_df.index)] + cell_colors,  # Color coding for each metric
        align='left',
        font=dict(size=11, color='black')
    )
)])

# Update table layout
table_fig.update_layout(
    title='Performance Metrics Table (Green=Better, Red=Worse)',
    width=1500,
    height=500,
)

# Save the table as PNG
pio.write_image(table_fig, 'performance_metrics_table.png')

# Print the statistics table
print("\nDetailed Statistics:")
print(stats_df.round(4))

# BVH performance comparison data
bvh_data = {
    'Implementation': ['Classic BVH', 'Lean BVH'],
    'Execution Time (ms)': [407.888788, 334.069267],
    'Percentage': [100, (334.069267/407.888788)*100]  # Calculate percentage relative to Classic BVH
}

bvh_df = pd.DataFrame(bvh_data)

# Calculate improvement percentage
improvement = ((407.888788 - 334.069267) / 407.888788) * 100

# Create bar chart for BVH comparison
bvh_fig = go.Figure([
    go.Bar(
        x=bvh_df['Implementation'],
        y=bvh_df['Execution Time (ms)'],
        text=[f"{time:.2f} ms" for time in bvh_df['Execution Time (ms)']],
        textposition='auto',
        marker_color=['#1f77b4', '#2ca02c'],  # Blue for Classic, Green for Lean
        width=0.6
    )
])

# Update layout with annotations
bvh_fig.update_layout(
    title='BVH Implementation Performance Comparison',
    title_font_size=20,
    xaxis_title='BVH Implementation',
    yaxis_title='Execution Time Per 1 000 000 Samples (ms)',
    height=600,
    width=900,
    bargap=0.2,
    annotations=[
        dict(
            x=1,  # Position above the Lean BVH bar
            y=bvh_df['Execution Time (ms)'][1] + 20,  # Slightly above the bar
            xref="x",
            yref="y",
            text=f"Performance improvement: {improvement:.2f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#2ca02c",
            ax=0,
            ay=-40
        )
    ]
)

# Add a horizontal line showing the Classic BVH time for reference
bvh_fig.add_shape(
    type="line",
    x0=-0.4,
    y0=bvh_df['Execution Time (ms)'][0],
    x1=1.4,
    y1=bvh_df['Execution Time (ms)'][0],
    line=dict(
        color="red",
        width=2,
        dash="dash",
    )
)

# Save the BVH comparison figure
pio.write_image(bvh_fig, 'bvh_performance_comparison.png')

# Print the BVH performance improvement
print(f"\nBVH Performance Improvement:")
print(f"Classic BVH: {bvh_df['Execution Time (ms)'][0]:.2f} ms")
print(f"Lean BVH: {bvh_df['Execution Time (ms)'][1]:.2f} ms")
print(f"Improvement: {improvement:.2f}%")

# Bounding Box collision functions performance comparison data
bbox_data = {
    'Implementation': ['BoundingBoxCollisionVector', 'BoundingBoxCollisionPair'],
    'Execution Time (ms)': [291.248548, 215.934921],
    'Percentage': [100, (215.934921/291.248548)*100]  # Calculate percentage relative to BoundingBoxCollisionVector
}

bbox_df = pd.DataFrame(bbox_data)

# Calculate improvement percentage
bbox_improvement = ((291.248548 - 215.934921) / 291.248548) * 100

# Create bar chart for Bounding Box collision functions comparison
bbox_fig = go.Figure([
    go.Bar(
        x=bbox_df['Implementation'],
        y=bbox_df['Execution Time (ms)'],
        text=[f"{time:.2f} ms" for time in bbox_df['Execution Time (ms)']],
        textposition='auto',
        marker_color=['#ff7f0e', '#9467bd'],  # Different colors from BVH chart
        width=0.6
    )
])

# Update layout with annotations
bbox_fig.update_layout(
    title='Bounding Box Collision Functions Performance Comparison',
    title_font_size=20,
    xaxis_title='Collision Function Implementation',
    yaxis_title='Execution Time Per 1 000 000 Samples (ms)',
    height=600,
    width=900,
    bargap=0.2,
    annotations=[
        dict(
            x=1,  # Position above the BoundingBoxCollisionPair bar
            y=bbox_df['Execution Time (ms)'][1] + 20,  # Slightly above the bar
            xref="x",
            yref="y",
            text=f"Performance improvement: {bbox_improvement:.2f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#9467bd",
            ax=0,
            ay=-40
        )
    ]
)

# Add a horizontal line showing the BoundingBoxCollisionVector time for reference
bbox_fig.add_shape(
    type="line",
    x0=-0.4,
    y0=bbox_df['Execution Time (ms)'][0],
    x1=1.4,
    y1=bbox_df['Execution Time (ms)'][0],
    line=dict(
        color="red",
        width=2,
        dash="dash",
    )
)

# Save the Bounding Box comparison figure
pio.write_image(bbox_fig, 'bbox_performance_comparison.png')

# Print the Bounding Box collision functions performance improvement
print(f"\nBounding Box Collision Functions Performance Improvement:")
print(f"BoundingBoxCollisionVector: {bbox_df['Execution Time (ms)'][0]:.2f} ms")
print(f"BoundingBoxCollisionPair: {bbox_df['Execution Time (ms)'][1]:.2f} ms")
print(f"Improvement: {bbox_improvement:.2f}%")