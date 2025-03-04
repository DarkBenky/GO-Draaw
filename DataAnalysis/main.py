import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json 
import plotly.io as pio  # Add this import at the top
import plotly.colors as pc

# Load the data
data = json.load(open('versionTimes.json'))
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
colors = pc.qualitative.Plotly[:len(stats_df.index)]  # Using Set3 palette, you can also try 'Set1', 'Paired', etc.

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
            marker_color=colors,  # Add this line to set colors
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
    colors = [f'rgb(200,{int(255*(1.5-x))},{int(255*(1.5-x))})' for x in normalized]
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