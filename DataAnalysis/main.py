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
dt = json.load(open('versionTimes.json'))
data = dt['VersionTimes']
df = pd.DataFrame(data)


baseline_stats = {}
memData = dt['MemoryStatsValues']
for key, value in memData.items():
    baseline_stats[key] = value[0]

previousEnd = None
for key, value in memData.items():
    baseline = baseline_stats[key]
    print(f"Baseline for {key}: {baseline}: {value[0]}")
    for i in range(len(value)):
        if previousEnd is None:
            value[i]['totalAlloc']    = (value[i]['totalAlloc']  - baseline['totalAlloc'])
            value[i]['heapInuse']     = (value[i]['heapInuse']   - baseline['heapInuse'])
            value[i]['heapObjects']   = (value[i]['heapObjects'] - baseline['heapObjects'])
            value[i]['numOfGC']       = (value[i]['numOfGC']     - baseline['numOfGC'])
        else:
            value[i]['totalAlloc']    = (value[i]['totalAlloc']  - baseline['totalAlloc']) - previousEnd['totalAlloc']
            value[i]['heapInuse']     = (value[i]['heapInuse']   - baseline['heapInuse']) - previousEnd['heapInuse']
            value[i]['heapObjects']   = (value[i]['heapObjects'] - baseline['heapObjects']) - previousEnd['heapObjects']
            value[i]['numOfGC']       = (value[i]['numOfGC']     - baseline['numOfGC']) - previousEnd['numOfGC']
    previousEnd = value[-1]

# remove the first element from each list
for key, value in memData.items():
    if len(value) > 0:
        value.pop(0)

# normalize each run to the percentage change from the start of the run
for key, value in memData.items():
    if len(value) > 0:
        start = value[0]['totalAlloc']
        for i in range(len(value)):
            value[i]['totalAlloc'] = (value[i]['totalAlloc'] - start) / start * 100
            value[i]['heapInuse'] = (value[i]['heapInuse'] - start) / start * 100
            value[i]['heapObjects'] = (value[i]['heapObjects'] - start) / start * 100
            value[i]['numOfGC'] = (value[i]['numOfGC'] - start) / start * 100
        
            
# Convert the data to a DataFrame
mem_df = pd.DataFrame(memData)
print(mem_df)

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

# After the existing bbox performance comparison code, add:

# Create a line graph for total allocation changes across versions
print("\nCreating memory allocation trend graphs...")

# Extract total allocation data from memData
mem_versions = list(memData.keys())
allocation_data = {}

for version in mem_versions:
    timestamps = [entry.get('timestamp', i) for i, entry in enumerate(memData[version])]
    allocation_values = [entry.get('totalAlloc', 0) for entry in memData[version]]
    allocation_data[version] = {'timestamps': timestamps, 'values': allocation_values}

# Create line graph for total allocation
allocation_fig = go.Figure()

# Add a line for each version
for version in mem_versions:
    allocation_fig.add_trace(
        go.Scatter(
            x=allocation_data[version]['timestamps'],
            y=allocation_data[version]['values'],
            mode='lines+markers',
            name=version,
            marker=dict(size=6),
            line=dict(width=2)
        )
    )

# Update layout
allocation_fig.update_layout(
    title='Memory Allocation Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Total Allocation (bytes)',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
allocation_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
allocation_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the allocation comparison figure
pio.write_image(allocation_fig, 'memory_allocation_over_time.png')

# Also create heap usage graph
heap_fig = go.Figure()

# Add a line for each version's heap usage
for version in mem_versions:
    heap_values = [entry.get('heapInuse', 0) for entry in memData[version]]
    heap_fig.add_trace(
        go.Scatter(
            x=allocation_data[version]['timestamps'],
            y=heap_values,
            mode='lines+markers',
            name=version,
            marker=dict(size=6),
            line=dict(width=2)
        )
    )

# Update layout
heap_fig.update_layout(
    title='Heap Usage Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Heap Usage (bytes)',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
heap_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
heap_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the heap usage comparison figure
pio.write_image(heap_fig, 'heap_usage_over_time.png')

print(f"Memory allocation graphs created and saved.")

# After the heap usage graph code, add:

# Create graphs for remaining memory metrics
print("\nCreating additional memory metric graphs...")

# HeapObjects graph
heap_objects_fig = go.Figure()

# Add a line for each version's heap objects
for version in mem_versions:
    heap_objects_values = [entry.get('heapObjects', 0) for entry in memData[version]]
    heap_objects_fig.add_trace(
        go.Scatter(
            x=allocation_data[version]['timestamps'],
            y=heap_objects_values,
            mode='lines+markers',
            name=version,
            marker=dict(size=6),
            line=dict(width=2)
        )
    )

# Update layout
heap_objects_fig.update_layout(
    title='Heap Objects Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Heap Objects (count)',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
heap_objects_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
heap_objects_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the heap objects comparison figure
pio.write_image(heap_objects_fig, 'heap_objects_over_time.png')

# NumOfGC graph
gc_fig = go.Figure()

# Add a line for each version's garbage collection count
for version in mem_versions:
    gc_values = [entry.get('numOfGC', 0) for entry in memData[version]]
    gc_fig.add_trace(
        go.Scatter(
            x=allocation_data[version]['timestamps'],
            y=gc_values,
            mode='lines+markers',
            name=version,
            marker=dict(size=6),
            line=dict(width=2)
        )
    )

# Update layout
gc_fig.update_layout(
    title='Garbage Collection Count Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Number of GC Events',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
gc_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
gc_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the GC count comparison figure
pio.write_image(gc_fig, 'gc_count_over_time.png')

# Create a combined view with all memory metrics in subplots
combined_fig = make_subplots(
    rows=4, 
    cols=1,
    subplot_titles=[
        'Total Allocation Over Time',
        'Heap Usage Over Time',
        'Heap Objects Over Time',
        'Garbage Collection Events Over Time'
    ],
    vertical_spacing=0.1
)

# Add all metrics to the combined figure
for version in mem_versions:
    # Total allocation
    combined_fig.add_trace(
        go.Scatter(
            x=allocation_data[version]['timestamps'],
            y=allocation_data[version]['values'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=2)
        ),
        row=1, col=1
    )
    
    # Heap usage
    combined_fig.add_trace(
        go.Scatter(
            x=allocation_data[version]['timestamps'],
            y=[entry.get('heapInuse', 0) for entry in memData[version]],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Heap objects
    combined_fig.add_trace(
        go.Scatter(
            x=allocation_data[version]['timestamps'],
            y=[entry.get('heapObjects', 0) for entry in memData[version]],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=2),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # GC count
    combined_fig.add_trace(
        go.Scatter(
            x=allocation_data[version]['timestamps'],
            y=[entry.get('numOfGC', 0) for entry in memData[version]],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=2),
            showlegend=False
        ),
        row=4, col=1
    )

# Update layout
combined_fig.update_layout(
    title='Memory Metrics Comparison Across Versions',
    title_font_size=20,
    height=1200,
    width=1200,
    legend=dict(
        groupclick="toggleitem",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Update y-axes titles
combined_fig.update_yaxes(title_text='Total Allocation (bytes)', row=1, col=1)
combined_fig.update_yaxes(title_text='Heap Usage (bytes)', row=2, col=1)
combined_fig.update_yaxes(title_text='Heap Objects (count)', row=3, col=1)
combined_fig.update_yaxes(title_text='GC Events', row=4, col=1)

# Add grid to all subplots
for i in range(1, 5):
    combined_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)
    combined_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)

# Save the combined figure
pio.write_image(combined_fig, 'all_memory_metrics_comparison.png')

print(f"All memory metric graphs created and saved.")

# After the memory metric data extraction, adjust the graphs to start from same baseline:

print("\nCreating memory allocation trend graphs with normalized starting points...")

# Extract and normalize memory metric data from memData
mem_versions = list(memData.keys())
normalized_data = {}

for version in mem_versions:
    # Get timestamps
    timestamps = [entry.get('timestamp', i) for i, entry in enumerate(memData[version])]
    
    # Get metrics - subtract first value to start from 0
    total_alloc_values = [entry.get('totalAlloc', 0) for entry in memData[version]]
    heap_inuse_values = [entry.get('heapInuse', 0) for entry in memData[version]]
    heap_objects_values = [entry.get('heapObjects', 0) for entry in memData[version]]
    gc_values = [entry.get('numOfGC', 0) for entry in memData[version]]
    
    # Store normalized data
    normalized_data[version] = {
        'timestamps': timestamps,
        'totalAlloc': total_alloc_values,
        'heapInuse': heap_inuse_values,
        'heapObjects': heap_objects_values,
        'numOfGC': gc_values
    }

# Create line graph for total allocation
allocation_fig = go.Figure()

# Add a line for each version
for version in mem_versions:
    allocation_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['totalAlloc'],
            mode='lines+markers',
            name=version,
            marker=dict(size=6),
            line=dict(width=2)
        )
    )

# Update layout
allocation_fig.update_layout(
    title='Memory Allocation Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Total Allocation (bytes)',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
allocation_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
allocation_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the allocation comparison figure
pio.write_image(allocation_fig, 'memory_allocation_over_time.png')

# Create heap usage graph
heap_fig = go.Figure()

# Add a line for each version's heap usage
for version in mem_versions:
    heap_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapInuse'],
            mode='lines+markers',
            name=version,
            marker=dict(size=6),
            line=dict(width=2)
        )
    )

# Update layout
heap_fig.update_layout(
    title='Heap Usage Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Heap Usage (bytes)',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
heap_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
heap_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the heap usage comparison figure
pio.write_image(heap_fig, 'heap_usage_over_time.png')

# HeapObjects graph
heap_objects_fig = go.Figure()

# Add a line for each version's heap objects
for version in mem_versions:
    heap_objects_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapObjects'],
            mode='lines+markers',
            name=version,
            marker=dict(size=6),
            line=dict(width=2)
        )
    )

# Update layout
heap_objects_fig.update_layout(
    title='Heap Objects Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Heap Objects (count)',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
heap_objects_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
heap_objects_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the heap objects comparison figure
pio.write_image(heap_objects_fig, 'heap_objects_over_time.png')

# NumOfGC graph
gc_fig = go.Figure()

# Add a line for each version's garbage collection count
for version in mem_versions:
    gc_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['numOfGC'],
            mode='lines+markers',
            name=version,
            marker=dict(size=6),
            line=dict(width=2)
        )
    )

# Update layout
gc_fig.update_layout(
    title='Garbage Collection Count Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Number of GC Events',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
gc_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
gc_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the GC count comparison figure
pio.write_image(gc_fig, 'gc_count_over_time.png')

# Create a combined view with all memory metrics in subplots
combined_fig = make_subplots(
    rows=4, 
    cols=1,
    subplot_titles=[
        'Total Allocation Over Time',
        'Heap Usage Over Time',
        'Heap Objects Over Time',
        'Garbage Collection Events Over Time'
    ],
    vertical_spacing=0.1
)

# Add all metrics to the combined figure
for version in mem_versions:
    # Total allocation
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['totalAlloc'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=2)
        ),
        row=1, col=1
    )
    
    # Heap usage
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapInuse'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Heap objects
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapObjects'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=2),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # GC count
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['numOfGC'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=2),
            showlegend=False
        ),
        row=4, col=1
    )

# Update layout
combined_fig.update_layout(
    title='Memory Metrics Comparison Across Versions',
    title_font_size=20,
    height=1200,
    width=1200,
    legend=dict(
        groupclick="toggleitem",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Update y-axes titles
combined_fig.update_yaxes(title_text='Total Allocation (bytes)', row=1, col=1)
combined_fig.update_yaxes(title_text='Heap Usage (bytes)', row=2, col=1)
combined_fig.update_yaxes(title_text='Heap Objects (count)', row=3, col=1)
combined_fig.update_yaxes(title_text='GC Events', row=4, col=1)

# Add grid to all subplots
for i in range(1, 5):
    combined_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)
    combined_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)

# Save the combined figure
pio.write_image(combined_fig, 'all_memory_metrics_comparison.png')

print(f"All memory metric graphs created and saved with aligned starting points.")

# Modify line thickness in all memory metric graphs

print("\nRecreating memory allocation trend graphs with thinner lines...")

# Create line graph for total allocation with thinner lines
allocation_fig = go.Figure()

# Add a line for each version with reduced thickness
for version in mem_versions:
    allocation_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['totalAlloc'],
            mode='lines+markers',
            name=version,
            marker=dict(size=5),  # Slightly smaller markers
            line=dict(width=1)    # Thinner lines
        )
    )

# Update layout
allocation_fig.update_layout(
    title='Memory Allocation Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Total Allocation (bytes)',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
allocation_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
allocation_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the allocation comparison figure
pio.write_image(allocation_fig, 'memory_allocation_over_time.png')

# Create heap usage graph with thinner lines
heap_fig = go.Figure()

# Add a line for each version's heap usage
for version in mem_versions:
    heap_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapInuse'],
            mode='lines+markers',
            name=version,
            marker=dict(size=5),  # Slightly smaller markers
            line=dict(width=1)    # Thinner lines
        )
    )

# Update layout
heap_fig.update_layout(
    title='Heap Usage Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Heap Usage (bytes)',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
heap_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
heap_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the heap usage comparison figure
pio.write_image(heap_fig, 'heap_usage_over_time.png')

# HeapObjects graph with thinner lines
heap_objects_fig = go.Figure()

# Add a line for each version's heap objects
for version in mem_versions:
    heap_objects_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapObjects'],
            mode='lines+markers',
            name=version,
            marker=dict(size=5),  # Slightly smaller markers
            line=dict(width=1)    # Thinner lines
        )
    )

# Update layout
heap_objects_fig.update_layout(
    title='Heap Objects Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Heap Objects (count)',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
heap_objects_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
heap_objects_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the heap objects comparison figure
pio.write_image(heap_objects_fig, 'heap_objects_over_time.png')

# NumOfGC graph with thinner lines
gc_fig = go.Figure()

# Add a line for each version's garbage collection count
for version in mem_versions:
    gc_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['numOfGC'],
            mode='lines+markers',
            name=version,
            marker=dict(size=5),  # Slightly smaller markers
            line=dict(width=1)    # Thinner lines
        )
    )

# Update layout
gc_fig.update_layout(
    title='Garbage Collection Count Over Time by Version',
    title_font_size=20,
    xaxis_title='Timestamp',
    yaxis_title='Number of GC Events',
    height=700,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Add grid
gc_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
gc_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the GC count comparison figure
pio.write_image(gc_fig, 'gc_count_over_time.png')

# Create a combined view with all memory metrics in subplots with thinner lines
combined_fig = make_subplots(
    rows=4, 
    cols=1,
    subplot_titles=[
        'Total Allocation Over Time',
        'Heap Usage Over Time',
        'Heap Objects Over Time',
        'Garbage Collection Events Over Time'
    ],
    vertical_spacing=0.1
)

# Add all metrics to the combined figure
for version in mem_versions:
    # Total allocation
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['totalAlloc'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),  # Smaller markers for combined plot
            line=dict(width=1)    # Thinner lines
        ),
        row=1, col=1
    )
    
    # Heap usage
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapInuse'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=1),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Heap objects
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapObjects'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=1),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # GC count
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['numOfGC'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4),
            line=dict(width=1),
            showlegend=False
        ),
        row=4, col=1
    )

# Update layout
combined_fig.update_layout(
    title='Memory Metrics Comparison Across Versions',
    title_font_size=20,
    height=1200,
    width=1200,
    legend=dict(
        groupclick="toggleitem",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.5)"
    )
)

# Update y-axes titles
combined_fig.update_yaxes(title_text='Total Allocation (bytes)', row=1, col=1)
combined_fig.update_yaxes(title_text='Heap Usage (bytes)', row=2, col=1)
combined_fig.update_yaxes(title_text='Heap Objects (count)', row=3, col=1)
combined_fig.update_yaxes(title_text='GC Events', row=4, col=1)

# Add grid to all subplots
for i in range(1, 5):
    combined_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)
    combined_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i, col=1)

# Save the combined figure
pio.write_image(combined_fig, 'all_memory_metrics_comparison.png')

print(f"All memory metric graphs recreated with thinner lines for better readability.")

# Improved memory metric graphs with better titles and formatting

print("\nCreating improved memory allocation trend graphs...")

import re

version_colors = {}
num_versions = len(mem_versions)
colors = []
colorscale = pc.sequential.Viridis

for i in range(num_versions):
    index = i / (num_versions - 1) if num_versions > 1 else 0.5
    rgb = pc.sample_colorscale(colorscale, [index])[0]
    # Convert from "rgb(r, g, b)" format to hex
    rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', rgb)
    if rgb_match:
        r, g, b = map(int, rgb_match.groups())
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        colors.append(hex_color)
    else:
        colors.append('#1f77b4')  # fallback color

for i, version in enumerate(mem_versions):
    version_colors[version] = colors[i]

# Function to format large numbers with appropriate unit suffixes
def format_bytes(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.1f} MB"
    else:
        return f"{size_bytes/1024**3:.1f} GB"

# Create line graph for total allocation with improved formatting
allocation_fig = go.Figure()

# Add a line for each version with consistent coloring
for version in mem_versions:
    max_alloc = max(normalized_data[version]['totalAlloc'])
    allocation_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['totalAlloc'],
            mode='lines+markers',
            name=f"{version} (max: {format_bytes(max_alloc)})",
            marker=dict(size=5, color=version_colors[version]),
            line=dict(width=1, color=version_colors[version])
        )
    )

# Update layout
allocation_fig.update_layout(
    title=dict(
        text='Memory Allocation Over Time by Version',
        font=dict(size=22)
    ),
    xaxis_title=dict(
        text='Time Sequence',
        font=dict(size=16)
    ),
    yaxis_title=dict(
        text='Total Memory Allocation (bytes)',
        font=dict(size=16)
    ),
    height=700,
    width=1200,
    legend=dict(
        title=dict(text='Version (with max allocation)', font=dict(size=12)),
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="lightgray",
        borderwidth=1
    ),
    plot_bgcolor='rgba(250, 250, 250, 0.9)',
    margin=dict(t=100)
)

# Add grid
allocation_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_font=dict(size=14))
allocation_fig.update_yaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='lightgray',
    title_font=dict(size=14),
    tickformat='.2s',  # Scientific notation for large numbers
)

# Add annotation explaining the chart
allocation_fig.add_annotation(
    text="This chart shows the cumulative memory allocation over time for each version",
    xref="paper", yref="paper",
    x=0.5, y=1.05,
    showarrow=False,
    font=dict(size=14, color="gray")
)

# Save the allocation comparison figure
pio.write_image(allocation_fig, 'memory_allocation_over_time.png')

# Create heap usage graph with improved formatting
heap_fig = go.Figure()

# Add a line for each version's heap usage
for version in mem_versions:
    max_heap = max(normalized_data[version]['heapInuse'])
    heap_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapInuse'],
            mode='lines+markers',
            name=f"{version} (max: {format_bytes(max_heap)})",
            marker=dict(size=5, color=version_colors[version]),
            line=dict(width=1, color=version_colors[version])
        )
    )

# Update layout
heap_fig.update_layout(
    title=dict(
        text='Heap Usage Over Time by Version',
        font=dict(size=22)
    ),
    xaxis_title=dict(
        text='Time Sequence',
        font=dict(size=16)
    ),
    yaxis_title=dict(
        text='Heap Memory in Use (bytes)',
        font=dict(size=16)
    ),
    height=700,
    width=1200,
    legend=dict(
        title=dict(text='Version (with max heap usage)', font=dict(size=12)),
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="lightgray",
        borderwidth=1
    ),
    plot_bgcolor='rgba(250, 250, 250, 0.9)',
    margin=dict(t=100)
)

# Add grid
heap_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_font=dict(size=14))
heap_fig.update_yaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='lightgray',
    title_font=dict(size=14),
    tickformat='.2s',  # Scientific notation for large numbers
)

# Add annotation explaining the chart
heap_fig.add_annotation(
    text="This chart shows the amount of heap memory in active use over time for each version",
    xref="paper", yref="paper",
    x=0.5, y=1.05,
    showarrow=False,
    font=dict(size=14, color="gray")
)

# Save the heap usage comparison figure
pio.write_image(heap_fig, 'heap_usage_over_time.png')

# HeapObjects graph with improved formatting
heap_objects_fig = go.Figure()

# Add a line for each version's heap objects
for version in mem_versions:
    max_objects = max(normalized_data[version]['heapObjects'])
    heap_objects_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapObjects'],
            mode='lines+markers',
            name=f"{version} (max: {max_objects:,} objects)",
            marker=dict(size=5, color=version_colors[version]),
            line=dict(width=1, color=version_colors[version])
        )
    )

# Update layout
heap_objects_fig.update_layout(
    title=dict(
        text='Heap Objects Count Over Time by Version',
        font=dict(size=22)
    ),
    xaxis_title=dict(
        text='Time Sequence',
        font=dict(size=16)
    ),
    yaxis_title=dict(
        text='Number of Objects in Heap',
        font=dict(size=16)
    ),
    height=700,
    width=1200,
    legend=dict(
        title=dict(text='Version (with max object count)', font=dict(size=12)),
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="lightgray",
        borderwidth=1
    ),
    plot_bgcolor='rgba(250, 250, 250, 0.9)',
    margin=dict(t=100)
)

# Add grid
heap_objects_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_font=dict(size=14))
heap_objects_fig.update_yaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='lightgray',
    title_font=dict(size=14),
    tickformat=',d',  # Format with commas for readability
)

# Add annotation explaining the chart
heap_objects_fig.add_annotation(
    text="This chart shows the number of objects allocated on the heap over time",
    xref="paper", yref="paper",
    x=0.5, y=1.05,
    showarrow=False,
    font=dict(size=14, color="gray")
)

# Save the heap objects comparison figure
pio.write_image(heap_objects_fig, 'heap_objects_over_time.png')

# NumOfGC graph with improved formatting
gc_fig = go.Figure()

# Add a line for each version's garbage collection count
for version in mem_versions:
    max_gc = max(normalized_data[version]['numOfGC']) if normalized_data[version]['numOfGC'] else 0
    gc_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['numOfGC'],
            mode='lines+markers',
            name=f"{version} (total: {int(max_gc)} GCs)",
            marker=dict(size=5, color=version_colors[version]),
            line=dict(width=1, color=version_colors[version])
        )
    )

# Update layout
gc_fig.update_layout(
    title=dict(
        text='Garbage Collection Activity Over Time by Version',
        font=dict(size=22)
    ),
    xaxis_title=dict(
        text='Time Sequence',
        font=dict(size=16)
    ),
    yaxis_title=dict(
        text='Cumulative Number of GC Events',
        font=dict(size=16)
    ),
    height=700,
    width=1200,
    legend=dict(
        title=dict(text='Version (with total GC count)', font=dict(size=12)),
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="lightgray", 
        borderwidth=1
    ),
    plot_bgcolor='rgba(250, 250, 250, 0.9)',
    margin=dict(t=100)
)

# Add grid
gc_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title_font=dict(size=14))
gc_fig.update_yaxes(
    showgrid=True, 
    gridwidth=1, 
    gridcolor='lightgray',
    title_font=dict(size=14),
    tickformat='d',  # Integer format
)

# Add annotation explaining the chart
gc_fig.add_annotation(
    text="This chart shows the cumulative number of garbage collection cycles over time",
    xref="paper", yref="paper",
    x=0.5, y=1.05,
    showarrow=False,
    font=dict(size=14, color="gray")
)

# Save the GC count comparison figure
pio.write_image(gc_fig, 'gc_count_over_time.png')

# Create a combined view with all memory metrics in subplots with improved formatting
combined_fig = make_subplots(
    rows=4, 
    cols=1,
    subplot_titles=[
        '<b>Total Memory Allocation Over Time</b>',
        '<b>Heap Memory Usage Over Time</b>',
        '<b>Heap Objects Count Over Time</b>',
        '<b>Garbage Collection Events Over Time</b>'
    ],
    vertical_spacing=0.12
)

# Add all metrics to the combined figure
for version in mem_versions:
    # Total allocation
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['totalAlloc'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4, color=version_colors[version]),
            line=dict(width=1, color=version_colors[version])
        ),
        row=1, col=1
    )
    
    # Heap usage
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapInuse'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4, color=version_colors[version]),
            line=dict(width=1, color=version_colors[version]),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Heap objects
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['heapObjects'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4, color=version_colors[version]),
            line=dict(width=1, color=version_colors[version]),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # GC count
    combined_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['numOfGC'],
            mode='lines+markers',
            name=version,
            legendgroup=version,
            marker=dict(size=4, color=version_colors[version]),
            line=dict(width=1, color=version_colors[version]),
            showlegend=False
        ),
        row=4, col=1
    )

# Update layout
combined_fig.update_layout(
    title=dict(
        text='Memory Performance Metrics Across Versions',
        font=dict(size=24)
    ),
    height=1400,  # Increased height for better readability
    width=1200,
    legend=dict(
        title=dict(text='Version', font=dict(size=12)),
        groupclick="toggleitem",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="lightgray",
        borderwidth=1
    ),
    plot_bgcolor='rgba(250, 250, 250, 0.9)',
)

# Update y-axes titles with improved formatting
combined_fig.update_yaxes(
    title_text='Memory Allocation (bytes)', 
    title_font=dict(size=14),
    tickformat='.2s',
    row=1, 
    col=1
)
combined_fig.update_yaxes(
    title_text='Heap Usage (bytes)', 
    title_font=dict(size=14),
    tickformat='.2s',
    row=2, 
    col=1
)
combined_fig.update_yaxes(
    title_text='Heap Objects (count)', 
    title_font=dict(size=14),
    tickformat=',d',
    row=3, 
    col=1
)
combined_fig.update_yaxes(
    title_text='GC Events (count)', 
    title_font=dict(size=14),
    tickformat='d',
    row=4, 
    col=1
)

# Add grid to all subplots
for i in range(1, 5):
    combined_fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        title_text='Time Sequence' if i == 4 else None,  # Only show x-axis title on bottom plot
        title_font=dict(size=14),
        row=i, 
        col=1
    )
    combined_fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        row=i, 
        col=1
    )

# Add overall annotation explaining the dashboard
combined_fig.add_annotation(
    text="This dashboard displays key memory metrics tracked during execution. Lower values generally indicate better optimization.",
    xref="paper", yref="paper",
    x=0.5, y=1.05,
    showarrow=False,
    font=dict(size=14, color="gray")
)

# Save the combined figure at higher resolution
pio.write_image(combined_fig, 'all_memory_metrics_comparison.png', scale=2)  # scale=2 doubles the resolution

print(f"All memory metric graphs recreated with improved formatting and clearer titles.")

# Fix the color scale issue and create improved memory allocation trend graphs

print("\nCreating improved memory allocation trend graphs...")

# Define a consistent color palette for versions using a more reliable method
version_colors = {}

# Use a colorscale from plotly.colors that we know exists
import plotly.colors as pc
colorscale = pc.sequential.Viridis
import re

# Generate enough colors for all versions
num_versions = len(mem_versions)
colors = []
for i in range(num_versions):
    index = i / (num_versions - 1) if num_versions > 1 else 0.5
    rgb = pc.sample_colorscale(colorscale, [index])[0]
    # Convert from rgb(r,g,b) format to hex
    rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', rgb)
    if rgb_match:
        r, g, b = map(int, rgb_match.groups())
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        colors.append(hex_color)
    else:
        colors.append(f'#1f77b4')  # Default color if conversion fails

for i, version in enumerate(mem_versions):
    version_colors[version] = colors[i]

# Function to format large numbers with appropriate unit suffixes
def format_bytes(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.1f} MB"
    else:
        return f"{size_bytes/1024**3:.1f} GB"

# Create line graph for total allocation with improved formatting
allocation_fig = go.Figure()

# Add a line for each version with consistent coloring
for version in mem_versions:
    max_alloc = max(normalized_data[version]['totalAlloc'])
    allocation_fig.add_trace(
        go.Scatter(
            x=normalized_data[version]['timestamps'],
            y=normalized_data[version]['totalAlloc'],
            mode='lines+markers',
            name=f"{version} (max: {format_bytes(max_alloc)})",
            marker=dict(size=5, color=version_colors[version]),
            line=dict(width=1, color=version_colors[version])
        )
    )

