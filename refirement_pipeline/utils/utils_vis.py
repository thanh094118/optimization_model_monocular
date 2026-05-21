#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 23:59:46 2024

@author: opencap
"""

# Plot keypoints

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import os
import plotly.graph_objects as go




def plot_objective_function(output_opt, save_path=None, show=False):
    # Extract the objective values
    objective_values = (
        output_opt["objective_values"].cpu().numpy()
    )  # Convert to NumPy array

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the objective values
    ax.plot(
        range(len(objective_values)),
        objective_values,
        label="Objective Function",
        linestyle="-",
        marker="",
    )

    # Customize the plot
    ax.set_title("Objective Function Plot")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.legend()
    ax.grid(True)

    # Save the plot if a save_path is provided
    if save_path is not None:
        fig_path = os.path.join(save_path, "objective_function_plot.png")
        plt.savefig(fig_path, dpi=300)
        # logger.info(f"Objective function plot saved at {fig_path}")

    # Show the plot if required
    if show:
        plt.show()

    # Close the figure to free memory if not showing
    if not show:
        plt.close(fig)


def plot_2d_keypoints_interactive_plotly(
    keypoints,
    image_w=720,
    image_h=1280,
    save_path=None,
    range_mono=None,
    fig_show=False,
):
    # Predefined color palette
    color_palette = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "yellow",
        "brown",
        "pink",
        "gray",
        "olive",
        "lime",
    ]

    # Assign colors to key_labels
    key_label_colors = {
        key_label: color_palette[i % len(color_palette)]
        for i, key_label in enumerate(keypoints.keys())
    }

    # Slice keypoints if range_mono is specified
    if range_mono is not None:
        start, end = range_mono[0], range_mono[-1] + 1  # Include the last frame
        keypoints = {
            key_label: points[start:end]  # Slice along the first dimension (time)
            for key_label, points in keypoints.items()
        }

    # Determine the maximum number of frames across all datasets
    T = max(points.shape[0] for points in keypoints.values())

    # Create the figure and frames
    frames = []

    for t in range(T):
        frame_data = []
        for key_label, points in keypoints.items():
            if t < points.shape[0]:  # Ensure frame exists after slicing
                frame_data.append(
                    go.Scatter(
                        x=points[t, :, 0],  # X-coordinate
                        y=points[t, :, 1],  # Y-coordinate
                        mode="markers",
                        marker=dict(size=8, color=key_label_colors[key_label]),
                        name=key_label,
                        text=[
                            f"Index: {i}<br>Coords: ({x:.2f}, {y:.2f})<br>Source: {key_label}"
                            for i, (x, y) in enumerate(points[t, :, :2])
                        ],
                        hoverinfo="text",
                    )
                )
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Add initial data for frame 0
    initial_data = []
    for key_label, points in keypoints.items():
        if points.shape[0] > 0:  # Ensure at least one frame exists
            initial_data.append(
                go.Scatter(
                    x=points[0, :, 0],
                    y=points[0, :, 1],
                    mode="markers",
                    marker=dict(size=8, color=key_label_colors[key_label]),
                    name=key_label,
                    text=[
                        f"Index: {i}<br>Coords: ({x:.2f}, {y:.2f})<br>Source: {key_label}"
                        for i, (x, y) in enumerate(points[0, :, :2])
                    ],
                    hoverinfo="text",
                )
            )

    # Add data and frames to the figure
    fig = go.Figure(data=initial_data, frames=frames)

    # Add slider and playback controls
    fig.update_layout(
        xaxis=dict(range=[0, image_w], title="X"),
        yaxis=dict(range=[image_h, 0], title="Y"),  # Flip y-axis
        title="2D Keypoints with Frame Slider",
        width=800,
        height=600,
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(t)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(t),
                        "method": "animate",
                    }
                    for t in range(T)
                ],
                "currentvalue": {"prefix": "Frame: ", "font": {"size": 16}},
                "pad": {"t": 50},
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                ],
            }
        ],
    )

    # Save the interactive plot as HTML
    if save_path:
        html_path = f"{save_path}/2d_keypoints_plot.html"
        fig.write_html(html_path)
        # print(f"Interactive plot saved to {html_path}")

    if fig_show:
        fig.show()

    return html_path if save_path else None
