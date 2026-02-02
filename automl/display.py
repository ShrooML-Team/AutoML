"""
Module `display`
----------------------

Provides functions to display model evaluation results and compare
them against random and zero-baseline models. Results can be shown
as tables and plots, supporting classification, regression, and
multilabel classification tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def print_metrics_table(metrics: dict):
    """
    Display model and baseline metrics as a formatted ASCII table.
    """
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    col_widths = [
        max(df['Metric'].apply(len).max(), len('Metric')),
        max(df['Value'].apply(lambda x: len(f"{x:.4f}")).max(), len('Value'))
    ]
    sep_line = '+' + '-'*(col_widths[0]+2) + '+' + '-'*(col_widths[1]+2) + '+'

    print("\n=== Evaluation Metrics ===")
    print(sep_line)
    print(f"| {'Metric'.ljust(col_widths[0])} | {'Value'.ljust(col_widths[1])} |")
    print(sep_line)

    for _, row in df.iterrows():
        metric = row['Metric'].ljust(col_widths[0])
        value = f"{row['Value']:.4f}".rjust(col_widths[1])
        print(f"| {metric} | {value} |")
    print(sep_line)


def plot_metrics(metrics: dict):
    """
    Display model performance and baselines in 3D using the Viridis colormap.
    """
    metrics_data = []
    for metric, value in metrics.items():
        if metric.startswith("model_"):
            metrics_data.append((metric.replace("model_", ""), "Model", value))
        elif metric.startswith("random_"):
            metrics_data.append((metric.replace("random_", ""), "Random Baseline", value))
        elif metric.startswith("zero_"):
            metrics_data.append((metric.replace("zero_", ""), "Zero Baseline", value))
        else:
            metrics_data.append((metric, "Other", value))

    df_plot = pd.DataFrame(metrics_data, columns=["Metric", "Type", "Value"])
    df_plot["Value"] = np.nan_to_num(df_plot["Value"])

    metrics_unique = df_plot["Metric"].unique()
    types_unique = df_plot["Type"].unique()
    x_ticks = {m: i for i, m in enumerate(metrics_unique)}
    y_ticks = {t: i for i, t in enumerate(types_unique)}

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(df_plot["Value"].min(), df_plot["Value"].max())
    cmap = plt.colormaps["viridis"]

    for _, row in df_plot.iterrows():
        xi = float(x_ticks[row["Metric"]])
        yi = float(y_ticks[row["Type"]])
        zi = 0
        dx = dy = 0.5
        dz = row["Value"]
        color = cmap(norm(dz))
        ax.bar3d(xi, yi, zi, dx, dy, dz, color=color)
        ax.text(xi + dx/2, yi + dy/2, dz + 0.01, f"{dz:.3f}", ha='center', va='bottom', fontsize=9, color='black')

    ax.set_xticks(list(x_ticks.values()))
    ax.set_xticklabels(list(x_ticks.keys()), rotation=45, ha='right')
    ax.set_yticks(list(y_ticks.values()))
    ax.set_yticklabels(list(y_ticks.keys()))
    ax.set_zlabel("Score")
    ax.set_title("Model Performance vs Baselines (3D, Viridis)")

    fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.2)
    plt.show()
