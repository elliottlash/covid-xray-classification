from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from covidxray.plotting import plot_histogram_classes, plot_boxplots_classes

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

metrics_file = ROOT / "data" / "image_metrics.csv"

# check if the metrics file exists
if not metrics_file.exists():
    print(f"Metrics file not found at {metrics_file}. Please run extract_metrics.py first.")
    sys.exit(1)
else:
    print(f"Metrics file found at {metrics_file}. Proceeding with analysis.")
    metrics_df = pd.read_csv(metrics_file)
    print(metrics_df.head())


plot_histogram_classes(metrics_df, column='avg_contrast', class_col='class', 
                       title="Average Contrast Distribution by class", 
                       xlabel="Average Contrast", 
                       save_path=ROOT / "results" / "avg_contrast_histogram.png")
plot_histogram_classes(metrics_df, column='avg_brightness', class_col='class', 
                       title="Average Brightness Distribution by class", 
                       xlabel="Average Brightness", 
                       save_path=ROOT / "results" / "avg_brightness_histogram.png")
plot_histogram_classes(metrics_df, column='laplacian_variance', class_col='class', 
                       title="Laplacian Variance Distribution by class", xlabel="Laplacian Variance", 
                       save_path=ROOT / "results" / "laplacian_variance_histogram.png")

plot_boxplots_classes(metrics_df, column='avg_contrast', class_col='class',
                    title="Average Contrast Boxplot by Class",
                    xlabel="class", ylabel="Average Contrast",
                    save_path=ROOT / "results" / "avg_contrast_boxplot.png")

plot_boxplots_classes(metrics_df, column='avg_brightness', class_col='class',
                    title="Average Brightness Boxplot by class",
                    xlabel="class", ylabel="Average Brightness",
                    save_path=ROOT / "results" / "avg_brightness_boxplot.png")

plot_boxplots_classes(metrics_df, column='laplacian_variance', class_col='class',
                    title="Laplacian Variance Boxplot by class",
                    xlabel="class", ylabel="Laplacian Variance",
                    save_path=ROOT / "results" / "laplacian_variance_boxplot.png")
