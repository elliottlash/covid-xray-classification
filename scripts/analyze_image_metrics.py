from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from covidxray.plotting import plot_histogram_classes

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

metrics_df = metrics_df[metrics_df['class'] != "Lung_Opacity"]


plot_histogram_classes(metrics_df, column='avg_contrast', class_col='class', 
                       title="Average Contrast Distribution by Class", 
                       xlabel="Average Contrast", 
                       save_path=ROOT / "results" / "avg_contrast_histogram.png")
plot_histogram_classes(metrics_df, column='avg_brightness', class_col='class', 
                       title="Average Brightness Distribution by Class", 
                       xlabel="Average Brightness", 
                       save_path=ROOT / "results" / "avg_brightness_histogram.png")
plot_histogram_classes(metrics_df, column='laplacian_variance', class_col='class', 
                       title="Laplacian Variance Distribution by Class", xlabel="Laplacian Variance", 
                       save_path=ROOT / "results" / "laplacian_variance_histogram.png")