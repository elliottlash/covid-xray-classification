from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.paths import get_dataset_paths
from src.metrics import extract_all_metrics, metrics_per_class, metrics_to_dataframe

if __name__ == "__main__":
    #Get Paths
    class_names, filepaths, maskpaths = get_dataset_paths()


    #Extract metrics for each class of image
    avg_contr_list, avg_bright_list, lapl_var_list = extract_all_metrics(filepaths, maskpaths)

    #Convert to dataframe
    df = metrics_to_dataframe(class_names, avg_contr_list, avg_bright_list, lapl_var_list)

    print(f"Extracted metrics for {len(df)} images.")
    print(df.head())

    #Save output
    output_csv = ROOT / "data" / "image_metrics.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")
