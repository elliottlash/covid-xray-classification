from pathlib import Path
import cv2
import numpy as np
import pandas as pd

def average_local_metrics(gray_image, lung_mask, kernel_size=5, eps=10.0):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image_min = cv2.erode(gray_image, kernel, iterations=1).astype(np.float32)
    image_max = cv2.dilate(gray_image, kernel, iterations=1).astype(np.float32)

    local_contrast = (image_max - image_min) / (image_max + image_min + eps)
    average_contrast = 100.0 * np.mean(local_contrast[lung_mask])

    local_brightness = (image_max + image_min) / (2.0 * 255.0)
    average_brightness = 100.0 * np.mean(local_brightness[lung_mask])

    return average_contrast, average_brightness

def laplacian_variance(gray_image, lung_mask):
    blurred = cv2.GaussianBlur(gray_image, (5,5), 0)
    lap_map = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)
    return np.var(lap_map[lung_mask])

def metrics_per_class(image_paths, mask_paths):
    class_avg_contrast = []
    class_avg_brightness = []
    class_laplacian_variance = []
    erode_kernel = np.ones((5,5), np.uint8)

    for image_path, mask_path in zip(image_paths, mask_paths):
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask.shape != gray_image.shape:
            mask = cv2.resize(mask, (gray_image.shape[1], gray_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        lung_mask = cv2.erode(mask, erode_kernel, iterations=1) > 0
        if not np.any(lung_mask):
            continue

        avg_contr, avg_bright = average_local_metrics(gray_image, lung_mask)
        lapl_var = laplacian_variance(gray_image, lung_mask)

        class_avg_contrast.append((image_path, avg_contr))
        class_avg_brightness.append((image_path, avg_bright))
        class_laplacian_variance.append((image_path, lapl_var))

    return class_avg_contrast, class_avg_brightness, class_laplacian_variance

def extract_all_metrics(filepaths, maskpaths):
    avg_contr_list = []
    avg_bright_list = []
    lapl_var_list = []

    #Loop through each class and get the metrics for the class

    for fp, mp in zip(filepaths, maskpaths):
        class_avg_contr, class_avg_bright, class_lapl_var = metrics_per_class(fp, mp)
        avg_contr_list.append(class_avg_contr)
        avg_bright_list.append(class_avg_bright)
        lapl_var_list.append(class_lapl_var)

    return avg_contr_list, avg_bright_list, lapl_var_list

def metrics_to_dataframe(class_names, avg_contr_list, avg_bright_list, lapl_var_list):
    rows = []

    #Loop over each class
    for cls, contr, bright, lapl in zip(class_names, avg_contr_list, avg_bright_list, lapl_var_list):
        #Loop over each image in the class
        for (path, contr_val), (_, bright_val), (_, lapl_val) in zip(contr, bright, lapl):
            #Add the image to a row of a dataframe
            rows.append({
                "image_name": Path(path).name,
                "class": cls,
                "avg_contrast": float(contr_val),
                "avg_brightness": float(bright_val),
                "laplacian_variance": float(lapl_val)})
    return pd.DataFrame(rows)

