import matplotlib.pyplot as plt
import cv2
import seaborn as sns

def plot_image(img, save_path=None, title=None):
    """
    Plots an image using matplotlib.

    Parameters:
    img (numpy.ndarray): The image to be plotted.
    save_path (str, optional): Path to save the plotted image. If None, the image will not be saved.
    title (str, optional): Title for the plot.
    Returns:
    fig, ax: The matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(1, figsize=(7, 7), layout="constrained")
    ax.imshow(img, cmap='gray')
    ax.axis('off')  # Hide axes

    if title:
        ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
        print(f"Image saved to {save_path}")
    
    return fig, ax


def read_image(image_path):
    """
    Reads an image from the specified path using OpenCV.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    numpy.ndarray: The read image.
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img


def plot_histogram_classes(data, column, class_col, title=None, xlabel=None, save_path=None, grid=True):
    """
    Plots a histogram for the specified column in the data.

    Parameters:
    data (pandas.DataFrame): The data containing the column to plot.
    column (str): The name of the column to plot.
    class_col (str): The name of the column containing the class labels.
    title (str, optional): Title for the plot.
    xlabel (str, optional): Label for the x-axis.
    save_path (str, optional): Path to save the plotted histogram. If None, the histogram will not be saved.

    Returns:
    fig, ax: The matplotlib figure and axes objects.
    """
   
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=data, x=column, common_norm=False,
                hue=class_col, ax=ax)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=14)
    ax.set_ylabel("Density", fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=17)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    fig.savefig(save_path)

    return fig, ax


def plot_boxplots_classes(data, column, class_col, title=None, xlabel=None, ylabel=None, save_path=None):
    """
    Plots boxplots for the specified column in the data.

    Parameters:
    data (pandas.DataFrame): The data containing the column to plot.
    column (str): The name of the column to plot.
    class_col (str): The name of the column containing the class labels.
    title (str, optional): Title for the plot.
    xlabel (str, optional): Label for the x-axis.
    save_path (str, optional): Path to save the plotted boxplots. If None, the boxplots will not be saved.

    Returns:
    fig, ax: The matplotlib figure and axes objects.
    """

    fig, ax = plt.subplots(1, figsize=(10, 6))
    sns.boxplot(x=class_col, y=column, data=data, ax=ax)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=17)
    fig.savefig(save_path)
    return fig, ax