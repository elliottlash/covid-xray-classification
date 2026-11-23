from pathlib import Path


def get_dataset_paths():
    '''
    This function outputs the names of the four classes of images in the dataset.
    It also outputs the paths to the image files (filepaths) and the paths to the masks (maskpaths) for each image.
    The path outputs are nested lists, i.e. filepaths is a list of four lists: one for each class of images.
    '''
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"

    dataset_dir = DATA_DIR / "COVID-19_Radiography_Dataset"
    if not dataset_dir.exists():
        dataset_dir = DATA_DIR

    class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    filepaths = []
    maskpaths = []

    for class_name in class_names:
        imgs = sorted((dataset_dir / class_name / "images").glob("*.png"))
        masks = [dataset_dir / class_name / "masks" / img.name for img in imgs]

        filepaths.append([str(path) for path in imgs])
        maskpaths.append([str(path) for path in masks])

    return class_names, filepaths, maskpaths

#The output_paths.py can be run standalone to verify count of images and masks.
if __name__ == "__main__":
    classes, image_files, masks = get_dataset_paths()
    for class_name, image, mask in zip(classes, image_files, masks):
        print(f"{class_name}: {len(image)} images, {len(mask)} masks")
