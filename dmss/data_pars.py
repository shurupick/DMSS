from os import listdir, path
import os

import pandas as pd

DATA_PATH = "./data/external/PolypGen2021_MultiCenterData_v3"


def parse_image_folder(folder_path: str) -> pd.DataFrame:
    """
    Parses the image folder and returns a DataFrame with image paths.

    Args:
        folder_path (str): Path to the folder containing images.
    Returns:
        pd.DataFrame with columns ['image_path', 'mask_path'].
    """
    # List all files in the directory
    image_paths_concat = []
    mask_paths_concat = []

    for fold in range(6):
        image_dir = path.join(folder_path, f"data_C{fold+1}", f"images_C{fold+1}")
        mask_dir = path.join(folder_path, f"data_C{fold+1}", f"masks_C{fold+1}")

        # List all files in the directory
        for image_path in listdir(image_dir):
            mask_path = image_path.replace(
                ".jpg", "_mask.jpg"
            )  # Assuming masks have the same name but .png extension
            if path.isfile(path.join(image_dir, image_path)) and path.isfile(
                path.join(mask_dir, mask_path)
            ):
                image_paths_concat.append(path.join(image_dir, image_path))
                mask_paths_concat.append(path.join(mask_dir, mask_path))

    return pd.DataFrame({"image_path": image_paths_concat, "mask_path": mask_paths_concat})


if __name__ == "__main__":
    print(os.getcwd())
    df_images = parse_image_folder(DATA_PATH)
    df_images.to_csv("./data/external/data.csv", index=False)
