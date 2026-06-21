import os
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
OUTPUT_IMG_DIR = "CH/resized"  # Folder where new images will be saved
TARGET_SIZE = (768, 768)  # The maximum width/height

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


def process_image(img_path, save_name):
    """
    Worker function to open, resize, and save a single image.
    Returns the new file path if successful, or None if it fails.
    """
    if not isinstance(img_path, str) or not os.path.exists(img_path):
        return None

    try:
        new_path = os.path.join(OUTPUT_IMG_DIR, save_name)

        # Skip if already processed (useful if script gets interrupted)
        if os.path.exists(new_path):
            return new_path

        with Image.open(img_path) as img:
            # Convert to RGB to drop alpha channels or handle grayscale safely
            img = img.convert("RGB")
            # thumbnail scales down the image in-place, maintaining aspect ratio
            img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
            # Save as JPEG for best size/quality ratio
            img.save(new_path, format="JPEG", quality=90)

        return new_path
    except Exception as e:
        print(f"\nError processing {img_path}: {e}")
        return None


def main(INPUT_CSV="test.csv", OUTPUT_CSV="test_resized.csv"):
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # We will store the new paths in these lists
    new_frontal_paths = [None] * len(df)
    new_lateral_paths = [None] * len(df)

    # Create a list of tasks for the multiprocessing pool
    tasks = []

    # Using ProcessPoolExecutor to max out CPU cores
    with ProcessPoolExecutor() as executor:
        for idx, row in df.iterrows():
            # Queue Frontal Image
            if pd.notna(row["Frontal"]):
                save_name = "frontal_" + row["Frontal"].split("/")[-1]
                future = executor.submit(process_image, row["Frontal"], save_name)
                tasks.append((future, idx, "Frontal"))

            # Queue Lateral Image
            if pd.notna(row["Lateral"]):
                save_name = "lateral_" + row["Lateral"].split("/")[-1]
                future = executor.submit(process_image, row["Lateral"], save_name)
                tasks.append((future, idx, "Lateral"))

        print(f"Processing {len(tasks)} images using all CPU cores...")

        # Process the progress bar as tasks complete
        for future, idx, col_type in tqdm(tasks, total=len(tasks)):
            new_path = future.result()
            if col_type == "Frontal":
                new_frontal_paths[idx] = new_path
            else:
                new_lateral_paths[idx] = new_path

    # Update DataFrame with the new paths
    df["Frontal"] = new_frontal_paths
    df["Lateral"] = new_lateral_paths

    # Save the updated dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Resized images saved to: {OUTPUT_IMG_DIR}/")
    print(f"Updated CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    # main("CH/test.csv",
    #     OUTPUT_CSV="CH/test_resized.csv")

    # main("CH/train_v2.csv",
    #     OUTPUT_CSV="CH/train_resized.csv")

    main(
        "CH/train.csv",
        OUTPUT_CSV="CH/train_resized_v0.csv",
    )
