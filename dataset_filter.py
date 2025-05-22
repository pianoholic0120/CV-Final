import os
import sys
import re
import argparse
import pandas as pd
from PIL import Image

def extract_grid_dimensions(filename):
    """
    Extracts the grid dimensions from the CSV file name, such as "2_2" or "4_4",
    and returns a tuple (num_rows, num_cols). If unable to parse, default to (1, 1).
    Note: Here we assume that the first number in the file name represents the number of horizontal divisions,
    and the second number represents the number of vertical divisions,
    but in reality, the splitting is determined based on the Row and Col definitions in the CSV.
    """
    m = re.search(r"(\d+)[_](\d+)", filename)
    if m:
        # For example, "2_2" represents splitting into 2 parts horizontally and vertically
        num_cols = int(m.group(1))
        num_rows = int(m.group(2))
        return num_rows, num_cols
    else:
        return 1, 1

def get_boundaries(length, parts):
    """
    Calculates the boundaries of each interval based on the total length and the number of parts.
    If it is not divisible, the first block is allocated an extra 1 pixel.
    Returns a list of length parts+1, where each pair of adjacent values defines a splitting interval.
    """
    boundaries = [0]
    base = length // parts
    remainder = length % parts
    for i in range(parts):
        # The first remainder blocks are allocated an extra 1 pixel
        step = base + (1 if i < remainder else 0)
        boundaries.append(boundaries[-1] + step)
    return boundaries

def process_images(csv_path, dataset_folder, output_folder, type):
    # Ensure that the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV data
    df = pd.read_csv(csv_path)
    # Check for required columns
    for col in ["Image Name", "Row", "Col"]:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain the '{col}' column")
    
    # Get the splitting information from the CSV file name (e.g., 2_2, 4_4, 6_6)
    csv_basename = os.path.basename(csv_path)
    grid_rows, grid_cols = extract_grid_dimensions(csv_basename)
    print(f"Extracted splitting information from the file name: {grid_rows} rows x {grid_cols} columns")
    
    # Process each record
    for index, row in df.iterrows():
        image_name = row["Image Name"]
        # Convert the Row and Col values from the CSV to integers
        try:
            r_idx = int(row["Row"])
            c_idx = int(row["Col"])
        except Exception as e:
            print(f"Error converting Row or Col in record {index}: {e}")
            continue

        # Check if the indices are within the acceptable range
        if not (0 <= r_idx < grid_rows) or not (0 <= c_idx < grid_cols):
            print(f"Coordinates ({r_idx}, {c_idx}) in record {index} are out of range, skipping.")
            continue

        # Adjust the image path based on the type
        actual_image_name = image_name
        if type == "mask":
            # For mask type, the file format is "0.png.png", "1.png.png", etc.
            # We need to append ".png" to the image name from the CSV
            actual_image_name = f"{image_name}.png"
        
        image_path = os.path.join(dataset_folder, actual_image_name)
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist, skipping.")
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                # Calculate the horizontal and vertical boundaries
                x_boundaries = get_boundaries(width, grid_cols)
                y_boundaries = get_boundaries(height, grid_rows)
                # Get the boundaries of the corresponding block based on the indices from the CSV
                left = x_boundaries[c_idx]
                right = x_boundaries[c_idx + 1]
                top = y_boundaries[r_idx]
                bottom = y_boundaries[r_idx + 1]
                cropped = img.crop((left, top, right, bottom))
                
                # Keep the original extension for the output file
                base, ext = os.path.splitext(image_name)
                if type == "mask":
                    # For mask type, we need to handle the double extension
                    # The base already contains the first extension (e.g., "0.png")
                    # out_name = f"{base}_r{r_idx}c{c_idx}{ext}.png"
                    out_name = f"{base}.png"
                else:
                    # out_name = f"{base}_r{r_idx}c{c_idx}{ext}"
                    out_name = f"{base}{ext}"
                
                out_path = os.path.join(output_folder, out_name)
                cropped.save(out_path)
                print(f"Successfully saved image in: {out_path}")
        except Exception as e:
            print(f"Failed to process {image_name} due to error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Split images into blocks based on the image number and coordinates recorded in a CSV")
    parser.add_argument("--csv", required=True, help="Full path to the CSV file")
    parser.add_argument("--dataset", required=True, help="Location of the original image dataset")
    parser.add_argument("--output", required=True, help="Location to store the processed images")
    parser.add_argument("--type", help="img or mask", default="img")
    args = parser.parse_args()

    process_images(args.csv, args.dataset, args.output, args.type)

if __name__ == "__main__":
    main()