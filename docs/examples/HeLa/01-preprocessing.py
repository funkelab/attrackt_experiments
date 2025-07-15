# # 01 – Preprocessing Data
#
# This notebook performs the following steps:
#
# 1. Download data
# 2. (Optional) Combine instance segmentation masks from silver and gold ground truth
# 3. Convert image and label data to Zarr format
# 4. Save detections into a CSV file
#
# ## Download Data
#
# The raw `.tif` images are first downloaded to the directory specified by `data_dir`.

from attrackt.scripts import extract_data, correct_gt_with_st, create_zarr, create_csv


zip_url = (
    "https://github.com/funkelab/attrackt_experiments/releases/download/v0.0.1/HeLa.zip"
)
data_dir = "./data"

extract_data(
    zip_url=zip_url,
    data_dir=data_dir,
)

# ## (Optional) Merge Silver and Gold Segmentation Masks
#
# In this optional step, we combine the **segmentation masks** from the silver and gold ground truth datasets.
#
# You can **skip this step** if:
# - Your masks are already dense and gold-standard, or
# - You do not have silver ground truth data available.
#
# **Note:** This step will overwrite the contents of `01_GT/TRA` with the newly merged masks.


correct_gt_with_st(
    silver_truth_dir_name=data_dir + "/HeLa/01_ST/SEG/",
    gold_truth_dir_name=data_dir + "/HeLa/01_GT/TRA/",
    combined_truth_dir_name=data_dir + "/HeLa/01_GT/TRA/",
)


correct_gt_with_st(
    silver_truth_dir_name=data_dir + "/HeLa/02_ST/SEG/",
    gold_truth_dir_name=data_dir + "/HeLa/02_GT/TRA/",
    combined_truth_dir_name=data_dir + "/HeLa/02_GT/TRA/",
)

# ## Convert Data to Zarr Format
#
# Next, we convert the data into a single Zarr file.
# This file contains one group for each sequence in the dataset.
#
# Within each sequence group, you'll find two datasets:
# - `img` – the raw image data
# - `mask` – the corresponding instance segmentation masks

container_path = data_dir + "/HeLa/HeLa.zarr"
img_dir_names = [data_dir + "/HeLa/01/", data_dir + "/HeLa/02/"]
mask_dir_names = [data_dir + "/HeLa/01_GT/TRA/", data_dir + "/HeLa/02_GT/TRA/"]
sequence_names = ["01", "02"]

create_zarr(
    container_path=container_path,
    img_dir_names=img_dir_names,
    mask_dir_names=mask_dir_names,
    sequence_names=sequence_names,
)


# ## Export Detection Data to CSV
#
# Finally, we generate a single CSV file containing the locations of all detected instances.
# Each detection is assigned a unique ID.
#
# The CSV includes the following columns:
# `sequence`, `id`, `t`, `[z]`, `y`, `x`, `parent_id`, `original_id`
#
# **Note:** The `man_track_file_names` input is optional.
# It is used here solely for evaluation purposes (e.g., assigning parent IDs), and is **not required for model training**.

create_csv(
    zarr_container_name=container_path,
    zarr_sequence_names=sequence_names,
    zarr_dataset_name="mask",
    man_track_file_names=[
        data_dir + "/HeLa/01_GT/TRA/man_track.txt",
        data_dir + "/HeLa/02_GT/TRA/man_track.txt",
    ],
    output_csv_file_name=data_dir + "/HeLa/detections.csv",
)
