# # 01 – Preprocessing Data
#
# This notebook performs the following steps:
#
# 1. Download data
# 2. (Optional) Combine instance segmentation masks from silver and gold ground truth
# 3. Save detections into a CSV file
# 4. Convert image and label data to Zarr format
# 5. Compute tracking accuracy metrics with only d(position) as edge cost
#
# ## Download Data
#
# The raw `.tif` images are first downloaded to the directory specified by `data_dir`.

from attrackt.scripts import extract_data, correct_gt_with_st, create_zarr, create_csv
from attrackt.scripts.motile import motile_infer
from attrackt.scripts.cumulate_scores import cumulate_scores


suffix = "HeLa-v1"  # just a string placeholder to identify the results, in case running multiple experiments. You can modify as needed.
data_dir = "./data/HeLa/"
zip_url = "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip"

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
# **Note:** This step will overwrite the contents of `01_GT/TRA` and `02_GT/TRA` with the newly merged masks.


correct_gt_with_st(
    silver_truth_dir_name=data_dir + "/Fluo-N2DL-HeLa/01_ST/SEG/",
    gold_truth_dir_name=data_dir + "/Fluo-N2DL-HeLa/01_GT/TRA/",
    combined_truth_dir_name=data_dir + "/Fluo-N2DL-HeLa/01_GT/TRA/",
)


correct_gt_with_st(
    silver_truth_dir_name=data_dir + "/Fluo-N2DL-HeLa/02_ST/SEG/",
    gold_truth_dir_name=data_dir + "/Fluo-N2DL-HeLa/02_GT/TRA/",
    combined_truth_dir_name=data_dir + "/Fluo-N2DL-HeLa/02_GT/TRA/",
)

# ## Export Detection Data to CSV
#
# Next, we generate a single CSV file containing the locations of all detected instances.
# Each detection is assigned a unique ID.
#
# The CSV includes the following columns:
# `sequence`, `id`, `t`, `[z]`, `y`, `x`, `parent_id`, `original_id`
#
# **Note:** The `man_track_file_names` input is optional.
# It is used here solely for evaluation purposes (e.g., assigning the values in the parent_id column), and is **not required for model training or for solving the ILP**.

detections_csv_file_name = data_dir + "/detections.csv"
sequence_names = [
    "01",
    "02",
]
test_sequence_names = [
    "02",
]
mask_dir_names = [
    data_dir + "/Fluo-N2DL-HeLa/01_GT/TRA",
    data_dir + "/Fluo-N2DL-HeLa/02_GT/TRA",
]
man_track_file_names = [
    data_dir + "/Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt",
    data_dir + "/Fluo-N2DL-HeLa/02_GT/TRA/man_track.txt",
]
img_dir_names = [
    data_dir + "/Fluo-N2DL-HeLa/01/",
    data_dir + "/Fluo-N2DL-HeLa/02/",
]

create_csv(
    mask_dir_names=mask_dir_names,
    sequence_names=sequence_names,
    man_track_file_names=man_track_file_names,
    output_csv_file_name=detections_csv_file_name,
)


# ## Convert Data to Zarr Format
#
# Then, we convert the data into a single Zarr file.
# This file contains one group for each sequence in the dataset.
#
# Within each sequence group, you'll find two datasets:
# - `img` – the raw image data
# - `mask` – the corresponding instance segmentation masks
# All the masks will be relabeled to ensure unique instance IDs across the
# entire dataset.

container_path = data_dir + "/HeLa.zarr"

create_zarr(
    container_path=container_path,
    img_dir_names=img_dir_names,
    mask_dir_names=mask_dir_names,
    sequence_names=sequence_names,
    mapping_csv_file_name=detections_csv_file_name,
)


# ## Compute Tracking Performance using only d(position) as edge cost
#
# Finally, to understand how well we perform using only position as a feature, we compute the tracking performance using only d(position) as edge cost.
# Please ensure `gurobi` is installed for a faster solving experience.

result_dir_name = f"results-position-only-{suffix}"
args = {
    "num_nearest_neighbours": 10,
    "direction_candidate_graph": "backward",
    "pin_nodes": True,
    "use_edge_distance": True,
    "edge_embedding_exists": False,
    "use_different_weights_hyper": True,
    "voxel_size": {"x": 1.0, "y": 1.0},
    "test_csv_file_name": detections_csv_file_name,
    "sequence_names": test_sequence_names,
    "result_dir_name": result_dir_name,
    "verbose": False,
}

motile_infer(args)


# ## Cumulate scores

result_test_sequences = [
    result_dir_name + "/" + test_sequence_name
    for test_sequence_name in test_sequence_names
]

cumulate_scores(sequence_names=result_test_sequences)
