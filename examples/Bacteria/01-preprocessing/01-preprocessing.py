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

from attrackt.scripts import extract_data, create_zarr, create_csv
from attrackt.scripts.motile import motile_infer
from attrackt.scripts.cumulate_scores import cumulate_scores


suffix = "Bacteria-v1"  # just a string placeholder to identify the results, in case running multiple experiments. You can modify as needed.
data_dir = "./data/Bacteria/"
zip_url = "https://drive.google.com/uc?id=1OUj5LtG8SGJZN_M6g63SVs6_54O7oFEK"

extract_data(
    zip_url=zip_url,
    data_dir=data_dir,
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
    "140408-02",
    "140408-04",
    "140408-10",
    "140409-03",
    "140415-08",
    "151027-05",
    "151027-06",
    "151027-10",
    "151028-01",
    "151029-05",
    "151029-11",
    "151029_E1-6",
    "151101_E2-1",
    "151101_E2-2",
    "151101_E3-11",
    "151101_E4-12",
    "151101_E4-19",
    "151101_E4-20",
    "150317-07",
    "150318-06",
    "150331-12",
    "151222-10",
    "151222-11",
    "150324-03",
    "150324-05",
    "150325-04",
    "160112-04",
    "150303-01",
    "150428-08",
    "151021-11",
    "140415-13",
    "151031-03",
    "150309-04",
    "150310-11",
    "151029_E1-1",
    "151029_E1-5",
    "151101_E3-12",
    "160112-06",
]

test_sequence_names = [
    "150309-04",
    "150310-11",
    "151029_E1-1",
    "151029_E1-5",
    "151101_E3-12",
    "160112-06",
]

mask_dir_names = [
    data_dir + "/Bacteria/train/140408-02/TRA",
    data_dir + "/Bacteria/train/140408-04/TRA",
    data_dir + "/Bacteria/train/140408-10/TRA",
    data_dir + "/Bacteria/train/140409-03/TRA",
    data_dir + "/Bacteria/train/140415-08/TRA",
    data_dir + "/Bacteria/train/151027-05/TRA",
    data_dir + "/Bacteria/train/151027-06/TRA",
    data_dir + "/Bacteria/train/151027-10/TRA",
    data_dir + "/Bacteria/train/151028-01/TRA",
    data_dir + "/Bacteria/train/151029-05/TRA",
    data_dir + "/Bacteria/train/151029-11/TRA",
    data_dir + "/Bacteria/train/151029_E1-6/TRA",
    data_dir + "/Bacteria/train/151101_E2-1/TRA",
    data_dir + "/Bacteria/train/151101_E2-2/TRA",
    data_dir + "/Bacteria/train/151101_E3-11/TRA",
    data_dir + "/Bacteria/train/151101_E4-12/TRA",
    data_dir + "/Bacteria/train/151101_E4-19/TRA",
    data_dir + "/Bacteria/train/151101_E4-20/TRA",
    data_dir + "/Bacteria/train/150317-07/TRA",
    data_dir + "/Bacteria/train/150318-06/TRA",
    data_dir + "/Bacteria/train/150331-12/TRA",
    data_dir + "/Bacteria/train/151222-10/TRA",
    data_dir + "/Bacteria/train/151222-11/TRA",
    data_dir + "/Bacteria/train/150324-03/TRA",
    data_dir + "/Bacteria/train/150324-05/TRA",
    data_dir + "/Bacteria/train/150325-04/TRA",
    data_dir + "/Bacteria/train/160112-04/TRA",
    data_dir + "/Bacteria/train/150303-01/TRA",
    data_dir + "/Bacteria/train/150428-08/TRA",
    data_dir + "/Bacteria/train/151021-11/TRA",
    data_dir + "/Bacteria/val/140415-13/TRA",
    data_dir + "/Bacteria/val/151031-03/TRA",
    data_dir + "/Bacteria/test/150309-04/TRA",
    data_dir + "/Bacteria/test/150310-11/TRA",
    data_dir + "/Bacteria/test/151029_E1-1/TRA",
    data_dir + "/Bacteria/test/151029_E1-5/TRA",
    data_dir + "/Bacteria/test/151101_E3-12/TRA",
    data_dir + "/Bacteria/test/160112-06/TRA",
]


man_track_file_names = [
    data_dir + "/Bacteria/train/140408-02/TRA/man_track.txt",
    data_dir + "/Bacteria/train/140408-04/TRA/man_track.txt",
    data_dir + "/Bacteria/train/140408-10/TRA/man_track.txt",
    data_dir + "/Bacteria/train/140409-03/TRA/man_track.txt",
    data_dir + "/Bacteria/train/140415-08/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151027-05/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151027-06/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151027-10/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151028-01/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151029-05/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151029-11/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151029_E1-6/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151101_E2-1/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151101_E2-2/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151101_E3-11/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151101_E4-12/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151101_E4-19/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151101_E4-20/TRA/man_track.txt",
    data_dir + "/Bacteria/train/150317-07/TRA/man_track.txt",
    data_dir + "/Bacteria/train/150318-06/TRA/man_track.txt",
    data_dir + "/Bacteria/train/150331-12/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151222-10/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151222-11/TRA/man_track.txt",
    data_dir + "/Bacteria/train/150324-03/TRA/man_track.txt",
    data_dir + "/Bacteria/train/150324-05/TRA/man_track.txt",
    data_dir + "/Bacteria/train/150325-04/TRA/man_track.txt",
    data_dir + "/Bacteria/train/160112-04/TRA/man_track.txt",
    data_dir + "/Bacteria/train/150303-01/TRA/man_track.txt",
    data_dir + "/Bacteria/train/150428-08/TRA/man_track.txt",
    data_dir + "/Bacteria/train/151021-11/TRA/man_track.txt",
    data_dir + "/Bacteria/val/140415-13/TRA/man_track.txt",
    data_dir + "/Bacteria/val/151031-03/TRA/man_track.txt",
    data_dir + "/Bacteria/test/150309-04/TRA/man_track.txt",
    data_dir + "/Bacteria/test/150310-11/TRA/man_track.txt",
    data_dir + "/Bacteria/test/151029_E1-1/TRA/man_track.txt",
    data_dir + "/Bacteria/test/151029_E1-5/TRA/man_track.txt",
    data_dir + "/Bacteria/test/151101_E3-12/TRA/man_track.txt",
    data_dir + "/Bacteria/test/160112-06/TRA/man_track.txt",
]

img_dir_names = [
    data_dir + "/Bacteria/train/140408-02/img",
    data_dir + "/Bacteria/train/140408-04/img",
    data_dir + "/Bacteria/train/140408-10/img",
    data_dir + "/Bacteria/train/140409-03/img",
    data_dir + "/Bacteria/train/140415-08/img",
    data_dir + "/Bacteria/train/151027-05/img",
    data_dir + "/Bacteria/train/151027-06/img",
    data_dir + "/Bacteria/train/151027-10/img",
    data_dir + "/Bacteria/train/151028-01/img",
    data_dir + "/Bacteria/train/151029-05/img",
    data_dir + "/Bacteria/train/151029-11/img",
    data_dir + "/Bacteria/train/151029_E1-6/img",
    data_dir + "/Bacteria/train/151101_E2-1/img",
    data_dir + "/Bacteria/train/151101_E2-2/img",
    data_dir + "/Bacteria/train/151101_E3-11/img",
    data_dir + "/Bacteria/train/151101_E4-12/img",
    data_dir + "/Bacteria/train/151101_E4-19/img",
    data_dir + "/Bacteria/train/151101_E4-20/img",
    data_dir + "/Bacteria/train/150317-07/img",
    data_dir + "/Bacteria/train/150318-06/img",
    data_dir + "/Bacteria/train/150331-12/img",
    data_dir + "/Bacteria/train/151222-10/img",
    data_dir + "/Bacteria/train/151222-11/img",
    data_dir + "/Bacteria/train/150324-03/img",
    data_dir + "/Bacteria/train/150324-05/img",
    data_dir + "/Bacteria/train/150325-04/img",
    data_dir + "/Bacteria/train/160112-04/img",
    data_dir + "/Bacteria/train/150303-01/img",
    data_dir + "/Bacteria/train/150428-08/img",
    data_dir + "/Bacteria/train/151021-11/img",
    data_dir + "/Bacteria/val/140415-13/img",
    data_dir + "/Bacteria/val/151031-03/img",
    data_dir + "/Bacteria/test/150309-04/img",
    data_dir + "/Bacteria/test/150310-11/img",
    data_dir + "/Bacteria/test/151029_E1-1/img",
    data_dir + "/Bacteria/test/151029_E1-5/img",
    data_dir + "/Bacteria/test/151101_E3-12/img",
    data_dir + "/Bacteria/test/160112-06/img",
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

container_path = data_dir + "/Bacteria.zarr"

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
