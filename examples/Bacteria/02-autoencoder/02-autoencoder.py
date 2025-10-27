# # 02 - Train auto-encoder model.
#
# This notebook performs the following steps:
#
# 1. **Train an autoencoder model**
#   Image crops centered on detections are extracted from the raw dataset and used to train the model.
#   You can choose between:
#   - A **standard autoencoder**, trained using a reconstruction loss to reproduce the input crops.
#   - A **variational autoencoder (VAE)**, trained using a combination of
#   negative log-likelihood and KL divergence loss.
#
# 2. **Infer embeddings using the trained encoder**
#   - For a standard autoencoder, the encoder’s output is directly used as the embedding.
#   - For a variational autoencoder, the posterior mean is used as the
#   embedding representation.

# ## Create an iterable, train dataset.
#
# It samples a random row from the detections file and extracts the image crop at that
# position.

import torch
import csv
from attrackt.autoencoder import (
    ZarrCsvDatasetAutoencoder,
    AutoencoderModel,
    AutoencoderModel3d,
)
from attrackt.scripts.autoencoder.train_autoencoder import train_autoencoder
from attrackt.scripts.autoencoder.infer_autoencoder import infer_autoencoder
from attrackt.scripts.motile import motile_infer
from attrackt.scripts.cumulate_scores import cumulate_scores


suffix = "Bacteria-v1"  # just a string placeholder to identify the model and results, in case running multiple experiments. You can modify as needed.
data_dir = "./data/Bacteria/"
container_path = data_dir + "/Bacteria.zarr"
detections_csv_file_name = data_dir + "/detections.csv"
crop_size = [64, 64]
num_spatial_dims = 2


train_dataset = ZarrCsvDatasetAutoencoder(
    zarr_container_name=container_path,
    detections_csv_file_name=detections_csv_file_name,
    num_spatial_dims=num_spatial_dims,
    crop_size=crop_size,
    length=None,
)


val_dataset = ZarrCsvDatasetAutoencoder(
    zarr_container_name=container_path,
    detections_csv_file_name=detections_csv_file_name,
    num_spatial_dims=num_spatial_dims,
    crop_size=crop_size,
    length=100,
    shuffle=True,
)

# ## Create model

num_in_out_channels = 1  # Number of input and output channels, e.g. 1 for gray-scale images, 3 for RGB images.
num_intermediate_channels = 32
num_z_channels = 4
attention_resolutions = [16]  # resolution level at which attention is applied.
num_res_blocks = 2
channels_multiply_factor = [
    1,
    1,
    2,
    2,
    4,
]  # how number of channels increase at each resolution level.
variational = False
embedding_dimension = 4  # Dimension of the embedding vector, used for VAE. If variational is False, this is ignored.

kwargs = dict(
    num_in_out_channels=num_in_out_channels,
    num_intermediate_channels=num_intermediate_channels,
    num_z_channels=num_z_channels,
    attention_resolutions=attention_resolutions,
    num_res_blocks=num_res_blocks,
    resolution=crop_size[-1],
    channels_multiply_factor=channels_multiply_factor,
    device="cuda" if torch.cuda.is_available() else "mps",
    variational=variational,
    embedding_dimension=embedding_dimension,  # Used for VAE.
)

ModelClass = AutoencoderModel if num_spatial_dims == 2 else AutoencoderModel3d
model = ModelClass(**kwargs)


# ## Train model

num_iterations = 100_000
learning_rate = 5e-5
batch_size = 8
num_workers = 2 if torch.cuda.is_available() else 0

# Now we can begin the training! <br>
# Uncomment the next few lines to train the model.

# +
# train_autoencoder(
#    dataset=train_dataset,
#    dataset_val=val_dataset,
#    model=model,
#    batch_size=batch_size,
#    learning_rate=learning_rate,
#    device="cuda" if torch.cuda.is_available() else "mps",
#    num_iterations=num_iterations,
#    num_workers=num_workers,
#    suffix=suffix,
# )
# -

# ## Infer embeddings
#
# Next, we use the trained model weights to infer the embeddings for each
# detection in the dataset.

model_checkpoint = f"autoencoder-2025-10-11-{suffix}/models_autoencoder/best.pth"  # Path to a saved model checkpoint.

with open(detections_csv_file_name, newline="") as f:
    reader = csv.reader(f)
    num_rows = sum(1 for row in reader) - 1  # subtract header row
print(f"Number of detections in dataset is {num_rows}.")

test_dataset = ZarrCsvDatasetAutoencoder(
    zarr_container_name=container_path,
    detections_csv_file_name=detections_csv_file_name,
    num_spatial_dims=num_spatial_dims,
    crop_size=crop_size,
    length=num_rows,
    shuffle=False,
)

node_embedding_file_name = f"embeddings-{suffix}.csv"

# Now we can begin the inference! <br>
# Uncomment the next few lines to run the inference.

# +
# infer_autoencoder(
#    dataset=test_dataset,
#    model=model,
#    model_checkpoint=model_checkpoint,
#    device="cuda" if torch.cuda.is_available() else "mps",
#    output_csv_file_name=node_embedding_file_name,
# )
# -

# ## Compute Tracking Performance using d(position) and d(embedding) as edge costs.
#
# Please ensure `gurobi` is installed for a faster solving experience.

result_dir_name = f"results-position-and-embeddings-{suffix}"
test_sequence_names = [
    "150309-04",
    "150310-11",
    "151029_E1-1",
    "151029_E1-5",
    "151101_E3-12",
    "160112-06",
]
args = {
    "num_nearest_neighbours": 10,
    "direction_candidate_graph": "backward",
    "pin_nodes": True,
    "use_edge_distance": True,
    "edge_embedding_exists": True,
    "use_different_weights_hyper": True,
    "voxel_size": {"x": 1.0, "y": 1.0},
    "test_csv_file_name": detections_csv_file_name,
    "node_embedding_file_name": node_embedding_file_name,
    "embedding_type": "distance",
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
