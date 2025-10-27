# # 03 - Train trackastra model with attrackt unsupervised loss.
#
# This notebook uses the zarr container (created in the first notebook) and the embeddings (exported to a csv in the second notebook) to train a trackastra model with the attrackt unsupervised loss.

# Create a new dataset class that takes in the zarr container and the image and mask datasets.


from attrackt.trackastra.data import CTCZarrData
import torch
from torch.utils.data import ConcatDataset
from attrackt.trackastra.model import TrackingTransformer
from functools import partial
from attrackt.scripts.trackastra.train_fully_unsupervised import train
from attrackt.scripts.trackastra.infer import infer
from attrackt.scripts.motile import motile_infer
from attrackt.scripts.cumulate_scores import cumulate_scores
from attrackt.scripts.sort import sort

suffix = "Bacteria-v1"  # just a string placeholder to identify the model and results, in case running multiple experiments. You can modify as needed.
data_dir = "./data/Bacteria/"
detections_csv = data_dir + "/detections.csv"
embeddings_csv = f"embeddings-{suffix}.csv"
sequences = [
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


test_sequences = [
    "150309-04",
    "150310-11",
    "151029_E1-1",
    "151029_E1-5",
    "151101_E3-12",
    "160112-06",
]


crop_size = (256, 256)
zarr_path = data_dir + "/Bacteria.zarr"


# ## Create train and val datasets.


ndim = 2
window = 6
max_tokens = 2048
augment = 2
zarr_img_key = "img"
zarr_mask_key = "mask"
zarr_img_channel = 0
zarr_mask_channel = 0
n_jobs = 8


ds_factory = partial(
    CTCZarrData,
    ndim=ndim,
    window_size=window,
    max_tokens=max_tokens,
    augment=augment,
    crop_size=crop_size,
    zarr_path=zarr_path,
    zarr_img_key=zarr_img_key,
    zarr_mask_key=zarr_mask_key,
    zarr_img_channel=zarr_img_channel,
    zarr_mask_channel=zarr_mask_channel,
    embeddings_csv=embeddings_csv,
    n_jobs=n_jobs,
)

train_datasets = [ds_factory(zarr_sequence=seq) for seq in sequences]
val_datasets = [
    ds_factory(zarr_sequence=seq, augment=0, crop_size=None) for seq in sequences
]

datasets = {
    "train": ConcatDataset(train_datasets),
    "val": ConcatDataset(val_datasets),
}

# ## Create model

d_model = 256
num_encoder_layers = 6
num_decoder_layers = 6
nhead = 4
pos_embed_per_dim = 32
feat_embed_per_dim = 8
spatial_pos_cutoff = 256
attn_positional_bias = "rope"
attn_positional_bias_n_spatial = 16
dropout = 0.1
causal_norm = "quiet_softmax"


model = TrackingTransformer(
    coord_dim=ndim,
    feat_dim=datasets["train"].datasets[0].feat_dim,
    d_model=d_model,
    pos_embed_per_dim=pos_embed_per_dim,
    feat_embed_per_dim=feat_embed_per_dim,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dropout=dropout,
    window=window,
    spatial_pos_cutoff=spatial_pos_cutoff,
    attn_positional_bias=attn_positional_bias,
    attn_positional_bias_n_spatial=attn_positional_bias_n_spatial,
    causal_norm=causal_norm,
    nhead=nhead,
)


# ## Train model

num_iterations = 1000_000
learning_rate = 1e-4
batch_size = 8
lambda_ = 0.1
num_workers = 2
d_model = 64

# Uncomment the next few lines to begin the training!

# +
# train(
#    datasets=datasets,
#    model=model,
#    batch_size=batch_size,
#    learning_rate=learning_rate,
#    device="cuda" if torch.cuda.is_available() else "mps",
#    num_iterations=num_iterations,
#    num_workers=num_workers,
#    causal_norm=causal_norm,
#    lambda_=lambda_,
#    window=window,
#    d_model=d_model,
#    log_loss_every=1_000,
#    suffix=suffix,
# )
# -


# ## Infer using the trained model

edge_embedding_file_name = f"associations-{suffix}.csv"
model_checkpoint = f"fully-unsupervised-2025-10-21-{suffix}/models/best/best.pth"  # Path to a saved model checkpoint. Modify accordingly.

# Uncomment the next few lines to use the trained model to infer the
# associations!

# +
# infer(
#    model_checkpoint= model_checkpoint,
#    transformer=model,
#    zarr_img_channel=0,
#    zarr_img_key="img",
#    zarr_mask_channel=0,
#    zarr_mask_key="mask",
#    zarr_path=data_dir + "/Bacteria.zarr",
#    test_zarr_sequences=test_sequences,
#    output_csv_file_name=edge_embedding_file_name,
#    edge_threshold=0.01,
#    test_time_augmentation=True,
# )
# -

# ## Compute Tracking Performance using d(position) and predicted associations as edge costs.
#
# Please ensure `gurobi` is installed for a faster solving experience.

result_dir_name = f"results-position-and-associations-{suffix}"

args = {
    "num_nearest_neighbours": 10,
    "direction_candidate_graph": "backward",
    "pin_nodes": True,
    "use_edge_distance": True,
    "edge_embedding_exists": True,
    "use_different_weights_hyper": True,
    "voxel_size": {"x": 1.0, "y": 1.0},
    "test_csv_file_name": detections_csv,
    "edge_embedding_file_name": edge_embedding_file_name,
    "embedding_type": "affinity",
    "sequence_names": test_sequences,
    "result_dir_name": result_dir_name,
    "verbose": False,
}

motile_infer(args)

# ## Sort the nodes using an uncertainty measure.

sort(
    detection_csv_file_name=detections_csv,
    prediction_csv_file_name=edge_embedding_file_name,
    ilp_csv_file_name=f"{result_dir_name}/tracking_results.csv",
    output_csv_file_name="sorted.csv",
    method="confidence",
    sequences=test_sequences,
    suffix=suffix,
)

# ## Cumulate scores

result_test_sequences = [
    result_dir_name + "/" + test_sequence for test_sequence in test_sequences
]
cumulate_scores(sequence_names=result_test_sequences)
