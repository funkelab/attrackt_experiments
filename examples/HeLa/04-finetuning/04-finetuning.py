# # 04 - Fine-tune trackastra model with supervision on `k` nodes.
#
# This notebook uses the zarr container (created in the first notebook), the embeddings (exported to a csv in the second notebook), and the model trained with unsupervised loss (in the third notebook) to fine-tune the model weights with GT supervision.

from attrackt.trackastra.data import CTCZarrData
import torch
from torch.utils.data import ConcatDataset
from attrackt.trackastra.model import TrackingTransformer
from functools import partial
from attrackt.scripts.trackastra.train_finetuned import train
from attrackt.scripts.identify_valid_sequences import identify_valid_sequences
from attrackt.scripts.motile import motile_infer
from attrackt.scripts.cumulate_scores import cumulate_scores
from attrackt.scripts.trackastra.infer_finetuned import infer

suffix = "HeLa-v1"  # just a string placeholder to identify the model and results, in case running multiple experiments. You can modify as needed.
data_dir = "./data/HeLa/"
detections_csv = data_dir + "/detections.csv"
embeddings_csv = f"embeddings-{suffix}.csv"

sequences = ["02"]
crop_size = (256, 256)
zarr_path = data_dir + "/HeLa.zarr"


# ## Create train and val unsupervised datasets

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

train_unsupervised_datasets = [ds_factory(zarr_sequence=seq) for seq in sequences]


# ## Create train and val supervised_datasets
k = 100  # Number of supervised nodes! Choose one of {10, 100, 1000, 10000, 100000}
supervised_csv = f"confidence-{suffix}/{k}/top/edges_sorted.csv"

# identify valid sequences as those that have representation in the
# supervised_csv file

valid_sequences = identify_valid_sequences(sequences, supervised_csv)

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
    supervised_csv=supervised_csv,
    n_jobs=n_jobs,
)

train_supervised_datasets = [ds_factory(zarr_sequence=seq) for seq in valid_sequences]
val_supervised_datasets = [
    ds_factory(zarr_sequence=seq, augment=0, crop_size=None) for seq in valid_sequences
]

# ## Create train and val pseudo-supervised datasets
pseudo_supervised_csv = f"confidence-{suffix}/{k}/bottom/edges_sorted.csv"

valid_sequences = identify_valid_sequences(sequences, pseudo_supervised_csv)
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
    supervised_csv=pseudo_supervised_csv,
    n_jobs=n_jobs,
)

train_pseudo_supervised_datasets = [
    ds_factory(zarr_sequence=seq) for seq in valid_sequences
]
val_pseudo_supervised_datasets = [
    ds_factory(zarr_sequence=seq, augment=0, crop_size=None) for seq in valid_sequences
]


datasets = {
    "train_unsupervised": ConcatDataset(train_unsupervised_datasets),
    "train_supervised": ConcatDataset(train_supervised_datasets),
    "val_supervised": ConcatDataset(val_supervised_datasets),
    "train_pseudo_supervised": ConcatDataset(train_pseudo_supervised_datasets),
    "val_pseudo_supervised": ConcatDataset(val_pseudo_supervised_datasets),
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
    feat_dim=datasets["train_unsupervised"].datasets[0].feat_dim,
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

learning_rate = 1e-4
batch_size = 8
lambda_ = 0.1
num_iterations = 100 * k
num_workers = 2
d_model = 64
unsupervised_model_checkpoint = f"fully-unsupervised-2025-10-11-{suffix}/models/best/best.pth"  # Path to a saved model checkpoint. Modify accordingly.
do_lora = True
lora_r = 32
lora_alpha = 32
lora_dropout = 0.1

# +
# train(
#    datasets=datasets,
#    model=model,
#    checkpoint_path=unsupervised_model_checkpoint,
#    batch_size=batch_size,
#    learning_rate=learning_rate,
#    device="cuda" if torch.cuda.is_available() else "mps",
#    num_iterations=num_iterations,
#    num_workers=num_workers,
#    causal_norm=causal_norm,
#    lambda_=lambda_,
#    window=window,
#    d_model=d_model,
#    do_lora=do_lora,
#    lora_r=lora_r,
#    lora_alpha=lora_alpha,
#    lora_dropout=lora_dropout,
#    suffix=suffix,
# )
# -

associations_csv = f"finetuned-associations-{k}-{suffix}.csv"
finetuned_model_checkpoint_dir = f"finetuned-2025-10-11-{suffix}/models/last/"
# Path to finetuned model checkpoint directory. Modify accordingly.

# +
# infer(
#    unsupervised_model_checkpoint=unsupervised_model_checkpoint,
#    transformer=model,
#    finetuned_model_checkpoint_dir=finetuned_model_checkpoint_dir,
#    zarr_img_channel=zarr_img_channel,
#    zarr_img_key=zarr_img_key,
#    zarr_mask_channel=zarr_mask_channel,
#    zarr_mask_key=zarr_mask_key,
#    zarr_path=zarr_path,
#    test_zarr_sequences=sequences,
#    output_csv_file_name=associations_csv,
# )
# -

# ## Compute Tracking Performance using d(position) and predicted associations as edge costs.
#
# Please ensure `gurobi` is installed for a faster solving experience.

result_dir_name = f"results-position-and-finetuned-associations-{suffix}"

args = {
    "num_nearest_neighbours": 10,
    "direction_candidate_graph": "backward",
    "pin_nodes": True,
    "use_edge_distance": True,
    "edge_embedding_exists": True,
    "use_different_weights_hyper": True,
    "voxel_size": {"x": 1.0, "y": 1.0},
    "test_csv_file_name": detections_csv,
    "edge_embedding_file_name": associations_csv,
    "embedding_type": "affinity",
    "sequence_names": sequences,
    "supervised_csv": supervised_csv,
    "result_dir_name": result_dir_name,
    "verbose": False,
}

motile_infer(args)


# ## Cumulate scores

result_sequences = [result_dir_name + "/" + sequence for sequence in sequences]
cumulate_scores(sequence_names=result_sequences)
