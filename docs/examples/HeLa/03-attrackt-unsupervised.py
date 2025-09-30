# # 03 - Train trackastra model with attrackt unsupervised loss.
#
# This notebook uses the zarr container (created in the first notebook) and the embeddings (exported to a csv in the second notebook) to train a trackastra model with the attrackt unsupervised loss.

# Create a new dataset class that takes in the zarr container and the image and mask datasets.

from attrackt.scripts.trackastra.train_fully_unsupervised import train
from attrackt.scripts.trackastra.infer import infer
from types import SimpleNamespace

data_dir = "./data"

# ## Train model
#
# This may take longer than 24 hours!

args = SimpleNamespace(
    attn_positional_bias="rope",
    attn_positional_bias_n_spatial=16,
    augment=2,
    batch_size=8,
    cache=False,
    causal_norm="quiet_softmax",
    checkpoint_path=None,
    crop_size=(256, 256),
    d_model=256,
    delta_cutoff=2,
    distributed=False,
    dropout=0.1,
    dry=False,
    embeddings_csv=data_dir + "/HeLa/detections_embeddings.csv",
    epochs=600,
    example_images=False,
    feat_embed_per_dim=8,
    logger="tensorboard",
    lr=1e-4,
    lambda_=0.1,
    max_steps=1000000,
    max_tokens=2048,
    mixedp=True,
    model=None,
    name="HeLa",
    n_pool_sampler=0,
    nhead=4,
    ndim=2,
    num_decoder_layers=6,
    num_encoder_layers=6,
    num_workers=0,
    outdir="HeLa",
    pos_embed_per_dim=32,
    preallocate=True,
    profile=False,
    resume=False,
    seed=42,
    spatial_pos_cutoff=512,
    timestamp=True,
    tracking_frequency=-1,
    train_samples=10000,
    train_zarr_sequences=["01", "02"],
    val_zarr_sequences=["01", "02"],
    warmup_epochs=10,
    weight_by_dataset=False,
    weight_by_ndivs=False,
    window=6,
    zarr_img_channel=0,
    zarr_img_key="img",
    zarr_mask_channel=0,
    zarr_mask_key="mask",
    zarr_path=data_dir + "/HeLa/HeLa.zarr",
)


train(args)

# ## Infer using the trained model

infer(
    model_checkpoint="HeLa/2025-09-13_09-34-22_HeLa",
    zarr_img_channel=0,
    zarr_img_key="img",
    zarr_mask_channel=0,
    zarr_mask_key="mask",
    zarr_path=data_dir + "/HeLa/HeLa.zarr",
    test_zarr_sequences=["01", "02"],
    output_csv_file_name="associations.csv",
    edge_threshold=0.0,
    test_time_augmentation=True,
)


# ## Compute Tracking Performance using d(position) and predicted associations as edge costs.
#
# Please ensure `gurobi` is installed for a faster solving experience.

# +
result_dir_name = data_dir + "/HeLa/results_position_and_associations/"
detections_csv_file_name = data_dir + "/HeLa/detections.csv"
edge_embedding_file_name = "associations.csv"

args = {
    "num_nearest_neighbours": 10,
    "direction_candidate_graph": "backward",
    "pin_nodes": True,
    "use_edge_distance": True,
    "edge_embedding_exists": True,
    "use_different_weights_hyper": True,
    "voxel_size": {"x": 1.0, "y": 1.0},
    "test_csv_file_name": detections_csv_file_name,
    "edge_embedding_file_name": edge_embedding_file_name,
    "embedding_type": "affinity",
    "sequence_names": ["01", "02"],
    "result_dir_name": result_dir_name,
    "verbose": False,
}
# -

from attrackt.motile_scripts import infer

infer(args)

# ## Sort the nodes using an uncertainty measure.

from attrackt.scripts.sort import sort

sort(
    detection_csv_file_name="data/HeLa/detections.csv",
    prediction_csv_file_name="associations.csv",
    ilp_csv_file_name="data/HeLa/results_position_and_associations/tracking_results.csv",
    output_csv_file_name="sorted.csv",
    method="confidence",
    sequences=["02"],
)
