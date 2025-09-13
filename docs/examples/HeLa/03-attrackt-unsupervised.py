# # 03 - Train trackastra model with attrackt unsupervised loss.
#
# This notebook uses the zarr container (created in the first notebook) and the embeddings (exported to a csv in the second notebook) to train a trackastra model with the attrackt unsupervised loss.

# Create a new dataset class that takes in the zarr container and the image and mask datasets.

from attrackt.scripts.trackastra.train_fully_unsupervised import train
from types import SimpleNamespace

data_dir =  './data'

args = SimpleNamespace(
    assoc_csv=data_dir + '/HeLa/detections.csv',
    attn_positional_bias='rope',
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
    embedding_csv=data_dir + '/HeLa/detections_embeddings.csv',
    epochs=5,
    example_images=False,
    feat_embed_per_dim=8,
    logger='tensorboard',
    lr=1e-4,
    lambda_=0.1,
    max_steps=1000000,
    max_tokens=2048,
    mixedp=True,
    model=None,
    name='HeLa',
    n_pool_sampler=0,
    nhead=4,
    ndim=2,
    num_decoder_layers=6,
    num_encoder_layers=6,
    num_workers=0,
    outdir='HeLa',
    pos_embed_per_dim=32,
    preallocate=True,
    profile=False,
    resume=False,
    seed=42,
    spatial_pos_cutoff=512,
    timestamp=True,
    tracking_frequency=-1,
    train_samples=10000,
    train_zarr_sequences=['01', '02'],
    val_zarr_sequences=['01', '02'],
    warmup_epochs=10,
    weight_by_dataset=False,
    weight_by_ndivs=False,
    window=6,
    zarr_img_channel=0,
    zarr_img_key='img',
    zarr_mask_channel=0,
    zarr_mask_key='mask',
    zarr_path=data_dir + '/HeLa/HeLa.zarr',
)


train(args)

# ## Create train and val datasets.

data_dir =  './data'
ndim = 2
window = 6
max_tokens = 2048
augment = 2
crop_size = (256, 256)
zarr_path = data_dir + "/HeLa/HeLa.zarr"
train_zarr_sequences = [ '01', '02']
val_zarr_sequences = ['01', '02']
zarr_img_key = 'img'
zarr_mask_key = 'mask'
zarr_img_channel = 0
zarr_mask_channel = 0
detections_csv_file_name = data_dir + "/HeLa/detections.csv"
embedding_csv_file_name = data_dir + "/HeLa/detections_embeddings.csv"

# +
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
    assoc_csv=detections_csv_file_name,
    embedding_csv=embedding_csv_file_name,
)

train_datasets = [ds_factory(zarr_sequence=seq) for seq in train_zarr_sequences]
val_datasets = [ds_factory(zarr_sequence=seq, augment=0, crop_size=None)
                for seq in val_zarr_sequences]

datasets = {
    "train": ConcatDataset(train_datasets),
    "val":   ConcatDataset(val_datasets),
}
# -

# ## Create model

d_model =  256
num_encoder_layers = 6
num_decoder_layers = 6
nhead = 4
pos_embed_per_dim = 32
feat_embed_per_dim = 8
spatial_pos_cutoff = 256
attn_positional_bias = 'rope'
attn_positional_bias_n_spatial=16  
dropout = 0.1
causal_norm= 'quiet_softmax'


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
        nhead = nhead
        )

# ## Train model

num_iterations = 1000_000
learning_rate = 1e-4
batch_size = 8
lambda_ = 0.1
num_workers = 2
delta_cutoff = 2
d_model = 64

train(
    datasets=datasets,
    model=model,
    batch_size=batch_size,
    learning_rate=learning_rate,
    device="cuda" if torch.cuda.is_available() else "mps",
    num_iterations=num_iterations,
    delta_cutoff=delta_cutoff,
    num_workers = num_workers,
    causal_norm=causal_norm,
    lambda_ = lambda_,
    window = window,
    d_model = d_model,
    log_loss_every=1250
)







