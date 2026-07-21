"""Central defaults shared by matching scripts and experiment notes."""

DEFAULT_MATCHING_CONFIG = {
    "manifest_root": "/kaggle/input/deforestation-in-ukraine",
    "manifest_output": "/kaggle/working/pairs.csv",
    "max_positive_pairs_per_tile": 20,
    "negative_ratio": 1,
    "seed": 42,
    "bands": "B04,B03,B02",
    "method": "sift",
    "single_output": "matching_output/matches.png",
    "methods": "sift,orb,loftr",
    "tile_size": 1024,
    "overlap": 128,
    "max_features": 4000,
    "max_side": 2400,
    "ransac_threshold": 5.0,
    "device": "auto",
    "benchmark_output_dir": "/kaggle/working/matching_results",
    "backbone": "efficientnet_b0",
    "embedding_dim": 256,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "workers": 2,
    "output": "/kaggle/working/matcher_best.pt",
    "wandb_project": "sentinel-matching",
    "wandb_run_name": None,
}


def get_matching_config() -> dict:
    """Return a mutable copy for a single experiment."""
    return DEFAULT_MATCHING_CONFIG.copy()
