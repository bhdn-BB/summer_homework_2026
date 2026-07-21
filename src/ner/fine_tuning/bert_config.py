from pathlib import Path


NER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = NER_ROOT.parent.parent


FINE_TUNING_CONFIG = {
    "model_name": "bert-base-cased",
    # Use bert-large-cased from the CLI when comparing a larger checkpoint.
    "num_labels": 3,
    "id2label": {
        0: "O",
        1: "B-Mountain",
        2: "I-Mountain",
    },
    "label2id": {
        "O": 0,
        "B-Mountain": 1,
        "I-Mountain": 2,
    },
    "data_path": str(NER_ROOT / "data" / "few_nerd_mountains_output"),
    "train_split": "train",
    "eval_split": "validation",
    "test_split": "test",
    "output_dir": str(NER_ROOT / "fine_tuning" / "results" / "bert_mountain"),
    "log_file": str(NER_ROOT / "fine_tuning" / "logs" / "bert_training.log"),
    "save_model_path": str(NER_ROOT / "fine_tuning" / "models" / "bert_mountain"),
    "eval_strategy": "epoch",
    "logging_strategy": "steps",
    "logging_steps": 50,
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "best_model_metric": "f1",
    "greater_is_better": True,
    "learning_rate": 2e-5,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "num_epochs": 5,
    "weight_decay": 0.01,
    "save_total_limit": 2,
    "seed": 42,
    "push_to_hub": False,
    "hub_model_id": None,
    "max_train_examples": None,
    "max_eval_examples": None,
    "max_test_examples": None,
    "max_length": 256,
    "gradient_accumulation_steps": 1,
    "fp16": False,
    "device": "auto",
    "resume_from_checkpoint": None,
    # Available losses: cross_entropy, weighted_cross_entropy, focal.
    "loss_name": "cross_entropy",
    "normalize_class_weights": True,
    "focal_gamma": 2.0,
    "focal_use_class_weights": False,
    "report_to": ["wandb"],
    "wandb_project": "mountain-ner",
    "wandb_run_name": None,
}


def get_fine_tuning_config() -> dict:
    """Return a shallow copy of the Python fine-tuning configuration."""
    return FINE_TUNING_CONFIG.copy()
