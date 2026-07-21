"""Custom Hugging Face Trainer used by the BERT fine-tuning scripts."""

import torch
import torch.nn.functional as F
from transformers import Trainer


class MountainTrainer(Trainer):
    """Trainer with configurable losses for mountain token classification."""

    def __init__(self, *args, loss_config: dict, class_weights=None, **kwargs):
        """Initialize the trainer with loss settings and optional class weights."""
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute the configured token-classification loss for one batch."""
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        loss_name = self.loss_config["loss_name"]

        if loss_name == "cross_entropy":
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        elif loss_name == "weighted_cross_entropy":
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                weight=self.class_weights.to(logits.device),
                ignore_index=-100,
            )
        elif loss_name == "focal":
            loss = self._focal_loss(logits, labels)
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        return (loss, outputs) if return_outputs else loss

    def _focal_loss(self, logits, labels):
        """Compute focal loss while ignoring padding and special-token labels."""
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        valid_mask = flat_labels != -100
        flat_logits = flat_logits[valid_mask]
        flat_labels = flat_labels[valid_mask]

        weights = None
        if self.loss_config["focal_use_class_weights"]:
            weights = self.class_weights.to(logits.device)

        ce_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            weight=weights,
            reduction="none",
        )
        probabilities = torch.softmax(flat_logits, dim=-1)
        target_probabilities = probabilities.gather(
            dim=-1,
            index=flat_labels.unsqueeze(-1),
        ).squeeze(-1)

        return ((1.0 - target_probabilities).pow(self.loss_config["focal_gamma"]) * ce_loss).mean()
