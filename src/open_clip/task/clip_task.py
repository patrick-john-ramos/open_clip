from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .image_text_task import ImageTextTask


class CLIPTask(ImageTextTask):
    """Standard CLIP training task wrapping model + ClipLoss."""

    def __init__(
            self,
            model: nn.Module,
            *,
            loss: Optional[nn.Module] = None,
            default_loss: bool = True,
            local_loss: bool = False,
            gather_with_grad: bool = False,
            cache_labels: bool = True,
            rank: int = 0,
            world_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            verbose: bool = True,
            num_subcaptions: int = 0,
    ):
        super().__init__(model, device=device, dtype=dtype, verbose=verbose)
        if loss is not None:
            self.loss = loss
        elif default_loss:
            from open_clip.loss import ClipLoss
            self.loss = ClipLoss(
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                cache_labels=cache_labels,
                rank=rank,
                world_size=world_size,
            )
        # else: eval-only construction, no self.loss attribute
        self.num_subcaptions = num_subcaptions

    def training_forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        if self.num_subcaptions > 0:
            return self._subcaption_training_forward(batch)
        model_out = self.trainable_module(image=batch["image"], text=batch["text"])
        losses = self.loss(**model_out, output_dict=True)
        total_loss = sum(v for k, v in losses.items() if k.endswith('_loss'))
        losses["loss"] = total_loss
        return losses, self._report(model_out)

    def _subcaption_training_forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
        # DreamLIP: average K independent CLIP losses that share image features.
        text = batch["text"]                                   # (B, K, L)
        assert text.ndim == 3 and text.shape[1] == self.num_subcaptions, (
            f"expected text of shape (B, {self.num_subcaptions}, L), got {tuple(text.shape)}; "
            "--num-subcaptions requires the WebDataset train pipeline"
        )
        B, K = text.shape[:2]
        model_out = self.trainable_module(image=batch["image"], text=text.flatten(0, 1))   # (B*K, L)
        text_features = model_out["text_features"].reshape(B, K, -1)
        contrastive = 0.0
        for j in range(K):                                     # K independent CLIP losses sharing image features
            out_j = {**model_out, "text_features": text_features[:, j]}
            contrastive = contrastive + self.loss(**out_j, output_dict=True)["contrastive_loss"]
        contrastive = contrastive / K
        losses = {"contrastive_loss": contrastive, "loss": contrastive}
        return losses, self._report(model_out)

    def eval_forward(self, batch: Dict[str, torch.Tensor]):
        inputs = {key: batch[key] for key in self.data_keys if key in batch}
        return self.get_trainable_module(use_ema=True)(**inputs)
