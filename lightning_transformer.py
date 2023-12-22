import torch
import torch.nn.functional as F
from transformer import Transformer
from lightning import LightningModule


class LightningTransformer(LightningModule):
    def __init__(self, transformer: Transformer, learning_rate: float, ignore_index: int):
        super().__init__()
        self.transformer = transformer
        self.learning_rate = learning_rate
        self.ignore_index = ignore_index

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor]):
        source, target, source_ignore_mask, target_ignore_mask = batch
        predicted_targets = self.transformer(source, target[:, :-1], source_ignore_mask, target_ignore_mask[: , :-1])
        loss = F.cross_entropy(predicted_targets.permute(0, 2, 1), target[:, 1:], ignore_index=self.ignore_index)
        self.log("Training Loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor],
                        batch_index: int):
        source, target, source_ignore_mask, target_ignore_mask = batch
        predicted_targets = self.transformer(source, target[:, :-1], source_ignore_mask, target_ignore_mask[:, :-1])
        loss = F.cross_entropy(predicted_targets.permute(0, 2, 1), target[:, 1:], ignore_index=self.ignore_index)
        self.log("Validation Loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor], batch_index: int):
        source, target, source_ignore_mask, target_ignore_mask = batch
        predicted_targets = self.transformer(source, target[:, :-1], source_ignore_mask, target_ignore_mask[:, :-1])
        loss = F.cross_entropy(predicted_targets.permute(0, 2, 1), target[:, 1:], ignore_index=self.ignore_index)
        self.log("Test Loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
