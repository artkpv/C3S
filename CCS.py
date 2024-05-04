import lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig


class CCS(pl.LightningModule):
    """
    Contrast Consistent Search probe.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self._cfg = cfg.ccs
        self.fc = nn.Sequential(nn.Linear(cfg._input_dim, 1), nn.Sigmoid())
        self.known_questions = set()

    def _normalize(
        self,
        x: torch.Tensor,
        pn_type: int,  # Positive (1) or negative (0).
        update_running=False,
    ):
        """
        Mean-normalizes the data x (of shape (n, d))
        If var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0)
        if self._cfg.var_normalize:
            normalized_x /= x.std(axis=0) + self._cfg.epsilon

        return normalized_x

    def loss(self, p0: torch.Tensor, p1: torch.Tensor):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        confidence = (torch.min(p0, p1) ** 2).mean(0)
        consistency = ((p1 - (1 - p0)) ** 2).mean(0)
        return confidence + consistency

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        neg_x, pos_x, *_ = batch
        pos_x = self._normalize(pos_x, pn_type=1)
        neg_x = self._normalize(neg_x, pn_type=0)
        neg_p = self.fc(neg_x).squeeze()
        pos_p = self.fc(pos_x).squeeze()
        loss = self.loss(neg_p, pos_p)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        neg_x, pos_x, y, *_ = batch
        pos_x = self._normalize(pos_x, pn_type=1)
        neg_x = self._normalize(neg_x, pn_type=0)
        with torch.no_grad():
            neg_p = self.fc(neg_x).squeeze()
            pos_p = self.fc(pos_x).squeeze()
        avg_confidence = 0.5 * (pos_p + (1 - neg_p))
        predictions = (avg_confidence.detach() >= 0.5).int()
        acc = (predictions == y).float().mean()
        acc = max(acc, 1 - acc)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # Known questions:
        for q_id in batch[4].unique():
            q_indexes = (batch[4] == q_id).nonzero(as_tuple=False).squeeze()
            if (predictions[q_indexes] == y[q_indexes]).all():
                q_id = q_id.item()
                self.known_questions.add(q_id)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._cfg.lr)


