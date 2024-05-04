# %%
import torch
import torch.nn as nn
import lightning as pl
from omegaconf import DictConfig

class LogisticRegression(pl.LightningModule):
    def __init__(self, cfg : DictConfig):
        super().__init__()
        self._cfg = cfg.lr

        self.fc = nn.Linear(cfg._input_dim, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        pos_x, neg_x, y, *_ = batch
        x = neg_x - pos_x
        y_hat = self.fc(x).squeeze()
        loss = self.loss(y_hat, y)
        return loss

    def forward(self, x):
        return self.fc(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._cfg.lr)