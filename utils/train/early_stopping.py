from pathlib import Path

import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path: Path = None):
        """
        Parameters
        ----------
        patience : int
            検証損失が改善しないエポック数の許容
        delta : float
            改善とみなす最小変化量
        path : Path
            ベストモデル保存先
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        if self.path:
            torch.save(model.state_dict(), self.path)
