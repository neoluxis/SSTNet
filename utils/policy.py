"""
Simple early-stopping policy for training loops.

Provides an EasyStop class that monitors a scalar metric (e.g. mAP)
and signals to stop training when the metric does not improve for
`patience` consecutive checks.
"""
import math


class EasyStop:
    """Early stopping helper.

    Args:
        patience (int): number of epochs with no improvement to wait before stopping.
        min_delta (float): minimum change to qualify as an improvement.
        mode (str): 'max' (default) to monitor metrics where larger is better (e.g. mAP),
                    or 'min' for metrics where smaller is better (e.g. loss).
        baseline (float|None): optional initial best value.
    """
    def __init__(self, patience=10, min_delta=0.0, mode='max', baseline=None):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        if baseline is not None:
            self.best = float(baseline)
        else:
            self.best = -math.inf if mode == 'max' else math.inf
        self.num_bad_epochs = 0

    def step(self, current):
        """Call at the end of an epoch with the current monitored value.

        Returns True if training should stop (no improvement for `patience` steps).
        """
        if current is None:
            return False

        current = float(current)
        if self.mode == 'max':
            improved = (current - self.best) > self.min_delta
        else:
            improved = (self.best - current) > self.min_delta

        if improved:
            self.best = current
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience

    def reset(self):
        self.num_bad_epochs = 0


def build_policy(name="easystop", **kwargs):
    if name.lower() == "easystop":
        return EasyStop(**kwargs)
    raise ValueError(f"Unknown policy: {name}")
