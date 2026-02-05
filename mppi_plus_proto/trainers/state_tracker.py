import torch
import torch.nn as nn

from pathlib import Path


class ModelStateTracker:

    def __init__(self,
                 checkpoint_file: Path,
                 mode: str = "min"):
        assert mode in ["min", "max"], "Mode must be either 'min' or 'max'"
        self._mode = mode
        if self._mode == "min":
            self._best_value = float("inf")
        else:
            self._best_value = float("-inf")
        self._checkpoint_file = checkpoint_file

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def checkpoint_file(self) -> Path:
        return self._checkpoint_file

    def update(self, model: nn.Module, value: float):
        should_update = (
            (self._mode == "min" and value < self._best_value) or
            (self._mode == "max" and value > self._best_value)
        )
        if should_update:
            if self._checkpoint_file.exists():
                self._checkpoint_file.unlink()
            torch.save(model.state_dict(), self._checkpoint_file)
