"""Offline, user-owned model training and accelerator packaging.

Nothing in this package is imported by the service.  Training dependencies,
local history, and compiler toolchains stay in a separate producer process;
the service receives only immutable artifacts and a schema-valid binding.
"""

from .contracts import TrainingError, TrainingReport, TrainingSpec
from .forecast import ForecastTrainer
from .installation import InstalledTraining, install_training

__all__ = [
    "ForecastTrainer",
    "InstalledTraining",
    "TrainingError",
    "TrainingReport",
    "TrainingSpec",
    "install_training",
]
