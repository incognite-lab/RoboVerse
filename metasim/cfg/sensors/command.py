import numpy as np
from metasim.utils.configclass import configclass
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional
import torch

@configclass
class BaseCommand:
    """Configuration for Command."""
    name: str = "command0"
    cmd: torch.Tensor = np.zeros(3)
    actual_cmd: torch.Tensor = np.zeros((1,3))
@configclass
class CommandCfg(BaseCommand):
    """Simple commander wrapper."""
    def get_command(self) -> torch.Tensor:
        return self.actual_cmd
    def set_command(self, cmds: torch.Tensor) -> None:
        self.actual_cmd = cmds
