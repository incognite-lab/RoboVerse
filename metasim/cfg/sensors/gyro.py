import numpy as np
from metasim.utils.configclass import configclass
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional

@configclass
class BaseGyroSensor:
    """Configuration for gyroscope sensor."""
    name: str = "gyro0"
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_to: Optional[str] = None
    mount_link: Optional[str] = None
    enabled: bool = True
    noise_mean: float = 0.0
    noise_std: float = 0.01
    noise_clip: float = 0.05

@configclass
class GyroSensorCfg(BaseGyroSensor):
    """Simple gyroscope sensor wrapper."""

    #def __init__(self, cfg: BaseGyroSensor, sim_handler):
    #    self.cfg = cfg
    #    self.sim = sim_handler

    def get_data(self, States, envs_ids) -> np.ndarray:
        if not self.enabled:
            return np.zeros(3, dtype=np.float32)
        # získání stavu linku z fyzikální simulace
        data = np.zeros((len(envs_ids), 3), dtype=np.float32)
        for env in envs_ids:
            link_idx=States[self.mount_to].body_names.index(self.mount_link)
            quats = States[self.mount_to].body_state[env, link_idx, 2:6].cpu().numpy()
            euler = R.from_quat(quats).as_euler("xyz", degrees=True).astype(np.float32)
            euler = np.abs(euler)
            data[env] = euler

        # link_idx = States[self.mount_to].body_names.index(self.mount_link)

        # # extrakce kvaternionu [x, y, z, w]
        # quats = States[self.mount_to].body_state[:, link_idx, 2:6].cpu().numpy()
        #        # převod na Eulerovy úhly (XYZ)
        # euler = R.from_quat(quats).as_euler("xyz", degrees=True).astype(np.float32)
        return data


        #return np.abs(euler)
