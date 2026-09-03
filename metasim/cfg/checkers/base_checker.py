from __future__ import annotations

from metasim.cfg.objects import BaseObjCfg
from metasim.utils.configclass import configclass

try:
    from metasim.sim import BaseSimHandler
except:
    pass


@configclass
class BaseChecker:
    """Base class for all checkers. Checkers are used to check whether the task is executed successfully."""

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        """The code to run when the environment is reset."""
        pass

    def handles_state_reset(self) -> bool:
        """Return whether :meth:`reset` installs the simulator state itself.

        Most checkers only clear bookkeeping and therefore still need the
        generic trajectory state supplied by the environment wrapper.
        Staged tasks with curriculum snapshots can override this method so a
        reset does not write a generic state immediately before its own state.
        """
        return False

    def check(self, handler: BaseSimHandler):
        """Check whether the task is executed successfully."""
        import torch

        # log.warning("Checker not implemented, task will never succeed")
        return torch.zeros(handler.num_envs, dtype=torch.bool, device=handler.device)

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        """Get the viewers to be used for debugging the checker."""
        return []
