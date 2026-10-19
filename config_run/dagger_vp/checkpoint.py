"""DAgger checkpoints, keeping plain student weights compatible with evaluation."""

import os
import re

import torch
from loguru import logger as log
from .beta_schedule import beta_at_step


def save_dagger_checkpoint(path, student, optimizer, next_step, beta):
    torch.save(student.state_dict(), path)
    # Store training state separately so existing evaluation/deployment can still
    # load the .pth file directly as a state_dict.
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "next_step": next_step,
            "beta": beta,
        },
        f"{path}.training.pth",
    )


def load_dagger_checkpoint(path, student, optimizer, device, config):
    """Return the next global iteration and mixing ratio for resumed training."""
    training_path = f"{path}.training.pth"
    if os.path.isfile(training_path):
        checkpoint = torch.load(training_path, map_location=device, weights_only=True)
        student.load_state_dict(checkpoint["student_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        next_step = int(checkpoint["next_step"])
        # The configured schedule owns beta, including migration from old
        # checkpoints which stored an exponentially decayed mixing ratio.
        beta = beta_at_step(config, next_step)
        log.info("Restored DAgger step {}; recalculated beta={} from configured schedule", next_step, beta)
        return next_step, beta

    student.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    match = re.fullmatch(r"student_model_step_(\d+)\.pth", os.path.basename(path))
    # Historical filenames use the zero-based index of the completed iteration.
    next_step = config.get("dagger_resume_step", int(match.group(1)) + 1 if match else 0)
    if not isinstance(next_step, int) or isinstance(next_step, bool) or next_step < 0:
        raise ValueError("dagger_resume_step must be a non-negative integer")
    beta = beta_at_step(config, next_step)
    log.warning(
        "Legacy DAgger checkpoint: optimizer starts fresh; next step={} and beta={}. "
        "Set dagger_resume_step for a final or renamed legacy checkpoint.",
        next_step, beta,
    )
    return next_step, beta
