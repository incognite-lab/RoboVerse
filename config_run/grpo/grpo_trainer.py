# grpo_trainer.py
from __future__ import annotations

import numpy as np
import torch


def get_student_inputs_from_states(metasim_env, device):
    states = metasim_env.env.handler.get_states()
    robot_name = metasim_env.scenario.robots[0].name

    # raw image pro storage
    imgs_u8 = states.cameras["camera0"].rgb.permute(0, 3, 1, 2).contiguous()

    # float image pro policy
    imgs_f32 = imgs_u8.to(device=device, dtype=torch.float32) / 255.0

    joints = states.robots[robot_name].joint_pos.to(device=device, dtype=torch.float32)

    return imgs_u8, imgs_f32, joints


def _empty_episode():
    return {
        "imgs": [],
        "joints": [],
        "actions": [],
        "old_logps": [],
        "rewards": [],
        "return": 0.0,
        "length": 0,
        "success": False,
    }


def collect_parallel_episodes(
    env,
    metasim_env,
    policy,
    device,
    num_episodes,
    max_steps,
    success_bonus=20.0,
):
    low_t = torch.as_tensor(env.action_space.low, device=device, dtype=torch.float32)
    high_t = torch.as_tensor(env.action_space.high, device=device, dtype=torch.float32)

    _ = env.reset()

    # zjistíme shape
    imgs_u8, imgs_f32, joints_f32 = get_student_inputs_from_states(metasim_env, device)

    num_envs = env.num_envs
    c, h, w = imgs_u8.shape[1:]
    num_joints = joints_f32.shape[1]
    num_actions = env.action_space.shape[0]
    assert joints_f32.shape[1] == num_joints, (joints_f32.shape[1], num_joints)
    # -----------------------------
    # aktivní epizody pro každý env
    # -----------------------------
    active_imgs = torch.empty((num_envs, max_steps, c, h, w), dtype=torch.uint8)
    active_joints = torch.empty((num_envs, max_steps, num_joints), dtype=torch.float16)
    active_actions = torch.empty((num_envs, max_steps, num_actions), dtype=torch.float16)
    active_old_logps = torch.empty((num_envs, max_steps), dtype=torch.float32)

    active_returns = torch.zeros(num_envs, dtype=torch.float32)
    active_lengths = torch.zeros(num_envs, dtype=torch.long)
    active_success = torch.zeros(num_envs, dtype=torch.bool)

    # -----------------------------
    # dokončené epizody
    # -----------------------------
    finished_imgs = torch.empty((num_episodes, max_steps, c, h, w), dtype=torch.uint8)
    finished_joints = torch.empty((num_episodes, max_steps, num_joints), dtype=torch.float16)
    finished_actions = torch.empty((num_episodes, max_steps, num_actions), dtype=torch.float16)
    finished_old_logps = torch.empty((num_episodes, max_steps), dtype=torch.float32)

    finished_returns = torch.empty(num_episodes, dtype=torch.float32)
    finished_lengths = torch.empty(num_episodes, dtype=torch.long)
    finished_success = torch.empty(num_episodes, dtype=torch.bool)

    finished_count = 0

    while finished_count < num_episodes:
        imgs_u8, imgs_f32, joints_f32 = get_student_inputs_from_states(metasim_env, device)

        with torch.no_grad():
            actions_t, _, _, _ = policy.act(imgs_f32, joints_f32, deterministic=False)
            actions_t = torch.max(torch.min(actions_t, high_t), low_t)
            old_logps_t, _, _ = policy.evaluate_actions(imgs_f32, joints_f32, actions_t)

        _, rewards, dones, infos = env.step(actions_t.detach().cpu().numpy())

        # jednorázový převod celé dávky na CPU
        imgs_cpu = imgs_u8.detach().cpu()  # uint8
        joints_cpu = joints_f32.detach().cpu().to(torch.float16)
        actions_cpu = actions_t.detach().cpu().to(torch.float16)
        old_logps_cpu = old_logps_t.detach().cpu()

        for i in range(num_envs):
            t = int(active_lengths[i].item())

            if t >= max_steps:
                raise RuntimeError(f"Episode in env {i} exceeded max_steps={max_steps}")

            step_reward = float(rewards[i])
            if bool(dones[i]) and bool(infos[i].get("is_success", False)):
                step_reward += success_bonus

            # uložíme krok do aktivního bufferu env i
            active_imgs[i, t].copy_(imgs_cpu[i])
            active_joints[i, t].copy_(joints_cpu[i])
            active_actions[i, t].copy_(actions_cpu[i])
            active_old_logps[i, t] = old_logps_cpu[i]

            active_returns[i] += step_reward
            active_lengths[i] += 1

            if bool(dones[i]):
                T = int(active_lengths[i].item())

                finished_imgs[finished_count, :T].copy_(active_imgs[i, :T])
                finished_joints[finished_count, :T].copy_(active_joints[i, :T])
                finished_actions[finished_count, :T].copy_(active_actions[i, :T])
                finished_old_logps[finished_count, :T].copy_(active_old_logps[i, :T])

                finished_returns[finished_count] = active_returns[i]
                finished_lengths[finished_count] = T
                finished_success[finished_count] = bool(infos[i].get("is_success", False))

                finished_count += 1

                # reset aktivního slotu pro env i
                active_returns[i] = 0.0
                active_lengths[i] = 0
                active_success[i] = False

                if finished_count >= num_episodes:
                    break

    # vrátíme views, ne další kopie
    episodes = []
    for e in range(finished_count):
        T = int(finished_lengths[e].item())
        episodes.append({
            "imgs": finished_imgs[e, :T],
            "joints": finished_joints[e, :T],
            "actions": finished_actions[e, :T],
            "old_logps": finished_old_logps[e, :T],
            "return": float(finished_returns[e].item()),
            "length": T,
            "success": bool(finished_success[e].item()),
        })

    return episodes


def build_grpo_batch(episodes, group_size):
    assert len(episodes) % group_size == 0, "num_episodes musí být násobek group_size"

    rng = np.random.default_rng()
    order = rng.permutation(len(episodes))
    episodes = [episodes[i] for i in order]

    imgs_all = []
    joints_all = []
    actions_all = []
    old_logps_all = []
    adv_all = []

    all_returns = []
    all_lengths = []
    all_success = []

    for start in range(0, len(episodes), group_size):
        group = episodes[start:start + group_size]
        returns = torch.tensor([ep["return"] for ep in group], dtype=torch.float32)

        mean_r = returns.mean()
        std_r = returns.std(unbiased=False)
        group_adv = (returns - mean_r) / (std_r + 1e-8)

        for ep, adv in zip(group, group_adv):
            T = ep["actions"].shape[0]

            imgs_all.append(ep["imgs"])            # CPU uint8
            joints_all.append(ep["joints"])        # CPU float16
            actions_all.append(ep["actions"])      # CPU float16
            old_logps_all.append(ep["old_logps"])  # CPU float32
            adv_all.append(torch.full((T,), adv.item(), dtype=torch.float32))

            all_returns.append(ep["return"])
            all_lengths.append(ep["length"])
            all_success.append(1.0 if ep["success"] else 0.0)

    batch = {
        "imgs": torch.cat(imgs_all, dim=0),            # pořád CPU
        "joints": torch.cat(joints_all, dim=0),        # pořád CPU
        "actions": torch.cat(actions_all, dim=0),      # pořád CPU
        "old_logps": torch.cat(old_logps_all, dim=0),  # pořád CPU
        "advantages": torch.cat(adv_all, dim=0),       # pořád CPU
    }

    stats = {
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "success_rate": float(np.mean(all_success)),
        "mean_length": float(np.mean(all_lengths)),
        "num_episodes": len(episodes),
    }
    return batch, stats


def grpo_update(
    policy,
    optimizer,
    batch,
    device,
    clip_eps=0.2,
    ent_coef=1e-3,
    epochs=4,
    minibatch_size=2048,
    max_grad_norm=1.0,
):
    N = batch["actions"].shape[0]

    stats = {
        "loss": [],
        "policy_loss": [],
        "entropy": [],
        "ratio_mean": [],
    }

    for _ in range(epochs):
        perm = torch.randperm(N)

        for start in range(0, N, minibatch_size):
            idx = perm[start:start + minibatch_size]

            # přesun jen minibatche na GPU
            imgs = batch["imgs"][idx].to(device=device, dtype=torch.float32) / 255.0
            joints = batch["joints"][idx].to(device=device, dtype=torch.float32)
            actions = batch["actions"][idx].to(device=device, dtype=torch.float32)
            old_logps = batch["old_logps"][idx].to(device=device, dtype=torch.float32)
            advantages = batch["advantages"][idx].to(device=device, dtype=torch.float32)

            new_logps, entropy, _ = policy.evaluate_actions(imgs, joints, actions)

            ratio = torch.exp(new_logps - old_logps)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy_mean = entropy.mean()
            loss = policy_loss - ent_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            stats["loss"].append(loss.item())
            stats["policy_loss"].append(policy_loss.item())
            stats["entropy"].append(entropy_mean.item())
            stats["ratio_mean"].append(ratio.mean().item())

    return {k: float(np.mean(v)) for k, v in stats.items()}
