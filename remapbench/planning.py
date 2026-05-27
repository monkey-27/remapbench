"""
Planning metrics for RemapBench evaluation.
Greedy value-following policy + planning regret vs oracle shortest path.
"""
import numpy as np
from .env import build_transition, directed_path_len


def greedy_path_from_value(grid_after, start_xy, goal_after_xy, value_map_pred, max_steps=100):
    """
    Greedy policy: move to the non-stuck neighbor with highest predicted value.
    Loop detection stops the agent on revisit.

    Args:
        grid_after:     [C,H,W] uint8 numpy grid
        start_xy:       [2] int64  (x=col, y=row) as stored in dataset
        goal_after_xy:  [2] int64  (x=col, y=row) as stored in dataset
        value_map_pred: [H,W] float numpy predicted value map
        max_steps:      max steps before declaring failure

    Returns:
        dict: success (bool), steps (int), reached_goal (bool)
    """
    sr, sc = int(start_xy[1]), int(start_xy[0])
    gr, gc = int(goal_after_xy[1]), int(goal_after_xy[0])
    H, W   = grid_after.shape[1], grid_after.shape[2]

    T = build_transition(grid_after)  # [4, N]

    r, c = sr, sc
    if (r, c) == (gr, gc):
        return {"success": True, "steps": 0, "reached_goal": True}

    visited = {(r, c)}
    for step in range(1, max_steps + 1):
        s = r * W + c
        best_val, best_ns = -1e9, -1
        for a in range(4):
            ns = int(T[a, s])
            if ns == s:
                continue
            nr2, nc2 = ns // W, ns % W
            v = float(value_map_pred[nr2, nc2])
            if v > best_val:
                best_val, best_ns = v, ns

        if best_ns < 0:
            break  # all actions blocked

        nr, nc = best_ns // W, best_ns % W
        if (nr, nc) == (gr, gc):
            return {"success": True, "steps": step, "reached_goal": True}
        if (nr, nc) in visited:
            break  # loop
        visited.add((nr, nc))
        r, c = nr, nc

    return {"success": False, "steps": max_steps, "reached_goal": False}


def compute_planning_metrics(
    grids_after, start_xys, goal_after_xys, value_maps_pred,
    path_lens_after, max_steps=50, penalty=50,
):
    """
    Aggregate planning metrics over a batch.

    Args:
        grids_after:       list of [C,H,W] numpy arrays
        start_xys:         list of [2] int arrays
        goal_after_xys:    list of [2] int arrays
        value_maps_pred:   list of [H,W] float arrays
        path_lens_after:   list of int (oracle directed path lengths; -1=unreachable)
        max_steps:         greedy policy step budget
        penalty:           step count assigned to failed episodes for regret

    Returns:
        dict with planning_success_rate, mean_step_regret, failure_rate, n_samples
    """
    successes, regrets = [], []
    for grid, sxy, gxy, vmap, oracle_len in zip(
        grids_after, start_xys, goal_after_xys, value_maps_pred, path_lens_after
    ):
        if oracle_len < 0:
            continue  # skip unreachable goals

        result = greedy_path_from_value(grid, sxy, gxy, vmap, max_steps)
        successes.append(int(result["reached_goal"]))
        regret = (result["steps"] - oracle_len) if result["reached_goal"] else (penalty - oracle_len)
        regrets.append(max(0, regret))

    if not successes:
        return {"planning_success_rate": float("nan"), "mean_step_regret": float("nan"),
                "failure_rate": float("nan"), "n_samples": 0}

    return {
        "planning_success_rate": float(np.mean(successes)),
        "mean_step_regret":      float(np.mean(regrets)),
        "failure_rate":          float(1.0 - np.mean(successes)),
        "n_samples":             len(successes),
    }
