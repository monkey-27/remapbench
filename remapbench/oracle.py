"""Oracle computations: future occupancy, value map, action delta, error metrics."""
import numpy as np
from collections import deque
from .env import (
    CH_WALL, CH_NUISANCE, CH_DISTRACTOR, N_CHANNELS,
    build_transition, directed_path_len,
)


def _build_uniform_P(T, N):
    """P[s, s'] = (1/4) * sum_a 1[T[a,s]=s']. Uniform-over-attempted-actions policy."""
    P = np.zeros((N, N), dtype=np.float64)
    for a in range(4):
        for s in range(N):
            P[s, int(T[a, s])] += 0.25
    return P


def compute_future(grid, start, T, gamma=0.95):
    """
    Successor representation for start state under uniform attempted actions.
    Policy: each action is attempted with probability 1/4; invalid moves (walls/
    boundary) result in the agent staying — the distribution is over attempted
    actions, not over valid ones. This matches build_transition conventions.
    Solved via numpy.linalg.solve on the transposed SR system (not explicit inverse).
    Returns float32 [H,W], normalized by max.
    """
    H, W = grid.shape[1], grid.shape[2]
    N = H * W
    walls = grid[CH_WALL].flatten()

    P = _build_uniform_P(T, N)
    A = np.eye(N, dtype=np.float64) - gamma * P
    s0 = start[0] * W + start[1]
    e = np.zeros(N, dtype=np.float64)
    e[s0] = 1.0
    # Solve A^T x = e_s0 → x = M[s0, :] (row s0 of the successor rep matrix)
    fut = np.linalg.solve(A.T, e)
    fut[walls.astype(bool)] = 0.0
    fut = np.clip(fut, 0.0, None)
    mx = fut.max()
    if mx > 0:
        fut /= mx
    return fut.reshape(H, W).astype(np.float32)


def compute_value(grid, goal, T, gamma=0.95):
    """
    Value map via reverse BFS from goal under directed transitions.
    value[r,c] = gamma^dist_to_goal; unreachable = 0; goal = 1.
    Returns float32 [H,W].
    """
    H, W = grid.shape[1], grid.shape[2]
    N = H * W
    walls = grid[CH_WALL].flatten()
    sg = goal[0] * W + goal[1]

    # Reverse adjacency: rev_T[s'] = states s that can reach s' in one step
    rev_T = [[] for _ in range(N)]
    for a in range(4):
        for s in range(N):
            if walls[s]:
                continue
            ns = int(T[a, s])
            if ns != s:
                rev_T[ns].append(s)

    dist = {sg: 0}
    q = deque([sg])
    while q:
        sp = q.popleft()
        for s in rev_T[sp]:
            if s not in dist:
                dist[s] = dist[sp] + 1
                q.append(s)

    value = np.zeros(N, dtype=np.float32)
    for s, d in dist.items():
        value[s] = gamma ** d
    return value.reshape(H, W)


def compute_action_delta(T_before, T_after, H, W):
    """
    action_delta[a, r, c] = 1.0 if T_before[a,s] != T_after[a,s] for s = r*W+c.
    Returns float32 [4, H, W].
    """
    return (T_before != T_after).astype(np.float32).reshape(4, H, W)


def compute_all_oracle(grid, start, goal, gamma=0.95):
    """Compute full oracle bundle for a single grid."""
    T = build_transition(grid)
    H, W = grid.shape[1], grid.shape[2]
    future = compute_future(grid, start, T, gamma)
    value = compute_value(grid, goal, T, gamma)
    path_len = directed_path_len(T, start, goal, W)
    return {"T": T, "future": future, "value": value, "path_len": path_len}


def build_sample_arrays(
    grid_before, grid_after, start, goal_before, goal_after,
    intervention_id, target_multihot, gamma=0.95,
    layout_id=0, sample_id=0, layout_seed=0,
    intervention_seed=0, intervention_pair_id=-1,
    weak_action_change=0,
):
    """
    Assemble all sample fields by running oracles on before/after grids.
    Returns a dict suitable for np.savez_compressed.
    """
    H, W = grid_before.shape[1], grid_before.shape[2]

    T_b = build_transition(grid_before)
    T_a = build_transition(grid_after)

    future_b = compute_future(grid_before, start, T_b, gamma)
    future_a = compute_future(grid_after, start, T_a, gamma)
    value_b  = compute_value(grid_before, goal_before, T_b, gamma)
    value_a  = compute_value(grid_after,  goal_after,  T_a, gamma)
    act_delta = compute_action_delta(T_b, T_a, H, W)

    path_b = directed_path_len(T_b, start, goal_before, W)
    path_a = directed_path_len(T_a, start, goal_after,  W)
    if path_b < 0: path_b = -1
    if path_a < 0: path_a = -1

    # Nuisance error: L1 diff of nuisance_visual + distractor channels only
    nuisance_err = float(np.abs(
        grid_before[[CH_NUISANCE, CH_DISTRACTOR]].astype(np.float32) -
        grid_after[[CH_NUISANCE, CH_DISTRACTOR]].astype(np.float32)
    ).mean())

    # Full visual error: L1 diff across all observable grid channels (0..N_CHANNELS)
    full_visual_err = float(np.abs(
        grid_before.astype(np.float32) -
        grid_after.astype(np.float32)
    ).mean())

    # sensory_error: alias of nuisance_error for backward compatibility
    sensory_err = nuisance_err

    future_err = float(np.abs(future_a - future_b).mean())
    value_err  = float(np.abs(value_a  - value_b).mean())

    # action_error: fraction of free-cell × action pairs that changed
    free_mask = (grid_before[CH_WALL].flatten() == 0).astype(np.float32)
    n_free = free_mask.sum()
    action_err = float(
        (act_delta.reshape(4, -1) * free_mask[None, :]).sum() / (4.0 * n_free + 1e-8)
    )

    return dict(
        # Grids
        before_grid=grid_before,
        after_grid=grid_after,
        # Positions
        start_xy=np.array([start[1], start[0]], dtype=np.int64),
        goal_before_xy=np.array([goal_before[1], goal_before[0]], dtype=np.int64),
        goal_after_xy=np.array([goal_after[1],   goal_after[0]],  dtype=np.int64),
        # Labels
        intervention_id=np.int64(intervention_id),
        target_multihot=np.array(target_multihot, dtype=np.uint8),
        # Oracle maps
        future_before=future_b,
        future_after=future_a,
        delta_future=(future_a - future_b),
        value_before=value_b,
        value_after=value_a,
        delta_value=(value_a - value_b),
        action_delta=act_delta,
        # Scalar errors
        nuisance_error=np.float32(nuisance_err),
        full_visual_error=np.float32(full_visual_err),
        sensory_error=np.float32(sensory_err),   # alias = nuisance_error
        future_error=np.float32(future_err),
        value_error=np.float32(value_err),
        action_error=np.float32(action_err),
        # Path lengths
        path_len_before=np.int64(path_b),
        path_len_after=np.int64(path_a),
        # Metadata
        layout_id=np.int64(layout_id),
        sample_id=np.int64(sample_id),
        layout_seed=np.int64(layout_seed),
        intervention_seed=np.int64(intervention_seed),
        intervention_pair_id=np.int64(intervention_pair_id),
        weak_action_change=np.int64(weak_action_change),
    )
