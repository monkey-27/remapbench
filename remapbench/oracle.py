"""Oracle computations: future occupancy, value map, action delta."""
import numpy as np
from collections import deque
from .env import CH_WALL, CH_NUISANCE, CH_DISTRACTOR, ONE_WAY_CHANNELS, build_transition, directed_path_len


def _build_uniform_P(T, N):
    """P[s, s'] = average over actions of 1[T[a,s]=s']. Dense float64."""
    P = np.zeros((N, N), dtype=np.float64)
    for a in range(4):
        for s in range(N):
            P[s, int(T[a, s])] += 0.25
    return P


def compute_future(grid, start, T, gamma=0.95):
    """
    Successor representation row for start: M[start,:] = (I - gamma*P)^{-1}[start,:].
    Uses numpy.linalg.solve on the transposed system (not explicit inverse).
    Returns float32 [H,W] normalized by max.
    """
    H, W = grid.shape[1], grid.shape[2]
    N = H * W
    walls = grid[CH_WALL].flatten()

    P = _build_uniform_P(T, N)
    A = np.eye(N, dtype=np.float64) - gamma * P  # (I - gamma*P)
    # M = A^{-1}. To get row s0: solve A^T x = e_s0.
    s0 = start[0] * W + start[1]
    e = np.zeros(N, dtype=np.float64)
    e[s0] = 1.0
    fut = np.linalg.solve(A.T, e)

    # Zero out wall states and negatives (numerical noise)
    fut[walls.astype(bool)] = 0.0
    fut = np.clip(fut, 0.0, None)
    mx = fut.max()
    if mx > 0:
        fut /= mx
    return fut.reshape(H, W).astype(np.float32)


def compute_value(grid, goal, T, gamma=0.95):
    """
    Value map via reverse BFS from goal under directed transitions.
    value[r,c] = gamma^dist_to_goal, unreachable = 0, goal = 1.
    Returns float32 [H,W].
    """
    H, W = grid.shape[1], grid.shape[2]
    N = H * W
    walls = grid[CH_WALL].flatten()
    sg = goal[0] * W + goal[1]

    # Reverse adjacency: rev_T[s'] = set of s that can go to s' (non-self only)
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
    action_delta[a, r, c] = 1.0 if T_before[a,s] != T_after[a,s], else 0.
    Returns float32 [4, H, W].
    """
    changed = (T_before != T_after).astype(np.float32)  # [4, N]
    return changed.reshape(4, H, W)


def compute_all_oracle(grid, start, goal, gamma=0.95):
    """
    Compute full oracle bundle for a (grid, start, goal) triple.
    Returns dict with future, value, T (to compare before/after).
    """
    T = build_transition(grid)
    H, W = grid.shape[1], grid.shape[2]
    future = compute_future(grid, start, T, gamma)
    value = compute_value(grid, goal, T, gamma)
    path_len = directed_path_len(T, start, goal, W)
    return {"T": T, "future": future, "value": value, "path_len": path_len}


def build_sample_arrays(
    grid_before, grid_after, start, goal_before, goal_after,
    intervention_id, target_multihot, gamma=0.95
):
    """
    Assemble all sample fields by running oracles on before/after grids.
    Returns a dict suitable for saving.
    """
    H, W = grid_before.shape[1], grid_before.shape[2]

    T_b = build_transition(grid_before)
    T_a = build_transition(grid_after)

    future_b = compute_future(grid_before, start, T_b, gamma)
    future_a = compute_future(grid_after, start, T_a, gamma)
    value_b = compute_value(grid_before, goal_before, T_b, gamma)
    value_a = compute_value(grid_after, goal_after, T_a, gamma)
    act_delta = compute_action_delta(T_b, T_a, H, W)

    path_b = directed_path_len(T_b, start, goal_before, W)
    path_a = directed_path_len(T_a, start, goal_after, W)
    if path_b < 0:
        path_b = -1
    if path_a < 0:
        path_a = -1

    # Scalar error metrics
    nuisance_channels = [CH_NUISANCE, CH_DISTRACTOR]
    sensory_err = float(np.abs(
        grid_before[nuisance_channels].astype(np.float32) -
        grid_after[nuisance_channels].astype(np.float32)
    ).mean())
    # future/value errors are averaged over all cells (walls = 0, which is fine)
    future_err = float(np.abs(future_a - future_b).mean())
    value_err = float(np.abs(value_a - value_b).mean())
    # action_error: fraction of free (non-wall) (action, cell) pairs that changed
    # averaging over all N*4 dilutes signal by wall fraction; restrict to free cells
    free_mask = (grid_before[CH_WALL].flatten() == 0).astype(np.float32)  # [N]
    n_free = free_mask.sum()
    if n_free > 0:
        action_err = float(
            (act_delta.reshape(4, -1) * free_mask[None, :]).sum() / (4.0 * n_free)
        )
    else:
        action_err = 0.0

    return dict(
        before_grid=grid_before,
        after_grid=grid_after,
        start_xy=np.array([start[1], start[0]], dtype=np.int64),  # (x=col, y=row)
        goal_before_xy=np.array([goal_before[1], goal_before[0]], dtype=np.int64),
        goal_after_xy=np.array([goal_after[1], goal_after[0]], dtype=np.int64),
        intervention_id=np.int64(intervention_id),
        target_multihot=np.array(target_multihot, dtype=np.uint8),
        future_before=future_b,
        future_after=future_a,
        delta_future=(future_a - future_b),
        value_before=value_b,
        value_after=value_a,
        delta_value=(value_a - value_b),
        action_delta=act_delta,
        sensory_error=np.float32(sensory_err),
        future_error=np.float32(future_err),
        value_error=np.float32(value_err),
        action_error=np.float32(action_err),
        path_len_before=np.int64(path_b),
        path_len_after=np.int64(path_a),
    )
