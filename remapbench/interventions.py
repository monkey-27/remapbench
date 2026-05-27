"""
Intervention generators for RemapBench.
Each applier returns a 5-tuple:
  (new_grid, new_goal, intervention_name, target_multihot, weak_action_change)
Composed returns a 6-tuple:
  (new_grid, new_goal, "composed", target_multihot, composed_pair, weak_action_change)
"""
import numpy as np
from collections import deque
from .env import (
    N_CHANNELS, CH_WALL, CH_START, CH_GOAL, CH_NUISANCE, CH_DISTRACTOR,
    CH_DOOR, ONE_WAY_CHANNELS, ACT_DR, ACT_DC,
    bfs_reachable_undirected, bfs_dist_undirected, build_transition, directed_path_len,
)

# target_multihot indices: [sensory_update, value_remap, map_remap, action_remap]
MULTIHOT_SENSORY = [1, 0, 0, 0]   # sensory pathway updates; predictive map stable
MULTIHOT_VALUE   = [0, 1, 0, 0]   # value/reward readout updates
MULTIHOT_MAP     = [0, 0, 1, 0]   # predictive map / future-state structure updates
MULTIHOT_ACTION  = [0, 0, 0, 1]   # action-conditioned / affordance pathway updates

INTERVENTION_IDS = {
    "sensory_nuisance": 0,
    "goal_relocation":  1,
    "topology_change":  2,
    "action_change":    3,
    "composed":         4,
}

THRESH_SENSORY = 0.02
THRESH_VALUE   = 0.02
THRESH_FUTURE  = 0.02
THRESH_ACTION  = 0.02
MIN_PATH_CHANGE_GOAL = 3
MIN_PATH_CHANGE_TOPO = 2
MAX_TRIES = 80


def _copy_grid(grid):
    return grid.copy()


def _shortest_path_undirected(grid, start, goal):
    d = bfs_dist_undirected(grid, start)
    return d.get(goal, -1)


def _reachable_undirected(grid, start, goal):
    return goal in bfs_reachable_undirected(grid, start)


def _shortest_path_cells(grid, start, goal):
    H, W = grid.shape[1], grid.shape[2]
    walls = grid[CH_WALL]
    prev = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path, cur = [], goal
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for i in range(4):
            nr, nc = r + ACT_DR[i], c + ACT_DC[i]
            if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc] and (nr, nc) not in prev:
                prev[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return []


# ---------------------------------------------------------------------------
# 1. Sensory nuisance
# ---------------------------------------------------------------------------

def apply_sensory_nuisance(grid, start, goal, rng):
    """Flip nuisance/distractor bits. Transitions and goal unchanged."""
    H, W = grid.shape[1], grid.shape[2]
    T = build_transition(grid)
    free = [(r, c) for r in range(H) for c in range(W) if not grid[CH_WALL, r, c]]

    for _ in range(MAX_TRIES):
        g2 = _copy_grid(grid)
        n_flip = max(1, int(rng.integers(3, max(4, len(free) // 2))))
        idxs = rng.choice(len(free), size=min(n_flip, len(free)), replace=False)
        for i in idxs:
            r, c = free[i]
            if (r, c) in (start, goal):
                continue
            ch = CH_NUISANCE if rng.random() < 0.6 else CH_DISTRACTOR
            g2[ch, r, c] ^= 1

        T2 = build_transition(g2)
        if not np.array_equal(T, T2):
            continue
        diff = np.abs(
            g2[[CH_NUISANCE, CH_DISTRACTOR]].astype(float) -
            grid[[CH_NUISANCE, CH_DISTRACTOR]].astype(float)
        ).mean()
        if diff < THRESH_SENSORY:
            continue
        return g2, goal, "sensory_nuisance", MULTIHOT_SENSORY, 0

    raise ValueError("sensory_nuisance: could not find valid change")


# ---------------------------------------------------------------------------
# 2. Goal relocation
# ---------------------------------------------------------------------------

def apply_goal_relocation(grid, start, goal, rng):
    """Move goal to a different reachable cell. Transitions unchanged."""
    H, W = grid.shape[1], grid.shape[2]
    reachable = bfs_reachable_undirected(grid, start)
    path_before = _shortest_path_undirected(grid, start, goal)

    dists_from_start = bfs_dist_undirected(grid, start)
    candidates = [
        (r, c) for (r, c) in reachable
        if (r, c) != start and (r, c) != goal and not grid[CH_WALL, r, c]
    ]
    candidates.sort(key=lambda p: -dists_from_start.get(p, 0))

    for new_goal in candidates:
        path_after = _shortest_path_undirected(grid, start, new_goal)
        if path_after < 1:
            continue
        if abs(path_after - path_before) < MIN_PATH_CHANGE_GOAL:
            continue
        g2 = _copy_grid(grid)
        g2[CH_GOAL, goal[0], goal[1]] = 0
        g2[CH_GOAL, new_goal[0], new_goal[1]] = 1
        return g2, new_goal, "goal_relocation", MULTIHOT_VALUE, 0

    # Relax path-change requirement
    for new_goal in candidates:
        path_after = _shortest_path_undirected(grid, start, new_goal)
        if path_after >= 1:
            g2 = _copy_grid(grid)
            g2[CH_GOAL, goal[0], goal[1]] = 0
            g2[CH_GOAL, new_goal[0], new_goal[1]] = 1
            return g2, new_goal, "goal_relocation", MULTIHOT_VALUE, 0

    raise ValueError("goal_relocation: no valid new goal found")


# ---------------------------------------------------------------------------
# 3. Topology change
# ---------------------------------------------------------------------------

def apply_topology_change(grid, start, goal, rng):
    """Block a path cell or open a wall to create a shortcut."""
    H, W = grid.shape[1], grid.shape[2]
    path = _shortest_path_cells(grid, start, goal)
    path_before = len(path) - 1 if path else -1

    strategies = ["block", "open"]
    rng.shuffle(strategies)

    for relaxed in [False, True]:
        for strategy in strategies:
            if strategy == "block":
                interior = [p for p in path if p not in (start, goal)]
                rng.shuffle(interior)
                for cell in interior:
                    g2 = _copy_grid(grid)
                    g2[CH_WALL, cell[0], cell[1]] = 1
                    for ch in (CH_NUISANCE, CH_DISTRACTOR, CH_DOOR):
                        g2[ch, cell[0], cell[1]] = 0
                    if not _reachable_undirected(g2, start, goal):
                        continue
                    path_after = _shortest_path_undirected(g2, start, goal)
                    if path_after < 1:
                        continue
                    if not relaxed and abs(path_after - path_before) < MIN_PATH_CHANGE_TOPO:
                        continue
                    return g2, goal, "topology_change", MULTIHOT_MAP, 0

            elif strategy == "open":
                walls_rc = [
                    (r, c) for r in range(1, H-1) for c in range(1, W-1)
                    if grid[CH_WALL, r, c]
                ]
                rng.shuffle(walls_rc)
                for wr, wc in walls_rc[:40]:
                    g2 = _copy_grid(grid)
                    g2[CH_WALL, wr, wc] = 0
                    g2[CH_DOOR, wr, wc] = 1
                    if not _reachable_undirected(g2, start, goal):
                        continue
                    path_after = _shortest_path_undirected(g2, start, goal)
                    if path_after < 1:
                        continue
                    if not relaxed and abs(path_after - path_before) < MIN_PATH_CHANGE_TOPO:
                        continue
                    return g2, goal, "topology_change", MULTIHOT_MAP, 0

    raise ValueError("topology_change: could not find valid topology modification")


# ---------------------------------------------------------------------------
# 4. Action change (strengthened: path-relevant)
# ---------------------------------------------------------------------------

def apply_action_change(grid, start, goal, rng):
    """
    Add one-way behavior on a path-relevant free cell.
    Prefers cells on the shortest path; falls back to cells within Manhattan
    distance 2 of the path, then all reachable cells (marked as weak).
    Returns weak_action_change=1 if path-length relevance could not be confirmed.
    """
    H, W = grid.shape[1], grid.shape[2]
    path = _shortest_path_cells(grid, start, goal)
    path_set = set(path)
    path_before = len(path) - 1 if path else -1

    reachable = bfs_reachable_undirected(grid, start)
    T_before = build_transition(grid)

    # Build candidate tiers: (cells_on_path, near_path, all_free)
    near_path = {
        (r + dr, c + dc)
        for (r, c) in path_set
        for dr in range(-2, 3) for dc in range(-2, 3)
        if 0 <= r + dr < H and 0 <= c + dc < W
    }
    on_path = [
        p for p in reachable
        if p in path_set and p not in (start, goal) and not grid[CH_WALL, p[0], p[1]]
    ]
    near_only = [
        p for p in reachable
        if p in near_path and p not in path_set and p not in (start, goal)
        and not grid[CH_WALL, p[0], p[1]]
    ]
    all_free = [
        p for p in reachable
        if p not in path_set and p not in near_path and p not in (start, goal)
        and not grid[CH_WALL, p[0], p[1]]
    ]

    rng.shuffle(on_path)
    rng.shuffle(near_only)
    rng.shuffle(all_free)

    def _try_cell(r, c):
        if any(grid[ch, r, c] for ch in ONE_WAY_CHANNELS):
            return None
        directions = list(range(4))
        rng.shuffle(directions)
        for d in directions:
            g2 = _copy_grid(grid)
            for ch in ONE_WAY_CHANNELS:
                g2[ch, r, c] = 0
            g2[ONE_WAY_CHANNELS[d], r, c] = 1
            T2 = build_transition(g2)
            pl = directed_path_len(T2, start, goal, W)
            if pl < 0:
                continue
            if not (T_before != T2).any():
                continue
            return g2, T2, pl
        return None

    # Tier 1: on path — only accept if path_len changes OR cell is on path
    for (r, c) in on_path:
        result = _try_cell(r, c)
        if result is None:
            continue
        g2, T2, pl_after = result
        # On path is always relevant (strong)
        return g2, goal, "action_change", MULTIHOT_ACTION, 0

    # Tier 2: near path — require path_len change
    for (r, c) in near_only:
        result = _try_cell(r, c)
        if result is None:
            continue
        g2, T2, pl_after = result
        if abs(pl_after - path_before) >= 1:
            return g2, goal, "action_change", MULTIHOT_ACTION, 0

    # Tier 3: near path — no path_len requirement (near-path is still behaviorally close)
    for (r, c) in near_only:
        result = _try_cell(r, c)
        if result is None:
            continue
        g2, T2, pl_after = result
        return g2, goal, "action_change", MULTIHOT_ACTION, 0

    # Tier 4: fallback — any reachable cell, marked as weak
    for (r, c) in all_free:
        result = _try_cell(r, c)
        if result is None:
            continue
        g2, T2, pl_after = result
        return g2, goal, "action_change", MULTIHOT_ACTION, 1  # weak

    raise ValueError("action_change: could not find valid one-way placement")


# ---------------------------------------------------------------------------
# 5. Composed interventions
# ---------------------------------------------------------------------------

COMPOSED_PAIRS = [
    ("goal_relocation",  "topology_change"),
    ("sensory_nuisance", "action_change"),
    ("goal_relocation",  "action_change"),
    ("sensory_nuisance", "topology_change"),
]

COMPOSED_MULTIHOT = {
    ("goal_relocation",  "topology_change"):  [0, 1, 1, 0],
    ("sensory_nuisance", "action_change"):    [1, 0, 0, 1],
    ("goal_relocation",  "action_change"):    [0, 1, 0, 1],
    ("sensory_nuisance", "topology_change"):  [1, 0, 1, 0],
}

_APPLIERS = {
    "sensory_nuisance": apply_sensory_nuisance,
    "goal_relocation":  apply_goal_relocation,
    "topology_change":  apply_topology_change,
    "action_change":    apply_action_change,
}


def apply_composed(grid, start, goal, rng, pair_idx=None):
    if pair_idx is None:
        pair_idx = int(rng.integers(len(COMPOSED_PAIRS)))
    pair_idx = pair_idx % len(COMPOSED_PAIRS)
    pair = COMPOSED_PAIRS[pair_idx]
    multihot = COMPOSED_MULTIHOT[pair]

    fn1 = _APPLIERS[pair[0]]
    g1, goal1, _, _, weak1 = fn1(grid, start, goal, rng)

    fn2 = _APPLIERS[pair[1]]
    for _ in range(MAX_TRIES):
        try:
            g2, goal2, _, _, weak2 = fn2(g1, start, goal1, rng)
            return g2, goal2, "composed", multihot, pair, max(weak1, weak2)
        except ValueError:
            pass

    raise ValueError(f"composed {pair}: second intervention failed after retries")


def apply_intervention(grid, start, goal, rng, intervention_type, composed_pair_idx=None):
    """
    Dispatch to correct intervention.
    Single: returns (new_grid, new_goal, name, multihot, weak_action_change).
    Composed: returns (new_grid, new_goal, name, multihot, pair_tuple, weak_action_change).
    """
    if intervention_type == "sensory_nuisance":
        return apply_sensory_nuisance(grid, start, goal, rng)
    elif intervention_type == "goal_relocation":
        return apply_goal_relocation(grid, start, goal, rng)
    elif intervention_type == "topology_change":
        return apply_topology_change(grid, start, goal, rng)
    elif intervention_type == "action_change":
        return apply_action_change(grid, start, goal, rng)
    elif intervention_type == "composed":
        return apply_composed(grid, start, goal, rng, composed_pair_idx)
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
