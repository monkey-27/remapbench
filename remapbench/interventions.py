"""
Intervention generators for RemapBench.
Each function takes (grid, start, goal, rng) and returns
(new_grid, new_goal, intervention_name, target_multihot)
or raises ValueError on rejection.
"""
import numpy as np
from collections import deque
from .env import (
    N_CHANNELS, CH_WALL, CH_START, CH_GOAL, CH_NUISANCE, CH_DISTRACTOR,
    CH_DOOR, ONE_WAY_CHANNELS, ACT_DR, ACT_DC,
    bfs_reachable_undirected, bfs_dist_undirected, build_transition, directed_path_len,
)

# target_multihot indices: [no_remap, value_remap, map_remap, action_remap]
MULTIHOT_NO_REMAP = [1, 0, 0, 0]
MULTIHOT_VALUE = [0, 1, 0, 0]
MULTIHOT_MAP = [0, 0, 1, 0]
MULTIHOT_ACTION = [0, 0, 0, 1]

INTERVENTION_IDS = {
    "sensory_nuisance": 0,
    "goal_relocation": 1,
    "topology_change": 2,
    "action_change": 3,
    "composed": 4,
}

# Thresholds used for rejection sampling
THRESH_SENSORY = 0.02
THRESH_VALUE = 0.02
THRESH_FUTURE = 0.02
THRESH_ACTION = 0.02
MIN_PATH_CHANGE_GOAL = 3
MIN_PATH_CHANGE_TOPO = 2
MAX_TRIES = 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_grid(grid):
    return grid.copy()


def _shortest_path_undirected(grid, start, goal):
    d = bfs_dist_undirected(grid, start)
    return d.get(goal, -1)


def _reachable_undirected(grid, start, goal):
    reach = bfs_reachable_undirected(grid, start)
    return goal in reach


def _apply_nuisance_change(grid, rng):
    """Flip nuisance/distractor bits on free cells. Returns modified grid."""
    g = _copy_grid(grid)
    H, W = g.shape[1], g.shape[2]
    free = list(zip(*np.where(g[CH_WALL] == 0)))
    if not free:
        raise ValueError("No free cells")
    # Flip a random subset of nuisance/distractor cells
    n_flip = max(1, int(rng.integers(3, min(10, len(free) // 2) + 1)))
    idxs = rng.choice(len(free), size=min(n_flip, len(free)), replace=False)
    for i in idxs:
        r, c = free[i]
        if (r, c) == (g[CH_START].argmax() // W, g[CH_START].argmax() % W):
            continue
        if (r, c) == (g[CH_GOAL].argmax() // W, g[CH_GOAL].argmax() % W):
            continue
        ch = CH_NUISANCE if rng.random() < 0.6 else CH_DISTRACTOR
        g[ch, r, c] ^= 1
    return g


# ---------------------------------------------------------------------------
# 1. Sensory nuisance
# ---------------------------------------------------------------------------

def apply_sensory_nuisance(grid, start, goal, rng):
    """Change nuisance_visual / distractor channels. Transitions unchanged."""
    from .oracle import compute_future, compute_value, build_transition, compute_action_delta
    T = build_transition(grid)
    H, W = grid.shape[1], grid.shape[2]

    for _ in range(MAX_TRIES):
        g2 = _apply_nuisance_change(grid, rng)
        # Verify: nuisance channels changed but transitions identical
        T2 = build_transition(g2)
        if not np.array_equal(T, T2):
            continue
        # Check sensory error is meaningfully non-zero
        diff = np.abs(
            g2[[CH_NUISANCE, CH_DISTRACTOR]].astype(float) -
            grid[[CH_NUISANCE, CH_DISTRACTOR]].astype(float)
        ).mean()
        if diff < THRESH_SENSORY:
            continue
        return g2, goal, "sensory_nuisance", MULTIHOT_NO_REMAP
    raise ValueError("sensory_nuisance: could not find valid change")


# ---------------------------------------------------------------------------
# 2. Goal relocation
# ---------------------------------------------------------------------------

def apply_goal_relocation(grid, start, goal, rng):
    """Move goal to a different reachable cell. Transitions unchanged."""
    H, W = grid.shape[1], grid.shape[2]
    T = build_transition(grid)
    reachable = bfs_reachable_undirected(grid, start)
    path_before = _shortest_path_undirected(grid, start, goal)

    candidates = [
        (r, c) for (r, c) in reachable
        if (r, c) != start and (r, c) != goal and not grid[CH_WALL, r, c]
    ]
    if not candidates:
        raise ValueError("goal_relocation: no candidate positions")

    rng.shuffle(candidates := list(candidates))
    # prefer far candidates
    dists_from_start = bfs_dist_undirected(grid, start)
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
        return g2, new_goal, "goal_relocation", MULTIHOT_VALUE

    # Relax path-change requirement
    for new_goal in candidates:
        path_after = _shortest_path_undirected(grid, start, new_goal)
        if path_after < 1:
            continue
        g2 = _copy_grid(grid)
        g2[CH_GOAL, goal[0], goal[1]] = 0
        g2[CH_GOAL, new_goal[0], new_goal[1]] = 1
        return g2, new_goal, "goal_relocation", MULTIHOT_VALUE

    raise ValueError("goal_relocation: no valid new goal found")


# ---------------------------------------------------------------------------
# 3. Topology change
# ---------------------------------------------------------------------------

def _shortest_path_cells(grid, start, goal):
    """BFS shortest path: returns list of cells or []."""
    H, W = grid.shape[1], grid.shape[2]
    walls = grid[CH_WALL]
    prev = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            path = []
            cur = goal
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


def apply_topology_change(grid, start, goal, rng):
    """Block a path cell OR open a wall to create a shortcut. Preserves reachability."""
    H, W = grid.shape[1], grid.shape[2]
    path = _shortest_path_cells(grid, start, goal)
    path_before = len(path) - 1 if path else -1

    strategies = ["block", "open"]
    rng.shuffle(strategies)

    for strategy in strategies:
        if strategy == "block":
            # Block an interior path cell (not start/goal)
            interior = [p for p in path if p not in (start, goal)]
            rng.shuffle(interior)
            for cell in interior:
                g2 = _copy_grid(grid)
                g2[CH_WALL, cell[0], cell[1]] = 1
                g2[CH_DOOR, cell[0], cell[1]] = 0
                g2[CH_NUISANCE, cell[0], cell[1]] = 0
                g2[CH_DISTRACTOR, cell[0], cell[1]] = 0
                # Verify start-goal still reachable
                if not _reachable_undirected(g2, start, goal):
                    continue
                path_after = _shortest_path_undirected(g2, start, goal)
                if path_after < 1:
                    continue
                if abs(path_after - path_before) < MIN_PATH_CHANGE_TOPO:
                    continue
                return g2, goal, "topology_change", MULTIHOT_MAP

        elif strategy == "open":
            # Open a wall cell adjacent to free cells to create a shortcut
            walls_rc = list(zip(*np.where(grid[CH_WALL] == 1)))
            # Exclude border walls
            interior_walls = [
                (r, c) for (r, c) in walls_rc
                if 0 < r < H - 1 and 0 < c < W - 1
            ]
            rng.shuffle(interior_walls)
            for wr, wc in interior_walls[:30]:
                g2 = _copy_grid(grid)
                g2[CH_WALL, wr, wc] = 0
                g2[CH_DOOR, wr, wc] = 1
                if not _reachable_undirected(g2, start, goal):
                    continue
                path_after = _shortest_path_undirected(g2, start, goal)
                if path_after < 1:
                    continue
                if abs(path_after - path_before) < MIN_PATH_CHANGE_TOPO:
                    continue
                return g2, goal, "topology_change", MULTIHOT_MAP

    # Relax path-change requirement
    for strategy in ["block", "open"]:
        if strategy == "block":
            interior = [p for p in path if p not in (start, goal)]
            rng.shuffle(interior)
            for cell in interior:
                g2 = _copy_grid(grid)
                g2[CH_WALL, cell[0], cell[1]] = 1
                if not _reachable_undirected(g2, start, goal):
                    continue
                path_after = _shortest_path_undirected(g2, start, goal)
                if path_after >= 1:
                    return g2, goal, "topology_change", MULTIHOT_MAP
        else:
            walls_rc = list(zip(*np.where(grid[CH_WALL] == 1)))
            interior_walls = [(r, c) for (r, c) in walls_rc if 0 < r < H-1 and 0 < c < W-1]
            rng.shuffle(interior_walls)
            for wr, wc in interior_walls[:20]:
                g2 = _copy_grid(grid)
                g2[CH_WALL, wr, wc] = 0
                g2[CH_DOOR, wr, wc] = 1
                if _reachable_undirected(g2, start, goal):
                    path_after = _shortest_path_undirected(g2, start, goal)
                    if path_after >= 1:
                        return g2, goal, "topology_change", MULTIHOT_MAP

    raise ValueError("topology_change: could not find valid topology modification")


# ---------------------------------------------------------------------------
# 4. Action change
# ---------------------------------------------------------------------------

def apply_action_change(grid, start, goal, rng):
    """Add one-way tile on a reachable non-wall cell near the path."""
    H, W = grid.shape[1], grid.shape[2]
    path = _shortest_path_cells(grid, start, goal)
    reachable = bfs_reachable_undirected(grid, start)

    # Candidates: reachable cells not on special positions
    candidates = [
        (r, c) for (r, c) in reachable
        if (r, c) not in (start, goal) and not grid[CH_WALL, r, c]
    ]
    # Prefer path-adjacent cells first
    path_set = set(path)
    on_path = [p for p in candidates if p in path_set]
    off_path = [p for p in candidates if p not in path_set]
    ordered = on_path + off_path
    rng.shuffle(on_path)
    rng.shuffle(off_path)
    ordered = on_path + off_path

    T_before = build_transition(grid)

    for r, c in ordered:
        # Skip if already one-way
        if any(grid[ch, r, c] for ch in ONE_WAY_CHANNELS):
            continue
        # Pick a direction that doesn't immediately trap the agent
        directions = list(range(4))
        rng.shuffle(directions)
        for d in directions:
            g2 = _copy_grid(grid)
            # Clear all one-way on this cell first
            for ch in ONE_WAY_CHANNELS:
                g2[ch, r, c] = 0
            g2[ONE_WAY_CHANNELS[d], r, c] = 1
            # Verify start-goal still reachable (directed)
            T2 = build_transition(g2)
            pl = directed_path_len(T2, start, goal, W)
            if pl < 0:
                continue
            # Verify action delta is non-trivial
            changed = (T_before != T2).sum()
            if changed == 0:
                continue
            return g2, goal, "action_change", MULTIHOT_ACTION

    raise ValueError("action_change: could not find valid one-way placement")


# ---------------------------------------------------------------------------
# 5. Composed interventions
# ---------------------------------------------------------------------------

COMPOSED_PAIRS = [
    ("goal_relocation", "topology_change"),
    ("sensory_nuisance", "action_change"),
    ("goal_relocation", "action_change"),
    ("sensory_nuisance", "topology_change"),
]

COMPOSED_MULTIHOT = {
    ("goal_relocation", "topology_change"):  [0, 1, 1, 0],
    ("sensory_nuisance", "action_change"):   [1, 0, 0, 1],
    ("goal_relocation", "action_change"):    [0, 1, 0, 1],
    ("sensory_nuisance", "topology_change"): [1, 0, 1, 0],
}

_APPLIERS = {
    "sensory_nuisance": apply_sensory_nuisance,
    "goal_relocation": apply_goal_relocation,
    "topology_change": apply_topology_change,
    "action_change": apply_action_change,
}


def apply_composed(grid, start, goal, rng, pair_idx=None):
    """Apply two interventions. pair_idx selects from COMPOSED_PAIRS (or random)."""
    if pair_idx is None:
        pair_idx = int(rng.integers(len(COMPOSED_PAIRS)))
    pair = COMPOSED_PAIRS[pair_idx % len(COMPOSED_PAIRS)]
    multihot = COMPOSED_MULTIHOT[pair]

    # Apply first intervention
    fn1 = _APPLIERS[pair[0]]
    g1, goal1, _, _ = fn1(grid, start, goal, rng)

    # Apply second intervention on top
    fn2 = _APPLIERS[pair[1]]
    for _ in range(MAX_TRIES):
        try:
            g2, goal2, _, _ = fn2(g1, start, goal1, rng)
            return g2, goal2, "composed", multihot, pair
        except ValueError:
            pass

    raise ValueError(f"composed {pair}: second intervention failed after retries")


def apply_intervention(grid, start, goal, rng, intervention_type, composed_pair_idx=None):
    """
    Dispatch to the correct intervention.
    Returns (new_grid, new_goal, intervention_name, target_multihot[, composed_pair]).
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
