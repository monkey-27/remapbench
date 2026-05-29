"""
Intervention generators for RemapBench.

All appliers (single and composed) now return a normalized 6-tuple:
  (new_grid, new_goal, intervention_name, target_multihot, weak_action_change, extra_meta)

extra_meta is a dict. For action_change it carries cell/path-relevance metadata
(see DEFAULT_ACTION_META). For composed samples it additionally carries
"composed_pair" (the pair tuple). Non-action single interventions use defaults.
"""
import numpy as np
from collections import deque
from .env import (
    N_CHANNELS, CH_WALL, CH_START, CH_GOAL, CH_NUISANCE, CH_DISTRACTOR,
    CH_DOOR, ONE_WAY_CHANNELS, ACT_DR, ACT_DC,
    bfs_reachable_undirected, bfs_dist_undirected, build_transition, directed_path_len,
)

# Default action-change relevance metadata (used for all non-action samples).
DEFAULT_ACTION_META = {
    "action_cell_row": -1,
    "action_cell_col": -1,
    "action_cell_on_path": 0,
    "action_cell_near_path": 0,
    "action_path_action_changed": 0,
    "action_path_len_changed": 0,
}

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
        return g2, goal, "sensory_nuisance", MULTIHOT_SENSORY, 0, dict(DEFAULT_ACTION_META)

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
        return g2, new_goal, "goal_relocation", MULTIHOT_VALUE, 0, dict(DEFAULT_ACTION_META)

    # Relax path-change requirement
    for new_goal in candidates:
        path_after = _shortest_path_undirected(grid, start, new_goal)
        if path_after >= 1:
            g2 = _copy_grid(grid)
            g2[CH_GOAL, goal[0], goal[1]] = 0
            g2[CH_GOAL, new_goal[0], new_goal[1]] = 1
            return g2, new_goal, "goal_relocation", MULTIHOT_VALUE, 0, dict(DEFAULT_ACTION_META)

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
                    return g2, goal, "topology_change", MULTIHOT_MAP, 0, dict(DEFAULT_ACTION_META)

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
                    return g2, goal, "topology_change", MULTIHOT_MAP, 0, dict(DEFAULT_ACTION_META)

    raise ValueError("topology_change: could not find valid topology modification")


# ---------------------------------------------------------------------------
# 4. Action change (strengthened: path-relevant, path-action aware)
# ---------------------------------------------------------------------------

def _orig_path_action(path, r, c):
    """
    Return the action index taken at (r,c) along path toward the next cell,
    or -1 if (r,c) is not an interior path cell.
    """
    for idx, p in enumerate(path):
        if p == (r, c) and idx + 1 < len(path):
            nr, nc = path[idx + 1]
            dr, dc = nr - r, nc - c
            for a in range(4):
                if ACT_DR[a] == dr and ACT_DC[a] == dc:
                    return a
    return -1


def _action_meta(r, c, d, pl_after, pl_before_dir, path, on_path_flag, near_flag):
    """Build action-change relevance metadata for an accepted placement."""
    orig_act = _orig_path_action(path, r, c) if on_path_flag else -1
    path_act_changed = 1 if (orig_act >= 0 and d != orig_act) else 0
    path_len_changed = 1 if (pl_after >= 0 and pl_before_dir >= 0
                             and abs(pl_after - pl_before_dir) >= 1) else 0
    return {
        "action_cell_row": int(r),
        "action_cell_col": int(c),
        "action_cell_on_path": 1 if on_path_flag else 0,
        "action_cell_near_path": 1 if near_flag else 0,
        "action_path_action_changed": path_act_changed,
        "action_path_len_changed": path_len_changed,
    }


def apply_action_change(grid, start, goal, rng):
    """
    Add one-way behavior on a path-relevant free cell.

    Strong (weak=0) conditions:
      - Cell on shortest path AND directed path length changes ≥ 1, OR
      - Cell on shortest path AND the one-way direction differs from the
        original path action at that cell (forcing a detour), OR
      - Cell near path AND directed path length changes ≥ 1.

    Weak (weak=1) fallback:
      - Near-path cell with no path-length change, OR
      - Any free cell (last resort).

    Returns extra_meta with cell/path-relevance fields (see DEFAULT_ACTION_META).
    """
    H, W = grid.shape[1], grid.shape[2]
    path = _shortest_path_cells(grid, start, goal)
    path_set = set(path)
    path_before = len(path) - 1 if path else -1

    reachable = bfs_reachable_undirected(grid, start)
    T_before = build_transition(grid)
    pl_before_dir = directed_path_len(T_before, start, goal, W)

    near_path = {
        (r + dr, c + dc)
        for (r, c) in path_set
        for dr in range(-2, 3) for dc in range(-2, 3)
        if 0 <= r + dr < H and 0 <= c + dc < W
    }
    on_path  = [p for p in reachable
                if p in path_set and p not in (start, goal) and not grid[CH_WALL, p[0], p[1]]]
    near_only = [p for p in reachable
                 if p in near_path and p not in path_set and p not in (start, goal)
                 and not grid[CH_WALL, p[0], p[1]]]
    all_free  = [p for p in reachable
                 if p not in near_path and p not in (start, goal)
                 and not grid[CH_WALL, p[0], p[1]]]

    rng.shuffle(on_path); rng.shuffle(near_only); rng.shuffle(all_free)

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
            return g2, T2, pl, d
        return None

    # Tier 1: on path — strong if path_len changes OR path action changes
    for (r, c) in on_path:
        res = _try_cell(r, c)
        if res is None:
            continue
        g2, T2, pl_after, d = res
        path_len_changed   = abs(pl_after - path_before) >= 1
        orig_act = _orig_path_action(path, r, c)
        path_act_changed   = (orig_act >= 0 and d != orig_act)
        if path_len_changed or path_act_changed:
            meta = _action_meta(r, c, d, pl_after, pl_before_dir, path, True, True)
            return g2, goal, "action_change", MULTIHOT_ACTION, 0, meta  # strong

    # Tier 2: near path — strong only if path_len changes
    for (r, c) in near_only:
        res = _try_cell(r, c)
        if res is None:
            continue
        g2, T2, pl_after, d = res
        if abs(pl_after - path_before) >= 1:
            meta = _action_meta(r, c, d, pl_after, pl_before_dir, path, False, True)
            return g2, goal, "action_change", MULTIHOT_ACTION, 0, meta  # strong

    # Tier 3: on path — weak (path action didn't change in a behaviorally strong way)
    for (r, c) in on_path:
        res = _try_cell(r, c)
        if res is None:
            continue
        g2, T2, pl_after, d = res
        meta = _action_meta(r, c, d, pl_after, pl_before_dir, path, True, True)
        return g2, goal, "action_change", MULTIHOT_ACTION, 1, meta  # weak

    # Tier 4: near path — weak (no path-length change)
    for (r, c) in near_only:
        res = _try_cell(r, c)
        if res is None:
            continue
        g2, T2, pl_after, d = res
        meta = _action_meta(r, c, d, pl_after, pl_before_dir, path, False, True)
        return g2, goal, "action_change", MULTIHOT_ACTION, 1, meta  # weak

    # Tier 5: any free cell — weak fallback
    for (r, c) in all_free:
        res = _try_cell(r, c)
        if res is None:
            continue
        g2, T2, pl_after, d = res
        meta = _action_meta(r, c, d, pl_after, pl_before_dir, path, False, False)
        return g2, goal, "action_change", MULTIHOT_ACTION, 1, meta  # weak

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


def _merge_action_meta(m1, m2):
    """Prefer whichever sub-intervention meta carries a real action cell."""
    if m1.get("action_cell_row", -1) != -1:
        return dict(m1)
    if m2.get("action_cell_row", -1) != -1:
        return dict(m2)
    return dict(DEFAULT_ACTION_META)


def apply_composed(grid, start, goal, rng, pair_idx=None):
    if pair_idx is None:
        pair_idx = int(rng.integers(len(COMPOSED_PAIRS)))
    pair_idx = pair_idx % len(COMPOSED_PAIRS)
    pair = COMPOSED_PAIRS[pair_idx]
    multihot = COMPOSED_MULTIHOT[pair]

    fn1 = _APPLIERS[pair[0]]
    g1, goal1, _, _, weak1, meta1 = fn1(grid, start, goal, rng)

    fn2 = _APPLIERS[pair[1]]
    for _ in range(MAX_TRIES):
        try:
            g2, goal2, _, _, weak2, meta2 = fn2(g1, start, goal1, rng)
            extra = _merge_action_meta(meta1, meta2)
            extra["composed_pair"] = pair
            return g2, goal2, "composed", multihot, max(weak1, weak2), extra
        except ValueError:
            pass

    raise ValueError(f"composed {pair}: second intervention failed after retries")


def apply_intervention(grid, start, goal, rng, intervention_type, composed_pair_idx=None):
    """
    Dispatch to the correct intervention. All branches return a normalized 6-tuple:
      (new_grid, new_goal, name, multihot, weak_action_change, extra_meta)
    For composed, extra_meta additionally contains "composed_pair".
    """
    if intervention_type == "composed":
        return apply_composed(grid, start, goal, rng, composed_pair_idx)
    fn = _APPLIERS.get(intervention_type)
    if fn is None:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
    return fn(grid, start, goal, rng)
