"""Grid environment: constants, random generation, transition model."""
import numpy as np
from collections import deque

# ---------------------------------------------------------------------------
# Channel indices
# ---------------------------------------------------------------------------
N_CHANNELS = 10
CH_WALL = 0
CH_START = 1
CH_GOAL = 2
CH_NUISANCE = 3
CH_DISTRACTOR = 4
CH_DOOR = 5
CH_ONE_WAY_UP = 6
CH_ONE_WAY_DOWN = 7
CH_ONE_WAY_LEFT = 8
CH_ONE_WAY_RIGHT = 9

ONE_WAY_CHANNELS = [CH_ONE_WAY_UP, CH_ONE_WAY_DOWN, CH_ONE_WAY_LEFT, CH_ONE_WAY_RIGHT]

# Actions: 0=up(-r), 1=down(+r), 2=left(-c), 3=right(+c)
ACT_DR = [-1, 1, 0, 0]
ACT_DC = [0, 0, -1, 1]
ACT_NAMES = ["up", "down", "left", "right"]


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def make_grid(H, W, rng, wall_prob=0.25, min_free=10, min_path=6, max_tries=400):
    """
    Generate a random connected gridworld.
    Returns (grid [C,H,W] uint8, start_pos (r,c), goal_pos (r,c), reachable set).
    """
    for _ in range(max_tries):
        g = np.zeros((N_CHANNELS, H, W), dtype=np.uint8)
        g[CH_WALL, 0, :] = 1
        g[CH_WALL, -1, :] = 1
        g[CH_WALL, :, 0] = 1
        g[CH_WALL, :, -1] = 1
        g[CH_WALL, 1:-1, 1:-1] = (rng.random((H - 2, W - 2)) < wall_prob).astype(np.uint8)

        free = list(zip(*np.where(g[CH_WALL] == 0)))
        if len(free) < min_free:
            continue

        start_pos = free[int(rng.integers(len(free)))]
        reachable = bfs_reachable_undirected(g, start_pos)
        if len(reachable) < min_free:
            continue

        reachable_list = list(reachable)
        dists = bfs_dist_undirected(g, start_pos)
        far = [(r, c) for (r, c) in reachable_list if dists.get((r, c), 0) >= min_path]
        if not far:
            continue

        goal_pos = far[int(rng.integers(len(far)))]

        g[CH_START, start_pos[0], start_pos[1]] = 1
        g[CH_GOAL, goal_pos[0], goal_pos[1]] = 1

        # Random nuisance/distractor decorations on free non-special cells
        for r, c in reachable_list:
            if (r, c) in (start_pos, goal_pos):
                continue
            if rng.random() < 0.15:
                g[CH_NUISANCE, r, c] = 1
            if rng.random() < 0.10:
                g[CH_DISTRACTOR, r, c] = 1

        return g, start_pos, goal_pos, reachable

    raise RuntimeError(f"Could not generate valid grid after {max_tries} tries "
                       f"(H={H}, W={W}, wall_prob={wall_prob}, min_path={min_path})")


# ---------------------------------------------------------------------------
# BFS utilities (undirected – ignore one-way for layout purposes)
# ---------------------------------------------------------------------------

def bfs_reachable_undirected(grid, start):
    H, W = grid.shape[1], grid.shape[2]
    walls = grid[CH_WALL]
    visited = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for i in range(4):
            nr, nc = r + ACT_DR[i], c + ACT_DC[i]
            if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited


def bfs_dist_undirected(grid, start):
    H, W = grid.shape[1], grid.shape[2]
    walls = grid[CH_WALL]
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for i in range(4):
            nr, nc = r + ACT_DR[i], c + ACT_DC[i]
            if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc] and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


# ---------------------------------------------------------------------------
# Directed transition model
# ---------------------------------------------------------------------------

def build_transition(grid):
    """
    T[a, s] = next_state for action a from state s.
    One-way cells: all actions forced to the one-way direction (first active wins).
    Walls/boundary: agent stays in place.
    """
    H, W = grid.shape[1], grid.shape[2]
    N = H * W
    walls = grid[CH_WALL]
    T = np.empty((4, N), dtype=np.int32)

    for r in range(H):
        for c in range(W):
            s = r * W + c
            if walls[r, c]:
                T[:, s] = s
                continue
            forced = -1
            for ai, ch in enumerate(ONE_WAY_CHANNELS):
                if grid[ch, r, c]:
                    forced = ai
                    break
            for a in range(4):
                act = forced if forced >= 0 else a
                nr, nc = r + ACT_DR[act], c + ACT_DC[act]
                if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc]:
                    T[a, s] = nr * W + nc
                else:
                    T[a, s] = s

    return T


def directed_path_len(T, start, goal, W):
    """BFS shortest path length under T. Returns -1 if unreachable."""
    s0 = start[0] * W + start[1]
    sg = goal[0] * W + goal[1]
    if s0 == sg:
        return 0
    dist = {s0: 0}
    q = deque([s0])
    while q:
        s = q.popleft()
        for a in range(4):
            ns = int(T[a, s])
            if ns == s:
                continue
            if ns == sg:
                return dist[s] + 1
            if ns not in dist:
                dist[ns] = dist[s] + 1
                q.append(ns)
    return -1
