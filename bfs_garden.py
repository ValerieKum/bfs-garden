from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.viz import (PALETTE, card_figure, caption, save_animation,
                        use_headless_if_saving)

# ---- tunables -------------------------------------------------------------
GRID_N = 31            # garden is GRID_N x GRID_N cells (odd looks best)
MODE = "bfs"           # "bfs" = explore in rings, "dfs" = dive down one path
CELLS_PER_FRAME = 3    # cells revealed per animation frame
HOLD_FRAMES = 25       # frames to hold the finished path before looping


def make_garden(n: int) -> np.ndarray:
    """Randomized DFS (recursive backtracker). 0 = open path, 1 = hedge."""
    g = np.ones((n, n), dtype=np.uint8)
    stack = [(1, 1)]
    g[1, 1] = 0
    while stack:
        y, x = stack[-1]
        nbrs = []
        for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            ny, nx = y + dy, x + dx
            if 1 <= ny < n - 1 and 1 <= nx < n - 1 and g[ny, nx] == 1:
                nbrs.append((ny, nx, dy, dx))
        if not nbrs:
            stack.pop()
            continue
        ny, nx, dy, dx = nbrs[np.random.randint(len(nbrs))]
        g[y + dy // 2, x + dx // 2] = 0   # knock the hedge down between
        g[ny, nx] = 0
        stack.append((ny, nx))
    return g


def search(grid, start, goal, mode):
    """Return (visit_order, dist, path).

    BFS pulls from the front of a queue, so it fans out in even rings and the
    first time it reaches the fish it has the shortest path. DFS pulls from the
    back, so it dives down one corridor as far as it can before backing up.
    """
    n = grid.shape[0]
    frontier = deque([start])
    came = {start: None}
    dist = {start: 0}
    visit_order = []
    while frontier:
        cur = frontier.popleft() if mode == "bfs" else frontier.pop()
        visit_order.append(cur)
        if cur == goal:
            break
        y, x = cur
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n and grid[ny, nx] == 0 \
                    and (ny, nx) not in came:
                came[(ny, nx)] = cur
                dist[(ny, nx)] = dist[cur] + 1
                frontier.append((ny, nx))
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = came.get(c)
    path.reverse()
    return visit_order, dist, path


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="OUT.mp4")
    args = ap.parse_args()
    use_headless_if_saving(args.save)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    grid = make_garden(GRID_N)
    start, goal = (1, 1), (GRID_N - 2, GRID_N - 2)
    visit_order, dist, path = search(grid, start, goal, MODE)

    bg = hex_to_rgb(PALETTE["bg"])
    hedge = hex_to_rgb(PALETTE["grid"])
    cool = hex_to_rgb(PALETTE["cool"])
    cyan = hex_to_rgb(PALETTE["cyan"])
    pink = hex_to_rgb(PALETTE["pink"])
    hot = hex_to_rgb(PALETTE["hot"])

    base = np.where(grid[..., None] == 1, hedge, bg).astype(np.float64)
    max_d = max(dist.values()) or 1

    n_search = len(visit_order)
    search_frames = (n_search + CELLS_PER_FRAME - 1) // CELLS_PER_FRAME
    path_frames = len(path)
    total = search_frames + path_frames + HOLD_FRAMES

    fig, ax = card_figure()
    img = ax.imshow(base / 255.0, interpolation="nearest")
    caption(ax, "bfs · cat finds the fish" if MODE == "bfs"
            else "dfs · cat dives down one path")

    def render(frame):
        canvas = base.copy()
        revealed = min(frame * CELLS_PER_FRAME, n_search)
        # wavefront: each visited cell tinted by its ring distance from the cat
        for i in range(revealed):
            y, x = visit_order[i]
            t = dist[(y, x)] / max_d
            canvas[y, x] = cool * (1 - t) + cyan * t
        # the shortest path lights up once the fish is found
        if frame >= search_frames:
            steps = min(frame - search_frames + 1, path_frames)
            for i in range(steps):
                y, x = path[i]
                t = i / max(1, path_frames - 1)
                canvas[y, x] = pink * (1 - t) + hot * t
        canvas[start] = hot    # the cat
        canvas[goal] = pink    # the fish
        img.set_array(np.clip(canvas / 255.0, 0.0, 1.0))
        return (img,)

    anim = FuncAnimation(fig, render, frames=total, interval=40, blit=True)

    if args.save:
        save_animation(anim, args.save, fps=25)
    else:
        plt.show()


if __name__ == "__main__":
    main()
