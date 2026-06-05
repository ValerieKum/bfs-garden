# Breadth-First Search: A Cat Finds the Fish

Breadth-first search explores a garden level by level as a cat finds the shortest path to the fish.

Part of my portfolio of small, from-scratch visualisations of computer-science ideas. Built on numpy and matplotlib, so every moving part is visible.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python bfs_garden.py                  # live animated window
python bfs_garden.py --save out.gif   # export a looping GIF
python bfs_garden.py --save out.mp4   # smaller file, best for the web (needs ffmpeg)
```
