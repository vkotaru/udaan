"""Resize GIF files — reduce resolution and frame count for README/docs.

Usage:
    python -m udaan.utils.resize_gif .media/quadrotor.gif --width 480 --max-frames 50
    python -m udaan.utils.resize_gif .media/*.gif --width 480 --fps 15
"""

import argparse
import os

import imageio.v3 as iio
import numpy as np
from PIL import Image


def resize_gif(path, width=480, max_frames=50, fps=15, output=None):
    """Resize a GIF file.

    Args:
        path: input GIF path.
        width: target width in pixels (height scales proportionally).
        max_frames: maximum number of frames to keep.
        fps: output frame rate.
        output: output path (defaults to overwriting input).
    """
    frames = iio.imread(path)
    h, w = frames[0].shape[:2]
    height = int(width * h / w)

    step = max(1, len(frames) // max_frames)
    resized = []
    for i in range(0, len(frames), step):
        img = Image.fromarray(frames[i]).resize((width, height), Image.LANCZOS)
        resized.append(np.array(img))

    out_path = output or path
    iio.imwrite(out_path, resized, duration=1000 // fps, loop=0)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"{out_path}: {len(frames)} -> {len(resized)} frames, {w}x{h} -> {width}x{height}, {size_mb:.1f}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize GIF files")
    parser.add_argument("files", nargs="+", help="GIF files to resize")
    parser.add_argument("--width", type=int, default=480, help="Target width (default: 480)")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames (default: 50)")
    parser.add_argument("--fps", type=int, default=15, help="Output FPS (default: 15)")
    args = parser.parse_args()

    for f in args.files:
        resize_gif(f, width=args.width, max_frames=args.max_frames, fps=args.fps)
