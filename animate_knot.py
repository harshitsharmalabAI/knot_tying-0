"""Movie of a knot run (3-D, side and along-B views)."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

parser = argparse.ArgumentParser()
parser.add_argument("frames", nargs="?", default="tension_knot_frames.npz")
parser.add_argument("--out", default=None, help="output file (.mp4 or .gif)")
parser.add_argument("--every", type=int, default=1, help="use every n-th frame")
parser.add_argument("--fps", type=int, default=20, help="20 fps = real time")
parser.add_argument("--dpi", type=int, default=90)
parser.add_argument("--still", type=int, default=None, help="save this one frame as a PNG")
args = parser.parse_args()

data = np.load(args.frames)
positions, times, stage_of = data["pos"], data["t"], data["stage"]
b_tips, b_axes, a_positions, hand_nodes = data["b_tip"], data["b_u"], data["a_pos"], data["hand_g"]
gripping, wound_gaps = data["grip"], data["wound"]
labels = [str(s) for s in data["labels"]]
stitch, n_elements = data["bite"], int(data["n"])
b_length, b_radius = float(data["b_len"]), float(data["b_r"])
thread_radius, jaw_reach = float(data["r_rod"]), float(data["jaw"])
left_exit, right_exit, edge_bite = int(data["wound_iL"]), int(data["wound_iR"]), float(data["edge_bite"])
shown = np.arange(0, len(times), max(1, args.every))
print(f"{len(times)} frames, t = {times[0]:.2f} .. {times[-1]:.2f} s, "
      f"stages {stage_of.min()}-{stage_of.max()}; rendering {len(shown)}")

working = np.concatenate([positions[i][:, :int(np.floor(hand_nodes[i])) + 1] for i in shown[::5]], axis=1)
lo = np.percentile(working, 1, axis=1) - 0.03
hi = np.percentile(working, 99, axis=1) + 0.03
jaws = b_tips + jaw_reach * b_axes
instruments = np.concatenate([a_positions[np.all(np.isfinite(a_positions), axis=1)], b_tips, jaws]).T
lo = np.minimum(lo, instruments.min(axis=1) - 0.02)
hi = np.maximum(hi, instruments.max(axis=1) + 0.02)
lo[1] = min(lo[1], -0.01)
hi[1] = min(hi[1], float(np.nanmax(b_tips[:, 1])) + 0.08)

fig = plt.figure(figsize=(16, 8))
grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.3, wspace=0.15)
ax3d = fig.add_subplot(grid[:, 0], projection="3d")
ax_side = fig.add_subplot(grid[0, 1])
ax_end = fig.add_subplot(grid[1, 1])
title = fig.suptitle("", fontsize=13)

ax3d.set_xlim(lo[0], hi[0]); ax3d.set_ylim(lo[2], hi[2]); ax3d.set_zlim(lo[1], hi[1])
ax3d.set_xlabel("x (along B)"); ax3d.set_ylabel("z"); ax3d.set_zlabel("y (up)")
ax3d.view_init(elev=22, azim=-60)
try:
    ax3d.set_box_aspect((hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1]))
except Exception:
    pass
gx, gz = np.meshgrid([lo[0], hi[0]], [lo[2], hi[2]])
ax3d.plot_surface(gx, gz, np.full_like(gx, -thread_radius), color="#e8d9c8", alpha=0.35, linewidth=0)

for ax, (xlabel, ylabel, xi, yi) in ((ax_side, ("x (along B)", "y (up)", 0, 1)),
                                     (ax_end, ("z", "y (up)", 2, 1))):
    ax.set_xlim(lo[xi], hi[xi]); ax.set_ylim(lo[yi], hi[yi]); ax.set_aspect("equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    ax.axhline(-thread_radius, color="#c8a882", lw=2)
ax_side.set_title("side view", fontsize=10)
ax_end.set_title("end view (looking along B)", fontsize=10)


def make_lines(ax, in_3d=False):
    new = (lambda **kw: ax.plot([], [], [], **kw)[0]) if in_3d else (lambda **kw: ax.plot([], [], **kw)[0])
    return dict(tail=new(color="darkorange", lw=3), stitch=new(color="black", lw=4),
                long=new(color="tab:blue", lw=2), reserve=new(color="tab:blue", lw=1, ls="--", alpha=0.5),
                b=new(color="grey", lw=7 if in_3d else 9, alpha=0.6, solid_capstyle="butt"),
                jaw=new(color="red", marker="o", ms=6, ls=""), a=new(color="tab:green", marker="o", ms=9, ls=""))


lines_3d, lines_side, lines_end = make_lines(ax3d, True), make_lines(ax_side), make_lines(ax_end)
for key, name in (("tail", "tail"), ("long", "long strand"), ("reserve", "reserve in A's hand"),
                  ("a", "A"), ("b", "instrument B")):
    lines_3d[key].set_label(name)
ax3d.legend(loc="upper left", fontsize=8)
flaps = []
b_end_on = plt.Circle((0, 0), b_radius, color="grey", alpha=0.6)
ax_end.add_patch(b_end_on)


def set_3d(line, pts):
    pts = np.array(pts, dtype=float)
    if pts.size:
        outside = np.any((pts < lo[:, None] - 1e-9) | (pts > hi[:, None] + 1e-9), axis=0)
        pts[:, outside] = np.nan
    line.set_data(pts[0], pts[2])
    line.set_3d_properties(pts[1])


def draw(i):
    p = positions[i].astype(float)
    hand = int(np.floor(hand_nodes[i]))
    tip, axis = b_tips[i], b_axes[i]
    none = np.zeros((3, 0))
    parts = dict(tail=p[:, :stitch[0] + 1], stitch=p[:, stitch], long=p[:, stitch[-1]:hand + 1],
                 reserve=p[:, hand:] if hand < n_elements else none,
                 b=np.column_stack([tip, tip - b_length * axis]),
                 jaw=(tip + jaw_reach * axis)[:, None] if gripping[i] else none,
                 a=a_positions[i][:, None] if np.all(np.isfinite(a_positions[i])) else none)
    for key, pts in parts.items():
        set_3d(lines_3d[key], pts)
        lines_side[key].set_data(pts[0], pts[1])
        lines_end[key].set_data(pts[2], pts[1])
    b_end_on.center = (tip[2], tip[1])
    for f in flaps:
        f.remove()
    flaps.clear()
    left, right = p[0, left_exit], p[0, right_exit]
    flaps.append(ax_side.fill_between([left - 0.05, left + edge_bite], -0.01, -thread_radius, color="0.75", zorder=0))
    flaps.append(ax_side.fill_between([right - edge_bite, right + 0.05], -0.01, -thread_radius, color="0.75", zorder=0))
    k = int(stage_of[i])
    title.set_text(f"stage {k}: {labels[k]}     t = {times[i]:5.2f} s     wound gap {wound_gaps[i]*1000:+.1f} mm")


if args.still is not None:
    draw(min(args.still, len(times) - 1))
    out = args.out or f"knot_frame_{args.still}.png"
    fig.savefig(out, dpi=args.dpi)
    print("saved", out)
    raise SystemExit

out = args.out
have_ffmpeg = animation.writers.is_available("ffmpeg")
if out is None:
    out = "knot_animation.mp4" if have_ffmpeg else "knot_animation.gif"
if out.endswith(".mp4") and not have_ffmpeg:
    out = out[:-4] + ".gif"
    print("ffmpeg not found - writing a GIF instead (conda install -c conda-forge ffmpeg for MP4)")
writer = (animation.FFMpegWriter(fps=args.fps, bitrate=2400) if out.endswith(".mp4")
          else animation.PillowWriter(fps=args.fps))
with writer.saving(fig, out, dpi=args.dpi):
    for n, i in enumerate(shown):
        draw(i)
        writer.grab_frame()
        if n % 100 == 0:
            print(f"  frame {n}/{len(shown)}", flush=True)
print(f"saved {out}  ({len(shown)} frames at {args.fps} fps = {len(shown)/args.fps:.0f} s of video; "
      f"{(times[shown[-1]] - times[shown[0]]):.0f} s simulated)")
