"""Knot type of the thread from its knot determinant (use --strands for surgical knots)."""
import argparse
import numpy as np

KNOT_NAMES = {1: "unknot (no knot)", 3: "trefoil (single throw)", 5: "figure-eight",
              7: "5_2 (double throw)", 9: "square or granny (two single throws)",
              21: "surgeon's knot (double + single throw)"}


def outward(end, centre):
    d = (end - centre) * np.array([1.0, 0.0, 1.0])
    return d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-12 else np.array([1.0, 0.0, 0.0])


def close_over(p):
    """Join the two free ends far outside the knot: out, up and across the top."""
    centre = p.mean(axis=1)
    size = float(np.ptp(p, axis=1).max()) + 1e-9
    top = p[1].max() + 5.0 * size
    ends = []
    for end in (p[:, -1], p[:, 0]):
        far = end + 5.0 * size * outward(end, centre)
        ends.append((far, far + np.array([0.0, top - far[1], 0.0])))
    (last_far, last_top), (first_far, first_top) = ends
    return np.column_stack([p, last_far, last_top, first_top, first_far])


def close_under(strand):
    """Close one strand (exit first, free end last) under the tissue, like the stitch does."""
    centre = strand.mean(axis=1)
    size = float(np.ptp(strand, axis=1).max()) + 1e-9
    far = strand[:, -1] + 5.0 * size * outward(strand[:, -1], centre)
    low = strand[1].min() - 5.0 * size
    return np.column_stack([strand, far, [far[0], low, far[2]], [strand[0, 0], low, strand[2, 0]]])


def crossings(curve, view):
    """Crossings of a closed polygon seen along `view`."""
    n = curve.shape[1]
    e1 = np.cross(view, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(view, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(view, e1)
    flat = np.stack([e1 @ curve, e2 @ curve])
    height = view @ curve
    seg = np.roll(flat, -1, axis=1) - flat
    found = []
    for i in range(n):
        others = np.arange(i + 2, n)
        if i == 0:
            others = others[others != n - 1]
        if len(others) == 0:
            continue
        r = flat[:, others] - flat[:, [i]]
        den = seg[0, i] * seg[1, others] - seg[1, i] * seg[0, others]
        ok = np.abs(den) > 1e-14
        s = np.where(ok, (r[0] * seg[1, others] - r[1] * seg[0, others]) / np.where(ok, den, 1), -1)
        u = np.where(ok, (r[0] * seg[1, i] - r[1] * seg[0, i]) / np.where(ok, den, 1), -1)
        hit = ok & (s > 0) & (s < 1) & (u > 0) & (u < 1)
        for j, sj, uj in zip(others[hit], s[hit], u[hit]):
            h_i = height[i] + sj * (height[(i + 1) % n] - height[i])
            h_j = height[j] + uj * (height[(j + 1) % n] - height[j])
            sign = np.sign(seg[0, i] * seg[1, j] - seg[1, i] * seg[0, j])
            found.append((i + sj, j + uj, sign) if h_i > h_j else (j + uj, i + sj, -sign))
    return found


def alexander(cross, t):
    """Reduced Alexander matrix determinant at t; arcs run between under-passes."""
    n = len(cross)
    if n == 0:
        return 1.0
    unders = sorted((c[1], k) for k, c in enumerate(cross))
    under_at = np.array([u for u, _ in unders])
    M = np.zeros((n, n), dtype=complex)
    for row, (_, k) in enumerate(unders):
        over, _, sign = cross[k]
        arc_in, arc_out = row, (row + 1) % n
        over_arc = int(np.searchsorted(under_at, over)) % n
        if sign > 0:
            M[row, over_arc] += 1 - t; M[row, arc_in] += t; M[row, arc_out] += -1
        else:
            M[row, over_arc] += 1 - t; M[row, arc_in] += -1; M[row, arc_out] += t
    return np.linalg.det(M[1:, 1:]) if n > 1 else 1.0


def knot_determinant(p, views=7, seed=0, closed=False):
    """Most common determinant over several random viewing directions."""
    curve = np.asarray(p, dtype=float) if closed else close_over(np.asarray(p, dtype=float))
    rng = np.random.default_rng(seed)
    dets, counts = [], []
    for _ in range(views):
        view = rng.normal(size=3)
        view /= np.linalg.norm(view)
        c = crossings(curve, view)
        counts.append(len(c))
        dets.append(int(round(abs(alexander(c, -1.0).real))))
    values, freq = np.unique(dets, return_counts=True)
    return int(values[np.argmax(freq)]), dets, counts


def load(path):
    if path.endswith(".npz"):
        return [np.load(path)["position"]]
    shapes = np.load(path)
    return [shapes] if shapes.ndim == 2 else list(shapes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--first-stage", type=int, default=None,
                        help="stage number of the first snapshot in a *_stages.npy file")
    parser.add_argument("--strands", action="store_true",
                        help="check the tail and the long strand on their own (for surgical knots)")
    parser.add_argument("--tail-exit", type=int, default=30)
    parser.add_argument("--long-exit", type=int, default=37)
    args = parser.parse_args()
    for path in args.files:
        shapes = load(path)
        for i, p in enumerate(shapes):
            label = (f"stage {args.first_stage + i:2d}" if args.first_stage is not None and len(shapes) > 1
                     else (f"snapshot {i}" if len(shapes) > 1 else ""))
            if args.strands:
                parts = []
                for name, strand in (("tail", p[:, args.tail_exit::-1]), ("long strand", p[:, args.long_exit:])):
                    det, dets, _ = knot_determinant(close_under(strand), closed=True)
                    unsure = "" if len(set(dets)) == 1 else "?"
                    parts.append(f"{name} {det}{unsure} ({'clean' if det == 1 else KNOT_NAMES.get(det, 'knotted')})")
                print(f"{path} {label}: " + ";  ".join(parts))
            else:
                det, dets, counts = knot_determinant(p)
                unsure = "" if len(set(dets)) == 1 else f"   (directions disagree: {dets})"
                print(f"{path} {label}: determinant {det:3d} -> {KNOT_NAMES.get(det, 'other / composite')}"
                      f"   [{min(counts)}-{max(counts)} crossings]{unsure}")
