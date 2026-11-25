"""E3a: dead-band, P and PI tension control on a spring model of one strand."""
import csv

import numpy as np

import common
import tension_knot_exp as tk

EA = tk.YOUNGS_MODULUS * np.pi * tk.THREAD_RADIUS ** 2     # 78.5 N
STRAND = 0.30
K_S = EA / STRAND
LOSS = 1.0
T0, DT, T_END = 1.5, 0.001, 7.0
CASES = [("deadband", 15.0, 0, 0), ("deadband", 12.0, 0, 0),
         ("p", 12.0, 0.005, 0), ("p", 12.0, 0.01, 0), ("p", 12.0, 0.02, 0),
         ("pi", 12.0, 0.01, 0.005), ("pi", 12.0, 0.01, 0.01), ("pi", 12.0, 0.02, 0.02)]


def label(kind, target, kp, ki):
    if kind == "deadband":
        return f"deadband T*={target:g} N" + (" (ours)" if target == 15 else "")
    return f"{kind.upper()} kp={kp:g}" + (f" ki={ki:g}" if kind == "pi" else "") + f" T*={target:g} N"


def simulate(kind, target, kp, ki, speed_out=tk.SPEED_OUT_A_CINCH):
    t = np.arange(0.0, T_END, DT)
    T, travel, integral = np.empty_like(t), 0.0, 0.0
    tension = T0
    for i in range(len(t)):
        T[i] = tension
        if kind == "deadband":
            new_travel = tk.servo_step(travel, tension, target, speed_out, DT)
            speed = (new_travel - travel) / DT
        else:
            speed, integral = tk.feedback_speed(kind, kp, ki, integral, target - tension, DT, speed_out)
        travel += speed * DT
        tension = max(0.0, tension + (K_S * speed - LOSS) * DT)
    return t, T


def metrics(t, T, target):
    hold = t >= T_END - 2.0
    band = np.abs(T - target) > 0.05 * target
    settle = t[np.where(band)[0][-1] + 1] if band.any() and np.where(band)[0][-1] + 1 < len(t) else (0.0 if not band.any() else np.nan)
    return dict(final_mean_N=T[hold].mean(), steady_error_pct=100 * (T[hold].mean() - target) / target,
                overshoot_pct=max(0.0, 100 * (T.max() - target) / target), settling_s=settle)


def main():
    out = common.out_dir("controller_model")
    rows, series = [], {}
    for kind, target, kp, ki in CASES:
        t, T = simulate(kind, target, kp, ki)
        name = label(kind, target, kp, ki)
        row = dict(controller=name, target_N=target, kp=kp, ki=ki, **metrics(t, T, target))
        if kind != "deadband":
            row["tau_s"] = 1.0 / (K_S * kp)
            if kind == "pi":
                wn = np.sqrt(K_S * ki)
                row["wn_rad_s"], row["zeta"] = wn, K_S * kp / (2 * wn)
        rows.append(row)
        series[name] = T.astype(np.float32)
        print(f"{name:32s} final {row['final_mean_N']:5.2f} N, error {row['steady_error_pct']:+6.1f}%, "
              f"overshoot {row['overshoot_pct']:4.1f}%, settles {row['settling_s']:.2f} s")
    keys = ["controller", "target_N", "kp", "ki", "final_mean_N", "steady_error_pct", "overshoot_pct",
            "settling_s", "tau_s", "wn_rad_s", "zeta"]
    with open(f"{out}/controller_model.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys} for r in rows)
    np.savez_compressed(f"{out}/controller_series.npz", t=t, K_s=K_S, loss=LOSS, **series)
    print(f"K_s = {K_S:.0f} N/m; wrote {out}/controller_model.csv")


if __name__ == "__main__":
    main()
