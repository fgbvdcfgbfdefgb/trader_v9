#!/usr/bin/env python3
"""Evaluate saved checkpoints on the most recent (held-out) days.

Usage:  python eval.py --ckpt runs/ckpt --days 40
Produces runs/eval/eval_summary.png + eval_results.csv + one PNG per day.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trader import data as D  # noqa: E402
from trader import plots  # noqa: E402
from trader.agents import Analyzer, Predictor, Trader  # noqa: E402
from trader.data import PAIRS  # noqa: E402
from trader.env import TradingEnv  # noqa: E402
from trader.features import HORIZONS, N_FEATURES  # noqa: E402
from train import EMB_DIM, DayProcessed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ckpt")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--days", type=int, default=40, help="most recent days to evaluate")
    ap.add_argument("--capital", type=float, default=20.0)
    ap.add_argument("--target", type=float, default=30.0)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--max-lev", type=float, default=5.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    days = D.load_days(args.data)
    dates = sorted(days)[-args.days:]
    print(f"[eval] {len(dates)} days on {device}")

    n_out = len(PAIRS) * len(HORIZONS)
    predictor = Predictor(N_FEATURES, n_out, hidden=64).to(device)
    analyzer = Analyzer(N_FEATURES, len(PAIRS), hidden=64, emb_dim=EMB_DIM).to(device)
    for name, model in (("predictor", predictor), ("analyzer", analyzer)):
        ck = torch.load(os.path.join(args.ckpt, f"{name}.pt"),
                        map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()

    ck_t = torch.load(os.path.join(args.ckpt, "trader.pt"),
                      map_location=device, weights_only=False)
    obs_dim = ck_t["model"]["trunk.0.weight"].shape[1]
    hidden = ck_t["model"]["trunk.0.weight"].shape[0]
    trader = Trader(obs_dim, act_dim=3, hidden=hidden).to(device)
    trader.load_state_dict(ck_t["model"])
    trader.eval()

    env_kwargs = dict(capital=args.capital, target=args.target, fee=args.fee,
                      slip=args.slip, max_lev=args.max_lev)

    rows = []
    for date in dates:
        proc = DayProcessed(date, days[date])
        with torch.no_grad():
            x = torch.as_tensor(proc.feats[None], device=device)
            forecast = predictor(x)[0].cpu().numpy().astype(np.float32)
            emb, vl, _ = analyzer(x)
            emb = emb[0].cpu().numpy().astype(np.float32)
            vol_cls = vl[0].argmax(-1).cpu().numpy()
        env = TradingEnv(proc.day, proc.feats, forecast, emb, **env_kwargs)
        obs = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                o_t = torch.as_tensor(obs, device=device)
                mu, _, _ = trader(o_t)
                a = mu.clamp(-1, 1).cpu().numpy()
            obs, r, done, info = env.step(a)
        rows.append({"date": date, "final_eq": info["equity"],
                     "hit": float(info["equity"] >= args.target),
                     "margin_call": float(env.margin_call)})
        closes = np.stack([proc.day[p]["close"] for p in PAIRS], axis=1)
        plots.plot_day_snapshot(
            os.path.join(args.out, f"eval_{date}.png"), 0, date, closes,
            env.weights_hist, env.equity_hist, args.capital, args.target,
            forecast[:, 0], proc.Y[:, 0], float(np.mean(np.abs(forecast[:, 0] - proc.Y[:, 0]))),
            vol_cls, hit=info["equity"] >= args.target, margin_call=env.margin_call,
            extra="held-out evaluation")
        print(f"  {date}: ${info['equity']:7.2f} {'HIT' if info['equity'] >= args.target else '   '}")

    finals = np.array([r["final_eq"] for r in rows])
    hits = np.array([r["hit"] for r in rows])
    with open(os.path.join(args.out, "eval_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "final_eq", "hit", "margin_call"])
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=110)
    ax.bar(range(len(rows)), finals, color=["#2ecc71" if h else "#95a5a6" for h in hits])
    ax.axhline(args.capital, color="gray", ls="--", lw=1, label=f"start ${args.capital:.0f}")
    ax.axhline(args.target, color="#2ecc71", ls=":", lw=1.4, label=f"target ${args.target:.0f}")
    ax.axhline(finals.mean(), color="#8e44ad", lw=1.4, label=f"mean ${finals.mean():.2f}")
    ax.set_xticks(range(len(rows)), [r["date"][5:] for r in rows], rotation=90, fontsize=7)
    ax.set_ylabel("final equity ($)")
    ax.set_title(f"Held-out evaluation: {len(rows)} days | mean ${finals.mean():.2f} | "
                 f"median ${np.median(finals):.2f} | hit-rate {hits.mean()*100:.1f}% | "
                 f"max ${finals.max():.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "eval_summary.png"))
    plt.close(fig)
    print(f"[eval] mean=${finals.mean():.2f} median=${np.median(finals):.2f} "
          f"hit={hits.mean()*100:.1f}% max=${finals.max():.2f} -> {args.out}")


if __name__ == "__main__":
    main()
