#!/usr/bin/env python3
"""
trader_v9 - offline multi-agent RL trading training.

Every epoch:
  1. A random trading day (UTC; 1-minute bars of BTC, ETH and LTC) is drawn.
  2. The price-predictor and market-analyzer agents train on that day (plus a
     small replay of recent days) - in parallel threads, each on its own GPU
     when the hardware allows it.
  3. The trade-maker agent (PPO) trades the entire day starting with $20,
     consuming the predictor's forecasts and the analyzer's market embedding,
     aiming to finish at or above the $30 target.
  4. Everything is saved: per-epoch PNG snapshots of the advisor's response to
     that day's market, cumulative training-curve PNGs, metrics.csv and model
     checkpoints in runs/.

Fully offline: the dataset ships inside the repo, no network is used at
training time. Hardware is auto-detected (CPU / single GPU / multi-GPU) and
model placement + size adapt accordingly.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trader import data as D  # noqa: E402
from trader import plots  # noqa: E402
from trader.agents import Analyzer, Predictor, Trader  # noqa: E402
from trader.data import PAIRS  # noqa: E402
from trader.env import TradingEnv  # noqa: E402
from trader.features import (  # noqa: E402
    HORIZONS,
    N_FEATURES,
    W,
    build_features,
    build_labels,
    build_targets,
)
from trader.ppo import RolloutBuffer, ppo_update  # noqa: E402
from trader.utils import choose_plan, detect_resources, format_resources, set_seed  # noqa: E402

EMB_DIM = 16
CSV_FIELDS = [
    "epoch", "date", "pred_loss", "pred_mae15", "pred_mae60",
    "ana_loss", "ana_vol_acc", "ana_trend_acc",
    "pi_loss", "v_loss", "entropy", "kl", "clip_frac",
    "ep_ret", "final_eq", "hit", "margin_call",
    "eval_mean_eq", "eval_median_eq", "eval_hit", "elapsed",
]


class DayProcessed:
    __slots__ = ("date", "day", "feats", "Y", "M", "vol_lab", "trend_lab", "lab_mask")

    def __init__(self, date, day):
        self.date = date
        self.day = day
        self.feats = build_features(day)
        self.Y, self.M = build_targets(day)
        self.vol_lab, self.trend_lab, self.lab_mask = build_labels(day)


def parse_args():
    p = argparse.ArgumentParser(
        description="trader_v9 multi-agent RL training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="./data", help="dataset directory (shipped in repo)")
    p.add_argument("--out", default="./runs", help="output directory")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    # hardware
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--min-free-gb", type=float, default=3.0,
                   help="a GPU is only used if it has at least this much free VRAM "
                        "(keeps a running miner untouched; force GPUs with --device cuda)")
    p.add_argument("--hidden", default="auto", help="trader MLP hidden width: 'auto' or int")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = let torch decide)")
    p.add_argument("--max-days", type=int, default=0, help="limit days loaded (0 = all)")
    # data split
    p.add_argument("--val-days", type=int, default=40, help="most recent days held out for eval")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-limit", type=int, default=16, help="val days per eval round")
    # market / task
    p.add_argument("--capital", type=float, default=20.0)
    p.add_argument("--target", type=float, default=30.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--max-lev", type=float, default=5.0)
    p.add_argument("--terminal-bonus", type=float, default=0.5)
    # predictor / analyzer
    p.add_argument("--pred-steps", type=int, default=8)
    p.add_argument("--ana-steps", type=int, default=8)
    p.add_argument("--batch-days", type=int, default=4)
    p.add_argument("--replay", type=int, default=30, help="recent days kept for replay")
    p.add_argument("--bg-steps", type=int, default=4,
                   help="extra predictor/analyzer steps run in background during the rollout")
    p.add_argument("--no-background-a", action="store_true",
                   help="disable background predictor/analyzer training during rollouts")
    # ppo
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr-trader", type=float, default=3e-4)
    p.add_argument("--lr-agents", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.003)
    # logging
    p.add_argument("--plot-every", type=int, default=1)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--resume", default="", help="ckpt dir to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "epochs"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # ---------------- resources & plan ----------------
    res = detect_resources()
    plan = choose_plan(args, res)
    print(format_resources(res))
    print(plan["layout"] + f" | trader hidden={plan['hidden']}")
    with open(os.path.join(args.out, "resources.txt"), "w") as f:
        f.write(format_resources(res) + "\n\n" + plan["layout"]
                + f"\ntrader hidden width: {plan['hidden']}\n")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "plan": {k: str(v) for k, v in plan.items()}}, f, indent=2)

    set_seed(args.seed)

    # ---------------- data ----------------
    days = D.load_days(args.data, max_days=args.max_days)
    dates = sorted(days)
    if len(dates) < 20:
        raise RuntimeError("need at least 20 valid days of data")
    n_val = args.val_days
    if n_val >= len(dates) - 10:
        n_val = max(0, (len(dates) - 10) // 2)
    val_dates = dates[len(dates) - n_val:] if n_val > 0 else []
    train_dates = dates[: len(dates) - n_val]
    print(f"[data] train days: {len(train_dates)} ({train_dates[0]}..{train_dates[-1]})"
          f" | held-out val days: {len(val_dates)}")

    # ---------------- agents ----------------
    dev_p, dev_a, dev_t = plan["pred"], plan["ana"], plan["trade"]
    n_out = len(PAIRS) * len(HORIZONS)
    obs_dim = W * N_FEATURES + n_out + EMB_DIM + 3 + 1

    predictor = Predictor(N_FEATURES, n_out, hidden=64).to(dev_p)
    analyzer = Analyzer(N_FEATURES, len(PAIRS), hidden=64, emb_dim=EMB_DIM).to(dev_a)
    trader = Trader(obs_dim, act_dim=3, hidden=plan["hidden"]).to(dev_t)
    opt_p = torch.optim.Adam(predictor.parameters(), lr=args.lr_agents)
    opt_a = torch.optim.Adam(analyzer.parameters(), lr=args.lr_agents)
    opt_t = torch.optim.Adam(trader.parameters(), lr=args.lr_trader)
    n_params = sum(p.numel() for p in trader.parameters())
    print(f"[agents] trader params: {n_params/1e6:.2f}M | obs_dim={obs_dim} | actions=3 weights in [-1,1]")

    # ---------------- resume ----------------
    start_epoch = 1
    if args.resume:
        for name, model, opt in (
            ("predictor", predictor, opt_p),
            ("analyzer", analyzer, opt_a),
            ("trader", trader, opt_t),
        ):
            p = os.path.join(args.resume, f"{name}.pt")
            if os.path.exists(p):
                ck = torch.load(p, map_location="cpu", weights_only=False)
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                start_epoch = max(start_epoch, int(ck.get("epoch", 0)) + 1)
        print(f"[resume] continuing from epoch {start_epoch}")

    # ---------------- day cache / replay ----------------
    proc_cache = OrderedDict()
    cache_lock = threading.Lock()
    cache_cap = max(args.replay, 8)

    def get_processed(date):
        with cache_lock:
            if date in proc_cache:
                proc_cache.move_to_end(date)
                return proc_cache[date]
        proc = DayProcessed(date, days[date])
        with cache_lock:
            proc_cache[date] = proc
            while len(proc_cache) > cache_cap:
                proc_cache.popitem(last=False)
        return proc

    def sample_batch(today=None, weight_today=4):
        with cache_lock:
            pool = list(proc_cache.values())
        if today is not None:
            pool = [today] * weight_today + pool
        if not pool:
            return None
        k = min(args.batch_days, len(pool))
        return random.sample(pool, k)

    # ---------------- agent training steps ----------------
    def predictor_train_batch(batch):
        x = torch.as_tensor(np.stack([d.feats for d in batch]), device=dev_p)
        Y = torch.as_tensor(np.stack([d.Y for d in batch]), device=dev_p)
        M = torch.as_tensor(np.stack([d.M for d in batch]), device=dev_p)
        pred = predictor(x)
        loss = ((pred - Y).pow(2) * M).sum() / M.sum().clamp(min=1.0)
        opt_p.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        opt_p.step()
        return float(loss.item())

    def analyzer_train_batch(batch):
        x = torch.as_tensor(np.stack([d.feats for d in batch]), device=dev_a)
        vol = torch.as_tensor(np.stack([d.vol_lab for d in batch]), device=dev_a)
        tr = torch.as_tensor(np.stack([d.trend_lab for d in batch]), device=dev_a)
        mask = torch.as_tensor(np.stack([d.lab_mask for d in batch]), device=dev_a)
        emb, vl, tl = analyzer(x)
        B, T = mask.shape
        v = vl.reshape(B * T, len(PAIRS), 3)
        t_ = tl.reshape(B * T, len(PAIRS), 3)
        yv = vol.reshape(B * T, len(PAIRS))
        yt = tr.reshape(B * T, len(PAIRS))
        m = mask.reshape(B * T)
        loss = F.cross_entropy(v[m], yv[m]) + F.cross_entropy(t_[m], yt[m])
        opt_a.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(analyzer.parameters(), 1.0)
        opt_a.step()
        return float(loss.item())

    @torch.no_grad()
    def predictor_eval(day):
        x = torch.as_tensor(day.feats[None], device=dev_p)
        Y = torch.as_tensor(day.Y[None], device=dev_p)
        M = torch.as_tensor(day.M[None], device=dev_p)
        pred = predictor(x)
        out = {"loss": float(((pred - Y).pow(2) * M).sum().item() / max(M.sum().item(), 1.0))}
        for k, h in enumerate(HORIZONS):
            cols = [i * len(HORIZONS) + k for i in range(len(PAIRS))]
            m, y, p = M[:, cols], Y[:, cols], pred[:, cols]
            out[f"mae{h}"] = float(((p - y).abs() * m).sum().item() / max(m.sum().item(), 1.0))
        return out

    @torch.no_grad()
    def analyzer_eval(day):
        x = torch.as_tensor(day.feats[None], device=dev_a)
        vol = torch.as_tensor(day.vol_lab[None], device=dev_a)
        tr = torch.as_tensor(day.trend_lab[None], device=dev_a)
        mask = torch.as_tensor(day.lab_mask[None], device=dev_a)
        emb, vl, tl = analyzer(x)
        m = mask[0].cpu().numpy()
        v = vl[0].argmax(-1).cpu().numpy()
        t_ = tl[0].argmax(-1).cpu().numpy()
        yv = tr[0].cpu().numpy()
        yt = vol[0].cpu().numpy()
        va = float((v[m] == yv[m]).mean()) if m.any() else float("nan")
        ta = float((t_[m] == yt[m]).mean()) if m.any() else float("nan")
        B, T = mask.shape
        vv = vl.reshape(B * T, len(PAIRS), 3)
        tt = tl.reshape(B * T, len(PAIRS), 3)
        yv2 = vol.reshape(B * T, len(PAIRS))
        yt2 = tr.reshape(B * T, len(PAIRS))
        mm = mask.reshape(B * T)
        loss = float(F.cross_entropy(vv[mm], yv2[mm]).item()
                     + F.cross_entropy(tt[mm], yt2[mm]).item())
        return {"loss": loss, "vol_acc": va, "trend_acc": ta}

    @torch.no_grad()
    def advisor_outputs(day):
        x = torch.as_tensor(day.feats[None], device=dev_p)
        forecast = predictor(x)[0].cpu().numpy().astype(np.float32)
        xa = torch.as_tensor(day.feats[None], device=dev_a)
        emb, vl, tl = analyzer(xa)
        emb = emb[0].cpu().numpy().astype(np.float32)
        vol_cls = vl[0].argmax(-1).cpu().numpy().astype(np.int64)  # (T,3)
        return forecast, emb, vol_cls

    # ---------------- trading ----------------
    env_kwargs = dict(capital=args.capital, target=args.target, fee=args.fee,
                      slip=args.slip, max_lev=args.max_lev,
                      terminal_bonus=args.terminal_bonus)
    buf = RolloutBuffer()

    def rollout(proc, forecast, emb, deterministic=False):
        env = TradingEnv(proc.day, proc.feats, forecast, emb, **env_kwargs)
        obs = env.reset()
        ep_ret = 0.0
        if not deterministic:
            buf.clear()
        done = False
        while not done:
            o_t = torch.as_tensor(obs, device=dev_t)
            if deterministic:
                with torch.no_grad():
                    a, _, _ = trader.act(o_t, deterministic=True)
                a_np = a.cpu().numpy()
                obs, r, done, info = env.step(a_np)
            else:
                with torch.no_grad():
                    a, logp, v = trader.act(o_t)
                a_np = a.cpu().numpy()
                prev_obs = obs
                obs, r, done, info = env.step(a_np)
                buf.add(prev_obs, a_np, float(logp.item()), float(v.item()), r, done)
                ep_ret += r
        return env, info, ep_ret

    def evaluate(limit):
        step = max(1, len(val_dates) // max(limit, 1))
        sel = val_dates[::step][:limit]
        finals = []
        for d in sel:
            proc = DayProcessed(d, days[d])
            forecast, emb, _ = advisor_outputs(proc)
            _, info, _ = rollout(proc, forecast, emb, deterministic=True)
            finals.append(info["equity"])
        finals = np.asarray(finals, dtype=np.float64)
        return {
            "mean": float(finals.mean()),
            "median": float(np.median(finals)),
            "hit": float((finals >= args.target).mean()),
            "finals": finals.tolist(),
        }

    def save_ckpt(directory, epoch):
        os.makedirs(directory, exist_ok=True)
        for name, model, opt in (
            ("predictor", predictor, opt_p),
            ("analyzer", analyzer, opt_a),
            ("trader", trader, opt_t),
        ):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch},
                       os.path.join(directory, f"{name}.pt"))

    # ---------------- metrics history ----------------
    hist = {k: [] for k in CSV_FIELDS}
    csv_path = os.path.join(args.out, "metrics.csv")
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                for k in CSV_FIELDS:
                    v = row.get(k, "")
                    if k == "date":
                        hist[k].append(v)
                    else:
                        try:
                            hist[k].append(float(v))
                        except (TypeError, ValueError):
                            hist[k].append(float("nan"))

    csv_new = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if csv_new:
        writer.writeheader()

    # ---------------- main loop ----------------
    best_eval = -1e9
    t_start = time.time()
    print(f"[train] start: {args.epochs} epochs | ${args.capital} -> ${args.target} target | "
          f"fee={args.fee} slip={args.slip} max_lev={args.max_lev}")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            date = random.choice(train_dates)
            proc = get_processed(date)

            # ---- Phase A: predictor + analyzer train on today (+replay), in parallel
            pred_losses, ana_losses = [], []

            def _pa():
                for _ in range(args.pred_steps):
                    b = sample_batch(today=proc)
                    if b:
                        pred_losses.append(predictor_train_batch(b))

            def _aa():
                for _ in range(args.ana_steps):
                    b = sample_batch(today=proc)
                    if b:
                        ana_losses.append(analyzer_train_batch(b))

            ths = [threading.Thread(target=_pa), threading.Thread(target=_aa)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()

            pm = predictor_eval(proc)
            am = analyzer_eval(proc)

            # advisor outputs for today (frozen snapshot for the rollout)
            forecast, emb, vol_cls = advisor_outputs(proc)

            # ---- Phase B: trader rollout; the other two agents keep training in
            #      background threads on their own GPUs while the trader trades.
            bg = []
            if plan["parallel"] and not args.no_background_a:
                def _bgp():
                    for _ in range(args.bg_steps):
                        b = sample_batch()
                        if b:
                            predictor_train_batch(b)

                def _bga():
                    for _ in range(args.bg_steps):
                        b = sample_batch()
                        if b:
                            analyzer_train_batch(b)

                bg = [threading.Thread(target=_bgp), threading.Thread(target=_bga)]
                for t in bg:
                    t.start()

            env, info, ep_ret = rollout(proc, forecast, emb)
            if len(buf) >= 16:
                ppo_stats = ppo_update(trader, opt_t, buf, dev_t, gamma=args.gamma,
                                       lam=args.lam, ppo_epochs=args.ppo_epochs,
                                       minibatch=args.minibatch, ent_coef=args.ent_coef)
            else:  # episode ended almost immediately; nothing useful to learn
                ppo_stats = {k: float("nan") for k in
                             ("pi_loss", "v_loss", "entropy", "kl", "clip_frac")}
            for t in bg:
                t.join()

            hit = info["equity"] >= args.target

            # ---- held-out evaluation
            eval_row = {"eval_mean_eq": float("nan"), "eval_median_eq": float("nan"),
                        "eval_hit": float("nan")}
            if val_dates and (epoch % args.eval_every == 0 or epoch == args.epochs):
                ev = evaluate(args.eval_limit)
                eval_row = {"eval_mean_eq": ev["mean"], "eval_median_eq": ev["median"],
                            "eval_hit": ev["hit"]}
                if ev["mean"] > best_eval:
                    best_eval = ev["mean"]
                    save_ckpt(os.path.join(args.out, "ckpt", "best"), epoch)

            # ---- logs / plots / checkpoints
            row = {
                "epoch": epoch, "date": date,
                "pred_loss": pm["loss"], "pred_mae15": pm["mae15"], "pred_mae60": pm["mae60"],
                "ana_loss": am["loss"], "ana_vol_acc": am["vol_acc"], "ana_trend_acc": am["trend_acc"],
                "pi_loss": ppo_stats["pi_loss"], "v_loss": ppo_stats["v_loss"],
                "entropy": ppo_stats["entropy"], "kl": ppo_stats["kl"],
                "clip_frac": ppo_stats["clip_frac"],
                "ep_ret": ep_ret, "final_eq": info["equity"], "hit": float(hit),
                "margin_call": float(env.margin_call),
                **eval_row, "elapsed": round(time.time() - t0, 2),
            }
            writer.writerow(row)
            csv_f.flush()
            for k in CSV_FIELDS:
                hist[k].append(row.get(k, float("nan")))

            if args.plot_every and (epoch % args.plot_every == 0 or epoch == args.epochs):
                closes = np.stack([proc.day[p]["close"] for p in PAIRS], axis=1)
                plots.plot_day_snapshot(
                    os.path.join(args.out, "epochs", f"epoch_{epoch:05d}_{date}.png"),
                    epoch, date, closes, env.weights_hist, env.equity_hist,
                    args.capital, args.target, forecast[:, 0], proc.Y[:, 0],
                    pm["mae15"], vol_cls, hit=hit, margin_call=env.margin_call,
                    extra=f"day PnL {info['equity'] - args.capital:+.2f}$")
                plots.plot_curves(os.path.join(args.out, "training_curves.png"),
                                  hist, args.capital, args.target)

            if epoch % args.ckpt_every == 0 or epoch == args.epochs:
                save_ckpt(os.path.join(args.out, "ckpt"), epoch)

            if args.log_every and (epoch % args.log_every == 0 or epoch == args.epochs):
                ev_txt = ""
                if not math.isnan(eval_row["eval_mean_eq"]):
                    ev_txt = (f" | EVAL mean=${eval_row['eval_mean_eq']:.2f} "
                              f"med=${eval_row['eval_median_eq']:.2f} "
                              f"hit={eval_row['eval_hit']*100:.0f}%")
                print(f"[{epoch:5d}] {date} eq=${info['equity']:7.2f} ret={ep_ret:8.1f} "
                      f"{'HIT ' if hit else 'miss'}{' MC' if env.margin_call else '    '} "
                      f"pMSE={pm['loss']:.4f} pMAE15={pm['mae15']:.4f} "
                      f"aVol={am['vol_acc']:.2f} aTrd={am['trend_acc']:.2f} "
                      f"ent={ppo_stats['entropy']:.2f}{ev_txt} "
                      f"[{time.time() - t0:.1f}s]", flush=True)
    finally:
        csv_f.close()
        save_ckpt(os.path.join(args.out, "ckpt"), epoch)
        print(f"[train] finished {epoch} epochs in {(time.time() - t_start)/60:.1f} min; "
              f"artifacts in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
