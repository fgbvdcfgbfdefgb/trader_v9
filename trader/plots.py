"""PNG plotting: per-epoch day snapshots (the advisor's response to that day's
market) and cumulative training curves. All saved with the Agg backend - no
display needed."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .data import PAIRS  # noqa: E402
from .features import W  # noqa: E402

COLORS = {"BTCUSDT": "#f7931a", "ETHUSDT": "#627eea", "LTCUSDT": "#41a5ee"}


def plot_day_snapshot(
    path,
    epoch,
    date,
    closes,
    weights,
    equity,
    capital,
    target,
    forecast15,
    realized15,
    mae15,
    vol_classes,
    hit=False,
    margin_call=False,
    extra="",
):
    """One PNG per epoch: prices + trade-maker exposure per coin, equity curve,
    the price-predictor's forecast vs reality, and the market-analyzer's
    regime read of the day (the advisor's response to that day's market)."""
    T = closes.shape[0]
    n = len(equity)
    fig = plt.figure(figsize=(16, 13), dpi=100)
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1.3, 1.1, 1.0], hspace=0.6)
    status = "TARGET HIT" if hit else "target missed"
    if margin_call:
        status += " | MARGIN CALL"
    fig.suptitle(
        f"epoch {epoch} | {date} UTC | final equity ${equity[-1]:.2f} / target ${target:.2f} | {status}"
        + (f" | {extra}" if extra else ""),
        fontsize=13,
    )

    for i, pair in enumerate(PAIRS):
        ax = fig.add_subplot(gs[i])
        y = closes[:, i] / max(closes[0, i], 1e-9)
        ax.plot(np.arange(T), y, color=COLORS[pair], lw=0.9)
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi - lo < 1e-12:
            hi = lo + 1e-3
        pad = (hi - lo) * 0.08
        xw = np.arange(W - 1, W - 1 + n)
        ax.fill_between(xw, lo - pad, hi + pad, where=weights[:, i] > 0.05, color="#2ecc71", alpha=0.25, linewidth=0)
        ax.fill_between(xw, lo - pad, hi + pad, where=weights[:, i] < -0.05, color="#e74c3c", alpha=0.25, linewidth=0)
        ax.set_xlim(0, T)
        ax.set_title(f"{pair} close (normalised) - green=long exposure, red=short exposure", fontsize=9)
        ax.set_ylabel("norm price")
        ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[3])
    ax.plot(np.arange(W - 1, W - 1 + n), equity, color="#8e44ad", lw=1.2)
    ax.axhline(capital, color="gray", ls="--", lw=0.8, label=f"start ${capital:.0f}")
    ax.axhline(target, color="#2ecc71", ls="--", lw=0.9, label=f"target ${target:.0f}")
    ax.set_title("trade-maker equity through the day", fontsize=10)
    ax.set_ylabel("$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[4])
    ax.plot(np.arange(T), realized15, color="#555555", lw=0.8, label="realized next-15m return (BTC, %)")
    ax.plot(np.arange(T), forecast15, color="#e67e22", lw=0.8, alpha=0.9, label="predictor forecast")
    ax.set_title(f"price-predictor agent - 15-minute horizon (BTC) - MAE {mae15:.4f} %", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[5])
    cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e74c3c"])
    ax.imshow(
        np.asarray(vol_classes).T,
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        extent=[0, T, 0, len(PAIRS)],
        interpolation="nearest",
    )
    ax.set_yticks([i + 0.5 for i in range(len(PAIRS))], labels=list(PAIRS))
    ax.set_title("market-analyzer agent - predicted volatility regime (green=calm, amber=normal, red=stormy)", fontsize=9)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _ema(x, a=0.08):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    e = 0.0
    started = False
    for i, v in enumerate(x):
        if not np.isfinite(v):
            out[i] = np.nan
            continue
        e = v if not started else a * v + (1 - a) * e
        started = True
        out[i] = e
    return out


def _panel(ax, hist, keys, title, labels=None):
    plotted = False
    for j, k in enumerate(keys):
        x = np.asarray(hist.get(k, []), dtype=float)
        if x.size == 0:
            continue
        ax.plot(x, lw=0.5, alpha=0.30, color=f"C{j}")
        ax.plot(_ema(x), lw=1.5, color=f"C{j}", label=(labels or keys)[j])
        plotted = True
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend(fontsize=7)


def plot_curves(path, hist, capital, target):
    """Cumulative training curves PNG (rewritten every epoch)."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), dpi=100)
    fig.suptitle("trader_v9 training curves", fontsize=13)
    _panel(axes[0, 0], hist, ["pred_loss"], "price-predictor loss (MSE)")
    _panel(axes[0, 1], hist, ["pred_mae15", "pred_mae60"], "predictor MAE (% , EMA)", ["15m", "60m"])
    _panel(axes[0, 2], hist, ["ana_vol_acc", "ana_trend_acc"], "market-analyzer accuracy", ["vol regime", "trend"])
    axes[0, 2].axhline(1 / 3, color="gray", ls="--", lw=0.7)
    _panel(axes[1, 0], hist, ["pi_loss", "v_loss"], "trade-maker PPO losses", ["policy", "value"])
    _panel(axes[1, 1], hist, ["entropy"], "policy entropy")
    _panel(axes[1, 2], hist, ["ep_ret"], "epoch return (sum of rewards)")
    _panel(axes[2, 0], hist, ["final_eq"], f"final equity per epoch (target ${target:.0f})")
    n = len(hist.get("final_eq", []))
    if n:
        axes[2, 0].axhline(capital, color="gray", ls="--", lw=0.8)
        axes[2, 0].axhline(target, color="#2ecc71", ls="--", lw=0.9)
    _panel(axes[2, 1], hist, ["eval_mean_eq"], "held-out eval mean equity")
    _panel(axes[2, 2], hist, ["eval_hit"], "held-out eval target hit-rate")
    if hist.get("eval_hit"):
        axes[2, 2].set_ylim(-0.05, 1.05)
    for ax in axes[2]:
        ax.set_xlabel("epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path)
    plt.close(fig)
