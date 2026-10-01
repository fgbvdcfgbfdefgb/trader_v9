# trader_v9 — multi-agent RL trader for BTC / ETH / LTC (1-minute data)

Three cooperating agents trade one full day of 1-minute BTC/ETH/LTC bars per
epoch, each episode starting with **$20** and aiming to finish at **$30+**:

| Agent | Model | Trained by | Role |
|---|---|---|---|
| **Price predictor** | GRU (23 features → per-coin 15m & 60m return forecasts) | supervised MSE, every epoch | tells the trader where prices are heading |
| **Market analyzer** | GRU (→ market-state embedding + vol-regime / trend heads) | supervised CE on next-hour regimes, every epoch | tells the trader what kind of market it is |
| **Trade maker** | MLP actor-critic (Gaussian policy over 3 target weights) | **PPO (RL)** on the trading env | decides long/short/flat + size per coin |

The trade maker consumes the other two agents' outputs each minute
(observation = 64-min feature window ⊕ 6 forecasts ⊕ 16-dim market embedding ⊕
current positions). All three train **concurrently every epoch**; with 3+ GPUs
each agent gets a dedicated GPU and the predictor/analyzer keep training in
background threads while the trader rolls out.

**Fully offline**: the dataset ships inside this repo (`data/`, ~86 MB) and no
network access is needed at training time — only the pip install.

---

## Dataset

Binance spot 1-minute klines (UTC), per coin per year, gzipped CSV:

```
data/BTCUSDT_2023_1m.csv.gz   data/ETHUSDT_2023_1m.csv.gz   data/LTCUSDT_2023_1m.csv.gz
data/BTCUSDT_2024_1m.csv.gz   ...                           ...
```

* Pairs: BTCUSDT, ETHUSDT, LTCUSDT — 2023-01 → 2026-08 (~1,330 valid days)
* Columns: `ts` (unix ms), `open`, `high`, `low`, `close`, `volume`
* `scripts/build_dataset.py` regenerates it from data.binance.vision (one-time,
  network needed only for that).

## Quick start (Snowflake / any machine)

```bash
pip install -r requirements.txt     # CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
python train.py --epochs 2000       # hardware auto-detected
```

or `bash scripts/snowflake_run.sh`.

## Hardware auto-setup

`train.py` inventories the machine first (see `runs/resources.txt`):

* **4 GPUs free** → predictor / analyzer / trader each pinned to its own GPU
  (`--min-free-gb` guard, default 3 GB) + background parallel training; trader MLP width 768.
* **2 GPUs** → trader alone on one, predictor+analyzer on the other; width 768.
* **1 GPU** → everything on it; width 512.
* **CPU only** (or GPUs busy, e.g. shared with a miner) → CPU, width 256.

Force with `--device cuda|cpu`, `--hidden 512`, `--min-free-gb 0`.

## Each epoch

1. A random past day is drawn (uniformly, with replacement).
2. Predictor + analyzer train on that day + a 30-day replay buffer.
3. The trader trades all 1,440 minutes with $20 (max 5× leverage per coin,
   taker fee 0.05 % + slippage 0.02 %, margin calls simulated), reward =
   log-equity change + bonus for finishing ≥ $30.
4. Everything is saved to `runs/`.

## Outputs (all PNG, every epoch)

```
runs/
  epochs/epoch_00042_2024-03-17.png   <- the advisors' response to that day's market:
                                          prices + exposure shading per coin, equity curve
                                          vs $20/$30, predictor forecast vs realized,
                                          analyzer regime read
  training_curves.png                  <- all training metrics (predictor loss/MAE, analyzer
                                          accuracy, PPO losses/entropy, epoch equity, eval)
  metrics.csv                          <- the same numbers as data
  ckpt/{predictor,analyzer,trader}.pt  <- latest checkpoints (+ ckpt/best by eval equity)
  resources.txt, config.json
```

Held-out evaluation: the most recent `--val-days` (default 40) days are **never**
trained on; every `--eval-every` (25) epochs the deterministic policy runs on
them and the results are logged (`eval_mean_eq`, `eval_hit`).

```bash
python eval.py --ckpt runs/ckpt --days 40   # full held-out report (runs/eval/*.png + CSV)
```

## Key flags

```
--epochs 2000        --capital 20   --target 30    --max-lev 5.0
--fee 0.0005         --slip 0.0002  --val-days 40  --eval-every 25
--device auto        --hidden auto  --min-free-gb 3.0
--resume runs/ckpt   --plot-every 1 --max-days 0 (all)
```

## Honest notes

* Turning $20 into $30 in **one day** is a +50 % daily return — extreme. With
  5× leverage, good days make it possible, but expect a low hit-rate, and the
  held-out eval is the number to trust (training-day equity can overfit).
* Costs (fees + slippage) are simulated on every rebalance; churning dies to
  fees quickly, which the agent must learn to avoid.
* This is a research toy, not financial advice.
