"""Trading environment: one episode = one full UTC day of 1-minute bars.

The agent starts every epoch with `capital` ($20) and must reach `target`
($30+). Each minute it outputs target portfolio weights in [-1, 1] per coin
(negative = short). Exposure per coin is weight * max_lev * equity. Trades
execute at the current minute's close with a taker fee + slippage. A margin
call liquidates the book if equity falls below the maintenance margin.
"""

import numpy as np

from .data import PAIRS
from .features import N_FEATURES, W


class TradingEnv:
    def __init__(
        self,
        day,
        feats,
        forecast,
        emb,
        capital=20.0,
        target=30.0,
        fee=5e-4,
        slip=2e-4,
        max_lev=5.0,
        maint_margin=0.01,
        terminal_bonus=0.5,
        reward_scale=100.0,
    ):
        self.close = np.stack([day[p]["close"] for p in PAIRS], axis=1).astype(np.float64)
        self.feats = np.asarray(feats, dtype=np.float32)
        self.forecast = np.asarray(forecast, dtype=np.float32)
        self.emb = np.asarray(emb, dtype=np.float32)
        self.capital = float(capital)
        self.target = float(target)
        self.fee = float(fee)
        self.slip = float(slip)
        self.max_lev = float(max_lev)
        self.maint_margin = float(maint_margin)
        self.terminal_bonus = float(terminal_bonus)
        self.reward_scale = float(reward_scale)
        self.T = len(self.close)
        self.n_steps = self.T - 1 - (W - 1)
        if forecast.shape[0] != self.T or emb.shape[0] != self.T:
            raise ValueError("forecast/emb must cover every minute of the day")
        self.obs_dim = W * N_FEATURES + forecast.shape[1] + emb.shape[1] + 3 + 1

    # ------------------------------------------------------------------ #
    def reset(self):
        self.t = W - 1
        self.step_i = 0
        self.cash = self.capital
        self.units = np.zeros(3, dtype=np.float64)
        self.equity = self.capital
        self.margin_call = False
        self.weights_hist = np.zeros((self.n_steps, 3), dtype=np.float32)
        self.equity_hist = np.zeros(self.n_steps, dtype=np.float32)
        return self._obs()

    def _obs(self):
        t = self.t
        win = self.feats[t - W + 1 : t + 1].reshape(-1)
        w_now = (self.units * self.close[t]) / max(self.equity, 1e-6) / self.max_lev
        w_now = np.clip(w_now, -1.5, 1.5)
        eq = np.clip(np.log(max(self.equity, 1e-9) / self.capital), -4.0, 4.0)
        obs = np.concatenate([win, self.forecast[t], self.emb[t], w_now, [eq]])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:3], -1.0, 1.0)
        t = self.t
        prices = self.close[t]
        eq_prev = self.cash + float(self.units @ prices)

        # rebalance to the requested weights at this minute's close.
        # Self-financing: cash pays for the net notional traded (cash may go
        # negative = margin borrowing), so a trade only costs fees+slippage.
        tgt_notional = a * self.max_lev * eq_prev
        tgt_units = tgt_notional / np.maximum(prices, 1e-9)
        delta = tgt_units - self.units
        trade_value = float(delta @ prices)  # + = net buying
        turnover = float(np.abs(delta) @ prices)
        self.cash -= trade_value + turnover * (self.fee + self.slip)
        self.units = tgt_units

        # one minute passes
        self.t += 1
        p_next = self.close[self.t]
        equity = self.cash + float(self.units @ p_next)

        gross = float(np.abs(self.units) @ p_next)
        done = False
        if gross > 0 and equity < self.maint_margin * gross:
            # margin call: liquidate everything
            equity = max(equity - gross * (self.fee + self.slip), 1e-9)
            self.cash, self.units = equity, np.zeros(3)
            self.margin_call = True
            done = True
        if equity <= 1e-6:
            equity = 1e-9
            self.cash, self.units = equity, np.zeros(3)
            done = True

        self.equity = equity
        r = float(
            np.clip(
                self.reward_scale * np.log(max(equity, 1e-9) / max(eq_prev, 1e-9)),
                -50.0,
                50.0,
            )
        )
        if self.t >= self.T - 1:
            done = True
        if done and equity >= self.target:
            r += self.terminal_bonus * self.reward_scale

        if self.step_i < self.n_steps:
            self.weights_hist[self.step_i] = a
            self.equity_hist[self.step_i] = equity
        self.step_i += 1
        return self._obs(), r, done, {
            "equity": equity,
            "margin_call": self.margin_call,
            "steps": self.step_i,
        }
