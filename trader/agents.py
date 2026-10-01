"""The three agents.

- Predictor: price-forecast agent. GRU over the day's features; at every minute
  it forecasts the cumulative log-return (%) over the next 15 and 60 minutes
  for each coin, using only information available up to that minute.
- Analyzer: market-analysis agent. GRU encoder producing a market-state
  embedding, with volatility-regime / trend heads used as its training signal.
- Trader: trade-maker agent. Gaussian policy over per-coin target portfolio
  weights + a value head, trained with PPO.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class Predictor(nn.Module):
    def __init__(self, n_features, n_out, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):  # (B,T,F) -> (B,T,n_out)
        h, _ = self.gru(x)
        return self.head(h)


class Analyzer(nn.Module):
    def __init__(self, n_features, n_pairs=3, hidden=64, emb_dim=16, n_classes=3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.emb = nn.Sequential(nn.Linear(hidden, emb_dim), nn.Tanh())
        self.vol_head = nn.Linear(emb_dim, n_pairs * n_classes)
        self.trend_head = nn.Linear(emb_dim, n_pairs * n_classes)
        self.n_pairs = n_pairs
        self.n_classes = n_classes

    def forward(self, x):  # -> emb (B,T,E), vol (B,T,P,C), trend (B,T,P,C)
        h, _ = self.gru(x)
        e = self.emb(h)
        v = self.vol_head(e).reshape(*e.shape[:-1], self.n_pairs, self.n_classes)
        t = self.trend_head(e).reshape(*e.shape[:-1], self.n_pairs, self.n_classes)
        return e, v, t


class Trader(nn.Module):
    def __init__(self, obs_dim, act_dim=3, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.25))
        self.vf = nn.Linear(hidden, 1)
        self.act_dim = act_dim

    def forward(self, obs):
        h = self.trunk(obs)
        mu = self.mu(h)
        std = self.log_std.clamp(-3.0, 1.0).exp().expand_as(mu)
        value = self.vf(h).squeeze(-1)
        return mu, std, value

    def act(self, obs, deterministic=False):
        mu, std, value = self.forward(obs)
        if deterministic:
            return mu.clamp(-1.0, 1.0), None, value
        dist = Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a.clamp(-1.0, 1.0), logp, value
