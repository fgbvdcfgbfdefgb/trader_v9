"""Misc helpers: seeding, hardware detection and the device/model-size plan."""

import os
import platform

import numpy as np
import torch


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def total_ram_gb():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def detect_resources():
    """Inventory of the machine: CPUs, RAM and GPUs (with free VRAM)."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_b, total_b = torch.cuda.mem_get_info(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_gb": round(total_b / 1e9, 2),
                    "free_gb": round(free_b / 1e9, 2),
                }
            )
    return {
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "ram_gb": None if total_ram_gb() is None else round(total_ram_gb(), 2),
        "gpus": gpus,
    }


def format_resources(res) -> str:
    lines = [
        f"host   : {res['hostname']}",
        f"python : {res['python']}  torch={res['torch']}  cuda={res['cuda_available']}",
        f"cpu    : {res['cpu_count']} cores   ram={res['ram_gb']} GB",
    ]
    if res["gpus"]:
        for g in res["gpus"]:
            lines.append(
                f"gpu {g['index']}  : {g['name']}  total={g['total_gb']}GB free={g['free_gb']}GB"
            )
    else:
        lines.append("gpu    : none -> CPU mode")
    return "\n".join(lines)


def choose_plan(args, res) -> dict:
    """Pick devices for the three agents + trader hidden size, from available hardware.

    - >=3 usable GPUs: predictor / analyzer / trade-maker each get a dedicated GPU
      (predictor & analyzer keep training in background threads while the trader
      rolls out -> true parallel training).
    - 2 GPUs: predictor + analyzer share one, trader gets the other.
    - 1 GPU: everything on it.
    - 0 GPUs (or GPUs too busy, e.g. shared with a miner): CPU with a smaller trader.

    A GPU is "usable" when it has at least --min-free-gb free VRAM, so we never
    step on other workloads (e.g. a running miner).
    """
    gpus = res["gpus"]
    if args.device == "cpu":
        usable = []
    else:
        usable = [g for g in gpus if g["free_gb"] >= args.min_free_gb]
        if args.device == "cuda" and not usable and gpus:
            usable = gpus  # explicitly forced: ignore the free-memory guard

    def dev(g):
        return torch.device(f"cuda:{g['index']}")

    if len(usable) >= 3:
        plan = {
            "pred": dev(usable[0]),
            "ana": dev(usable[1]),
            "trade": dev(usable[2]),
            "parallel": True,
            "layout": f"3+ GPUs: predictor={dev(usable[0])}, analyzer={dev(usable[1])}, trader={dev(usable[2])}",
        }
    elif len(usable) == 2:
        plan = {
            "pred": dev(usable[1]),
            "ana": dev(usable[1]),
            "trade": dev(usable[0]),
            "parallel": True,
            "layout": f"2 GPUs: predictor+analyzer={dev(usable[1])}, trader={dev(usable[0])}",
        }
    elif len(usable) == 1:
        plan = {
            "pred": dev(usable[0]),
            "ana": dev(usable[0]),
            "trade": dev(usable[0]),
            "parallel": False,
            "layout": f"1 GPU: all agents on {dev(usable[0])}",
        }
    else:
        plan = {
            "pred": torch.device("cpu"),
            "ana": torch.device("cpu"),
            "trade": torch.device("cpu"),
            "parallel": False,
            "layout": "CPU: all agents on cpu",
        }

    if getattr(args, "hidden", "auto") == "auto":
        if len(usable) >= 2:
            plan["hidden"] = 768
        elif len(usable) == 1:
            plan["hidden"] = 512
        else:
            plan["hidden"] = 256
    else:
        plan["hidden"] = int(args.hidden)
    return plan
