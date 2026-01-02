#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import requests

WEIGHTS_URL = "http://52.198.155.160:8080/weights"


Position = Tuple[str, str]  # (market_code, ticker)


@dataclass(frozen=True)
class Inputs:
    positions: List[Position]
    initial_slots: List[float]
    current_usd: List[float]
    crcl_weight: float
    total_value: float
    leverage: float
    threshold: float


class WeightService:
    """Client for the external /weights service."""

    def __init__(self, url: str = WEIGHTS_URL, timeout_s: float = 10.0) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def get_weights(self, positions: Sequence[Position], slots: Sequence[float]) -> List[float]:
        payload = {
            "positions": [[m, t] for (m, t) in positions],
            "slots": [float(s) for s in slots],
        }
        resp = self.session.post(self.url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()

        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected /weights response: {data}")

        try:
            return [float(x) for x in data]
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Non-numeric weight(s) from /weights: {data}") from e


def compute_scale(inp: Inputs) -> float:
    """
    Total dollar exposure allocated by the weights engine.

    scale = (1 - crcl_weight) * total_value * leverage
    """
    return (1.0 - inp.crcl_weight) * inp.total_value * inp.leverage


def compute_target_usd(weights: Sequence[float], scale: float) -> List[float]:
    """Convert weights to target USD holdings."""
    return [scale * w for w in weights]


def compute_rebalance_usd(
    *,
    ws: WeightService,
    slots: Sequence[float],
    inp: Inputs,
) -> List[float]:
    """
    Rebalance instruction per asset (USD):
      reb[i] = target_usd[i] - current_usd[i]
    """
    weights = ws.get_weights(inp.positions, slots)

    if len(weights) != len(inp.positions):
        raise RuntimeError(f"Weight length mismatch: got {len(weights)}, expected {len(inp.positions)}")
    if len(inp.current_usd) != len(inp.positions):
        raise RuntimeError(f"current_usd length mismatch: got {len(inp.current_usd)}, expected {len(inp.positions)}")

    scale = compute_scale(inp)
    target = compute_target_usd(weights, scale)
    return [t - h for t, h in zip(target, inp.current_usd)]


def argmax_abs(xs: Sequence[float]) -> int:
    """Index of the element with maximum absolute value."""
    if not xs:
        raise ValueError("argmax_abs() requires a non-empty sequence")

    best_i = 0
    best_v = abs(xs[0])
    for i in range(1, len(xs)):
        v = abs(xs[i])
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def optimize_slots(
    inp: Inputs,
    *,
    ws: WeightService | None = None,
    max_iters: int = 50_000,
    verbose_every: int = 0,
) -> List[float]:
    """
    Greedy coordinate search:
      - compute rebalance vector in USD
      - pick the asset with largest |rebalance|
      - adjust only its slot by a fixed per-asset step
      - error if the adjusted asset flips trade side (sign of rebalance)
      - stop once max |rebalance| < threshold
    """
    ws = ws or WeightService()

    # Working copy of slots.
    slots = list(inp.initial_slots)

    # Per-asset step size.
    steps = [s / 16.0 for s in inp.initial_slots]

    # Initial rebalance.
    reb = compute_rebalance_usd(ws=ws, slots=slots, inp=inp)

    for it in range(1, max_iters + 1):
        i = argmax_abs(reb)
        worst = reb[i]

        if abs(worst) < inp.threshold:
            return slots

        step = steps[i]
        if step == 0.0:
            raise RuntimeError(f"Cannot adjust {inp.positions[i]}: initial slot is 0 (step=0)")

        # Direction heuristic (same as original):
        #   if worst < 0 (need sell), increase slot
        #   if worst > 0 (need buy), decrease slot
        slots[i] += step if worst < 0 else -step

        new_reb = compute_rebalance_usd(ws=ws, slots=slots, inp=inp)

        before = reb[i]
        after = new_reb[i]
        if before * after < 0:
            raise RuntimeError(
                f"Trade side flipped at iter {it} for adjusted {inp.positions[i]}: "
                f"I_before={before:.6f}, I_after={after:.6f}"
            )

        if verbose_every and it % verbose_every == 0:
            m, t = inp.positions[i]
            print(
                f"iter={it:>6} {m}:{t:>5} "
                f"I={worst:>12.2f} -> {after:>12.2f} slot={slots[i]:.6f}"
            )

        reb = new_reb

    raise RuntimeError("Did not converge within max_iters")


def parse_floats_block(s: str) -> List[float]:
    return [float(x) for x in s.split()]


def parse_money_block(s: str) -> List[float]:
    return [float(x) for x in s.replace(",", "").split()]


def main() -> None:
    positions: List[Position] = [
        ("c", "AAVE"), ("u", "COIN"), ("u", "HOOD"),
        ("u", "IBIT"), ("t", "2330"), ("t", "3081"), ("t", "3131"),
        ("t", "3443"), ("t", "3661"), ("u", "AMZN"), ("u", "AVGO"),
        ("u", "GOOGL"), ("u", "NVDA"), ("u", "PLTR"), ("u", "TSLA"),
    ]

    initial_slots = parse_floats_block("""
0.250
0.500
0.500
0.500
1.000
0.063
0.125
0.250
0.250
0.188
0.500
1.000
0.750
1.000
1.250
    """)

    current_usd = parse_money_block("""
24,138.197
135,948.000
158,050.890
240,242.683
702,500.877
0.000
0.000
10,333.866
215,468.566
33,028.710
242,082.040
565,563.000
358,151.860
644,706.150
410,486.640
    """)

    inp = Inputs(
        positions=positions,
        initial_slots=initial_slots,
        current_usd=current_usd,
        crcl_weight=2 ** -7,
        total_value=float("2,699,802.986".replace(",", "")),
        leverage=1.401,
        threshold=float("29,545.084".replace(",", "")),
    )

    final_slots = optimize_slots(inp)

    print("\nFINAL SLOTS")
    for (m, t), s0, s1 in zip(inp.positions, inp.initial_slots, final_slots):
        print(f"{m}:{t:>5}  {s0:>8.6f}  {s1:>8.6f}")


if __name__ == "__main__":
    main()
