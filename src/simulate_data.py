"""Simulated Morning Traffic Dataset Generator.

Generates realistic 5-minute interval road segment traffic data for the 8:00 AM - 10:00 AM
morning arrival rush at a campus gate.

Key Dynamics Modeled:
- Explicit distinction between Demand (arrivals), Capacity (road + gate), and Queue (unserved).
- Skewed arrival bell curve peaking between 8:30 and 9:15 AM (center ~8:50 AM).
- Weather capacity reduction (friction) and Event gate-security throttling + arrival surge.
- Queue accumulation carrying over between 5-min intervals within the same morning (reset at 8:00 AM).
- AR(1) observational variability initialized from the stationary distribution.
- Strict score computation [0, 100] and class binning (Low, Moderate, High).
- Monotone decreasing speed ratio to align with Phase 4 live API schema.
- Data source labeled 'simulated' on every row.
"""

import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src import config as cfg


def _compute_mean_arrival_demand(
    minute_of_window: int,
    is_weekend: bool,
    morning_demand_mult: float,
    event_surge_mult: float,
) -> float:
    """Compute the expected arrival rate (lambda) for a 5-minute interval."""
    # Baseline off-peak arrival rate (8:00-8:25 and 9:30-10:00)
    base_offpeak = 35.0

    # Skewed peak arrival curve (peaking ~8:50 AM, which is minute 50)
    # Bell curve component centered around minute 50
    diff = minute_of_window - cfg.PEAK_CENTER_MINUTE
    peak_surge = 55.0 * np.exp(-0.5 * (diff / cfg.PEAK_SIGMA_MINUTES) ** 2)

    # Slight early skew: arrivals ramp up faster from 8:30 to 8:50 than they ramp down
    if 30 <= minute_of_window <= 50:
        peak_surge *= 1.10

    expected_demand = (base_offpeak + peak_surge) * morning_demand_mult

    # Weekend reduction: scale baseline arrival demand down
    if is_weekend:
        expected_demand *= cfg.WEEKEND_DEMAND_MULTIPLIER

    # Event surge: applies to both weekday and weekend base demand
    expected_demand *= event_surge_mult

    return float(max(5.0, expected_demand))


def simulate_traffic_dataset(
    num_days: int = cfg.NUM_DAYS,
    seed: int = cfg.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the full simulated traffic dataset along with latent diagnostic logs.

    Parameters
    ----------
    num_days : int
        Number of consecutive calendar days to simulate.
    seed : int
        Random seed for full reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        public_df : DataFrame of public observable features and target labels.
        diagnostics_df : DataFrame of latent variables (capacity, queue, noise) for verification.
    """
    rng = cfg.get_rng(seed)

    dates = pd.date_range(start=cfg.START_DATE, periods=num_days, freq="D")
    
    public_rows = []
    diagnostic_rows = []

    # Weather probability distribution
    weather_probs = [cfg.WEATHER_PROBABILITIES[w] for w in cfg.WEATHER_CATEGORIES]

    for day_idx, current_date in enumerate(dates):
        date_str = current_date.strftime("%Y-%m-%d")
        day_of_week = int(current_date.weekday())  # 0=Monday, 6=Sunday
        is_weekend = bool(day_of_week >= 5)

        # 1. Per-morning draws (constant through the entire morning window)
        # Weather draw
        weather = str(rng.choice(cfg.WEATHER_CATEGORIES, p=weather_probs))
        f_weather = cfg.WEATHER_CAPACITY_REDUCTION[weather]

        # Event flag draw (~10% of mornings)
        is_event = bool(rng.random() < cfg.EVENT_PROBABILITY)
        event_flag = int(is_event)

        # Event multipliers
        if is_event:
            event_surge_mult = 1.0 + float(rng.uniform(*cfg.EVENT_DEMAND_SURGE_RANGE))
            event_gate_reduction = float(rng.uniform(*cfg.EVENT_GATE_REDUCTION_RANGE))
        else:
            event_surge_mult = 1.0
            event_gate_reduction = 0.0

        # Per-morning general demand multiplier (lognormal centered at 1.0)
        morning_demand_mult = float(rng.lognormal(mean=0.0, sigma=0.08))

        # Latent state variables initialized at the start of the morning (8:00 AM)
        queue_t = 0.0
        # AR(1) noise initialized from stationary distribution N(0, sigma^2)
        eps_t = float(rng.normal(loc=0.0, scale=cfg.AR1_MARGINAL_SIGMA))

        # 2. Iterate through the 25 five-minute intervals (8:00 to 10:00 inclusive)
        for interval_idx in range(cfg.INTERVALS_PER_MORNING):
            minute_of_window = interval_idx * cfg.INTERVAL_MINUTES
            hour = cfg.WINDOW_START_HOUR + (minute_of_window // 60)
            minute = minute_of_window % 60
            timestamp_str = f"{date_str} {hour:02d}:{minute:02d}:00"

            # Compute arrival demand
            mean_demand = _compute_mean_arrival_demand(
                minute_of_window=minute_of_window,
                is_weekend=is_weekend,
                morning_demand_mult=morning_demand_mult,
                event_surge_mult=event_surge_mult,
            )
            # Draw realized vehicle arrivals from Poisson distribution
            vehicle_count = int(rng.poisson(lam=mean_demand))

            # Compute effective capacity C_t
            gate_jitter = float(rng.normal(loc=0.0, scale=cfg.GATE_JITTER_SIGMA))
            f_gate = max(0.0, min(0.60, event_gate_reduction + gate_jitter))
            c_t = cfg.BASE_CAPACITY * (1.0 - f_weather) * (1.0 - f_gate)
            c_t = max(20.0, c_t)  # Minimum physical throughput guarantee

            # Update latent queue: Q_t = max(0, Q_{t-1} + V_t - C_t)
            queue_t = max(0.0, queue_t + vehicle_count - c_t)

            # Update AR(1) observational noise for t > 0
            if interval_idx > 0:
                innovation = float(rng.normal(loc=0.0, scale=cfg.AR1_INNOVATION_SIGMA))
                eps_t = cfg.AR1_PHI * eps_t + innovation

            # Ground truth congestion score formula:
            # P_t = 65 * (V_t / C_t) + QUEUE_WEIGHT * (Q_t / BASE_CAPACITY) + eps_t
            raw_score = (
                cfg.VOLUME_CAPACITY_WEIGHT * (vehicle_count / c_t)
                + cfg.QUEUE_WEIGHT * (queue_t / cfg.BASE_CAPACITY)
                + eps_t
            )
            congestion_score = float(np.clip(raw_score, 0.0, 100.0))
            congestion_class = cfg.score_to_class(congestion_score)
            speed_ratio = cfg.score_to_speed_ratio(congestion_score)

            public_rows.append({
                "date": date_str,
                "timestamp": timestamp_str,
                "day_index": day_idx,
                "day_of_week": day_of_week,
                "minute_of_window": minute_of_window,
                "vehicle_count": vehicle_count,
                "weather": weather,
                "event_flag": event_flag,
                "congestion_score": round(congestion_score, 2),
                "congestion_class": congestion_class,
                "speed_ratio": round(speed_ratio, 4),
                "data_source": "simulated",
            })

            diagnostic_rows.append({
                "date": date_str,
                "timestamp": timestamp_str,
                "day_index": day_idx,
                "effective_capacity": round(c_t, 2),
                "queue_length": round(queue_t, 2),
                "ar1_noise": round(eps_t, 3),
                "raw_score_pre_clip": round(raw_score, 2),
            })

    public_df = pd.DataFrame(public_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    return public_df, diagnostics_df


def main() -> None:
    """Generate and save simulated datasets to disk."""
    print(f"Generating simulated dataset for {cfg.NUM_DAYS} days ({cfg.NUM_DAYS * cfg.INTERVALS_PER_MORNING} rows)...")
    public_df, diagnostics_df = simulate_traffic_dataset(num_days=cfg.NUM_DAYS, seed=cfg.RANDOM_SEED)

    # Save to disk
    public_df.to_csv(cfg.RAW_DATA_PATH, index=False)
    diagnostics_df.to_csv(cfg.DIAGNOSTICS_DATA_PATH, index=False)

    print(f"[OK] Saved public simulated data to: {cfg.RAW_DATA_PATH} ({len(public_df)} rows)")
    print(f"[OK] Saved latent diagnostics to: {cfg.DIAGNOSTICS_DATA_PATH} ({len(diagnostics_df)} rows)")
    print("\nClass distribution preview:")
    print(public_df["congestion_class"].value_counts(normalize=True).round(3) * 100)


if __name__ == "__main__":
    main()
