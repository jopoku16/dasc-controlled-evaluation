"""
datasets.py: real-data loaders, forecaster builders and simulation designs.

Data root: the environment variable DASC_DATA_ROOT, or ../data/real relative
to this file.  Expected layout (as in the original project):
    <root>/uci_household_power/household_power_consumption.txt
    <root>/open_meteo_dallas/open_meteo_dallas_hourly_temperature.csv
    <root>/fred_sp500/sp500_daily_fred.csv
Derived series are cached under ../results/derived/ so the slow electricity
load (about 15 s) runs once.

Each dataset entry provides
    load()                -> dict(y, datetime, hour, dow)
    forecasts(y, meta)    -> dict forecaster name -> pred array (causal)
    cfg                   -> dict with start, dev_end, pools, fws, block_len,
                             worst_block_len, clip_zero, groups(meta) -> dict
                             of grouping arrays for local coverage
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import forecasters as fc

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
DERIVED = RESULTS / "derived"


def data_root():
    env = os.environ.get("DASC_DATA_ROOT")
    return Path(env) if env else (HERE.parent / "data" / "real")


# ----------------------------------------------------------------------------
# Real data
# ----------------------------------------------------------------------------


def load_electricity(max_hours=12000):
    """Hourly mean of Global_active_power, first `max_hours` non-missing hourly rows.

    As in the original project the series is row-based: hourly rows with no
    observations are dropped, so two gaps (62 h and 2 h) exist inside the first
    12,000 rows; the lag-24 forecaster is misaligned for 24 rows after the long
    gap.  True datetimes are kept for hour-of-day grouping.
    """
    DERIVED.mkdir(parents=True, exist_ok=True)
    cache = DERIVED / "electricity_hourly.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["datetime"])
    else:
        txt = data_root() / "uci_household_power" / "household_power_consumption.txt"
        if not txt.exists():
            raise FileNotFoundError(f"missing {txt}; set DASC_DATA_ROOT")
        raw = pd.read_csv(txt, sep=";", na_values="?",
                          usecols=["Date", "Time", "Global_active_power"], low_memory=False)
        dt = pd.to_datetime(raw["Date"] + " " + raw["Time"], dayfirst=True, errors="coerce")
        raw = raw.assign(datetime=dt).dropna(subset=["datetime", "Global_active_power"])
        raw = raw.set_index("datetime").sort_index()
        hourly = raw["Global_active_power"].resample("h").mean().dropna()
        df = hourly.reset_index().rename(columns={"Global_active_power": "y"})
        df.to_csv(cache, index=False)
    df = df.iloc[:max_hours].reset_index(drop=True)
    return _meta(df)


def load_weather(max_hours=24000):
    path = data_root() / "open_meteo_dallas" / "open_meteo_dallas_hourly_temperature.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; set DASC_DATA_ROOT")
    df = pd.read_csv(path, parse_dates=["datetime"]).rename(columns={"temperature_2m": "y"})
    df = df.iloc[:max_hours].reset_index(drop=True)
    return _meta(df)


def load_finance():
    """Absolute daily log returns of the S&P 500 (FRED), in percentage points."""
    path = data_root() / "fred_sp500" / "sp500_daily_fred.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; set DASC_DATA_ROOT")
    df = pd.read_csv(path, parse_dates=["observation_date"]).sort_values("observation_date")
    df["SP500"] = pd.to_numeric(df["SP500"], errors="coerce")
    df = df.dropna(subset=["SP500"])
    close = df["SP500"].to_numpy(dtype=float)
    y = 100 * np.abs(np.diff(np.log(close)))
    out = pd.DataFrame({"datetime": df["observation_date"].to_numpy()[1:], "y": y})
    return _meta(out.reset_index(drop=True))


def _meta(df):
    dt = pd.to_datetime(df["datetime"])
    return {
        "y": df["y"].to_numpy(dtype=float),
        "datetime": dt.to_numpy(),
        "hour": dt.dt.hour.to_numpy(),
        "dow": dt.dt.dayofweek.to_numpy(),
        "month": dt.dt.to_period("M").astype(str).to_numpy(),
        "year": dt.dt.year.to_numpy(),
    }


# forecaster builders -----------------------------------------------------------

HOURLY_LAGS = (1, 2, 3, 24, 48, 168)
DAILY_LAGS = (1, 2, 3, 5, 10, 21)


def hourly_forecasts(y, meta, with_xgb=True):
    X = fc.lag_design(y, HOURLY_LAGS, fc.calendar_terms(meta["hour"], meta["dow"]))
    out = {"naive24": fc.seasonal_naive(y, 24),
           "ridge": fc.rolling_ridge(y, X, first_fit=900, train_window=1440, refit_every=168)}
    if with_xgb:
        out["xgb"] = fc.rolling_xgboost(y, X, first_fit=900, train_window=1440, refit_every=168)
    return out


def daily_forecasts(y, meta, with_xgb=True):
    X = fc.lag_design(y, DAILY_LAGS, fc.rolling_means(y, (5, 21)))
    out = {"naive1": fc.seasonal_naive(y, 1),
           "ridge": fc.rolling_ridge(y, X, first_fit=126, train_window=504, refit_every=21)}
    if with_xgb:
        out["xgb"] = fc.rolling_xgboost(y, X, first_fit=126, train_window=504, refit_every=21)
    return out


def sim_forecasts(y, meta=None, with_xgb=False):
    X = fc.lag_design(y, (1, 2, 3, 4, 5, 6))
    return {"naive1": fc.seasonal_naive(y, 1),
            "ridge": fc.rolling_ridge(y, X, first_fit=120, train_window=300, refit_every=50)}


# grouping for local coverage ---------------------------------------------------


def hourly_groups(meta, dev_end):
    return {"hour": meta["hour"].astype(str), "month": meta["month"]}


def finance_groups(meta, dev_end):
    y = meta["y"]
    vol = fc.rolling_means(y, (21,))[:, 0]           # mean |r| over the past 21 days
    ref = vol[21:dev_end]
    ref = ref[np.isfinite(ref)]
    cuts = np.quantile(ref, [1 / 3, 2 / 3])          # terciles fixed on the dev prefix
    bucket = np.full(len(y), "vol_mid", dtype=object)
    bucket[vol <= cuts[0]] = "vol_low"
    bucket[vol > cuts[1]] = "vol_high"
    bucket[~np.isfinite(vol)] = "vol_na"
    return {"vol": bucket, "year": meta["year"].astype(str)}


REAL_DATASETS = {
    "electricity": dict(
        load=load_electricity, forecasts=hourly_forecasts, groups=hourly_groups,
        start=1080, dev_frac=0.25, pools=(720, 1440), fws=(168, 336),
        block_len=168, worst_block_len=168, clip_zero=False,
        default_fw=168, default_pool=1440, default_m_min=336, legacy_recent=64,
    ),
    "weather": dict(
        load=load_weather, forecasts=hourly_forecasts, groups=hourly_groups,
        start=1440, dev_frac=0.25, pools=(1080, 2160), fws=(168, 336),
        block_len=168, worst_block_len=168, clip_zero=False,
        default_fw=336, default_pool=2160, default_m_min=504, legacy_recent=64,
    ),
    "finance": dict(
        load=load_finance, forecasts=daily_forecasts, groups=finance_groups,
        start=252, dev_frac=0.25, pools=(252, 504), fws=(32, 64),
        block_len=21, worst_block_len=63, clip_zero=True,
        default_fw=64, default_pool=504, default_m_min=126, legacy_recent=64,
    ),
}


# ----------------------------------------------------------------------------
# Simulations
# ----------------------------------------------------------------------------


def sim_original_three_phase(n=1400, seed=7):
    """The original run_first_simulation.simulate_stream: breaks at 450 and 900."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    freq = np.piecewise(
        t, [t < 450, (t >= 450) & (t < 900), t >= 900],
        [0.035, 0.055, lambda z: 0.055 + 0.00006 * (z - 900)],
    )
    amp = np.where((t // 180) % 2 == 0, 1.0, 1.7)
    level = np.where(t >= 900, 1.2, 0.0)
    sigma = np.where(t >= 900, 0.75, 0.45)
    signal = level + amp * np.sin(2 * np.pi * freq * t)
    y = signal + rng.normal(0, sigma)
    regime = np.where(t < 450, "A", np.where(t < 900, "B", "drift"))
    return y, regime


def _two_regime(n, seed, block, f1, f2, a1, a2, s1, s2):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    r = (t // block) % 2
    freq = np.where(r == 0, f1, f2)
    amp = np.where(r == 0, a1, a2)
    sig = np.where(r == 0, s1, s2)
    y = amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, sig)
    regime = np.where(r == 0, "R1", "R2")
    return y, regime


def sim_recurring_freq(n=1400, seed=0):
    """Two recurring regimes differing in frequency only (block 150, equal noise)."""
    return _two_regime(n, seed, 150, 0.03, 0.09, 1.0, 1.0, 0.5, 0.5)


def sim_recurring_freq_noise(n=1400, seed=0):
    """Two recurring regimes differing in frequency and noise scale (block 150)."""
    return _two_regime(n, seed, 150, 0.03, 0.09, 1.0, 1.0, 0.3, 0.9)


def sim_recurring_amplitude(n=1400, seed=0):
    """Two recurring regimes differing in amplitude only (block 150, same frequency)."""
    return _two_regime(n, seed, 150, 0.05, 0.05, 0.5, 2.0, 0.5, 0.5)


def sim_stationary_ar1(n=1400, seed=0, phi=0.7, sd=0.5):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sd, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y, np.full(n, "stationary")


def sim_pure_drift(n=1400, seed=0):
    """Fixed sinusoid, slowly rising level and noise scale, no recurrence."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    level = 0.002 * t
    sigma = 0.3 + 0.7 * t / (n - 1)
    y = level + np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, sigma)
    regime = np.where(t < n // 2, "early", "late")
    return y, regime


def sim_white_noise(n=1400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0, n), np.full(n, "noise")


def sim_stress(scenario, n=1500, seed=0):
    """The five stress scenarios of the original run_stress_tests.py (n = 1500)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base_freq = 0.04
    amp = 1.2 + 0.45 * ((t // 180) % 2)
    level = np.zeros(n)
    sigma = np.full(n, 0.45)
    freq = np.full(n, base_freq)
    noise = rng.normal(0, sigma)
    brk = None
    if scenario == "abrupt_shift":
        level[t >= 900] = 1.4
        sigma[t >= 900] = 0.8
        noise = rng.normal(0, sigma)
        brk = 900
    elif scenario == "gradual_frequency":
        freq = base_freq + 0.00006 * np.maximum(0, t - 500)
        brk = 500
    elif scenario == "heavy_tail":
        noise = 0.35 * rng.standard_t(df=3, size=n)
        sigma[:] = np.std(noise[:400])
        level[t >= 900] = 0.8
        brk = 900
    elif scenario == "mixed_drift":
        level[t >= 850] = 1.0
        sigma[t >= 850] = 0.75
        freq = base_freq + 0.00004 * np.maximum(0, t - 700)
        noise = rng.normal(0, sigma)
        brk = 850
    elif scenario == "weak_recurrence":
        amp = 0.25 + 0.05 * rng.normal(size=n)
        freq = 0.02 + 0.00003 * t
        level[t >= 850] = 0.8
        sigma[t >= 850] = 0.75
        noise = rng.normal(0, sigma)
        brk = 850
    else:
        raise ValueError(scenario)
    y = level + amp * np.sin(2 * np.pi * freq * t) + noise
    regime = np.where(t < brk, "pre", "post")
    return y, regime


SIM_SCENARIOS = {
    # name: (generator, expected to help?, break points for the drift detector)
    "original_three_phase": (sim_original_three_phase, "mixed", (450, 900)),
    "recurring_freq": (sim_recurring_freq, "help with naive, not with ridge", ()),
    "recurring_freq_noise": (sim_recurring_freq_noise, "help", ()),
    "recurring_amplitude": (sim_recurring_amplitude, "no (features blind)", ()),
    "stationary_ar1": (sim_stationary_ar1, "no", ()),
    "pure_drift": (sim_pure_drift, "no", ()),
    "white_noise": (sim_white_noise, "no", ()),
    "stress_abrupt_shift": (lambda n=1500, seed=0: sim_stress("abrupt_shift", n, seed), "no", (900,)),
    "stress_gradual_frequency": (lambda n=1500, seed=0: sim_stress("gradual_frequency", n, seed), "no", (500,)),
    "stress_heavy_tail": (lambda n=1500, seed=0: sim_stress("heavy_tail", n, seed), "no", (900,)),
    "stress_mixed_drift": (lambda n=1500, seed=0: sim_stress("mixed_drift", n, seed), "no", (850,)),
    "stress_weak_recurrence": (lambda n=1500, seed=0: sim_stress("weak_recurrence", n, seed), "no", (850,)),
}

SIM_CFG = dict(start=220, pools=(180, 360), fws=(32, 64), block_len=100,
               worst_block_len=100, clip_zero=False, default_fw=64,
               default_pool=360, default_m_min=80, legacy_recent=64)

TUNING_SEEDS = list(range(1000, 1010))
EVAL_SEEDS = list(range(100))


def make_sim(name, seed):
    gen = SIM_SCENARIOS[name][0]
    y, regime = gen(seed=seed)
    return {"y": y, "regime": regime}
