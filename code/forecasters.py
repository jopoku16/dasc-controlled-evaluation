"""
forecasters.py: causal point forecasters shared by every conformal method.

Every forecaster returns an array pred with pred[t] a function of y[0..t-1]
(and of calendar information for time t, which is known in advance).  pred[t]
is NaN while the forecaster is not yet available.

  seasonal_naive(y, lag)          pred[t] = y[t - lag]
  rolling_ridge(...)              ridge regression on lagged values (and
                                  optional calendar terms), refitted every
                                  `refit_every` steps on the most recent
                                  `train_window` rows whose targets are all
                                  at times < refit time; predictions between
                                  refits use the frozen model
  rolling_xgboost(...)            same scheme with a gradient-boosting model

At a refit time r the training rows are s in [r - train_window, r - 1] with
all lags available; row s uses target y[s] and features built from y[:s].
Standardisation statistics are computed on the training rows only.
"""

from __future__ import annotations

import numpy as np


def seasonal_naive(y, lag):
    y = np.asarray(y, dtype=float)
    pred = np.full(len(y), np.nan)
    pred[lag:] = y[:-lag]
    return pred


def lag_design(y, lags, extra=None):
    """X[t] = (y[t-lag] for lag in lags) + extra[t]; NaN where a lag is unavailable.

    extra: optional (n, k) array of calendar or other covariates known at time t
    without using y[t:].  Rolling means are built from y[:t] only.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    cols = []
    for lag in lags:
        c = np.full(n, np.nan)
        c[lag:] = y[:-lag]
        cols.append(c)
    X = np.column_stack(cols)
    if extra is not None:
        X = np.column_stack([X, np.asarray(extra, dtype=float)])
    return X


def rolling_means(y, windows):
    """m[t] = mean(y[t-w .. t-1]) for each w (NaN until w values exist)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.full((n, len(windows)), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    for j, w in enumerate(windows):
        t = np.arange(w, n)
        out[t, j] = (cs[t] - cs[t - w]) / w
    return out


def calendar_terms(hours=None, dows=None, period_hour=24, period_dow=7):
    """sin/cos encodings of hour of day and day of week (known in advance)."""
    cols = []
    if hours is not None:
        h = np.asarray(hours, dtype=float)
        cols += [np.sin(2 * np.pi * h / period_hour), np.cos(2 * np.pi * h / period_hour)]
    if dows is not None:
        d = np.asarray(dows, dtype=float)
        cols += [np.sin(2 * np.pi * d / period_dow), np.cos(2 * np.pi * d / period_dow)]
    return np.column_stack(cols) if cols else None


def _rolling_refit(y, X, first_fit, train_window, refit_every, fit_predict):
    """Generic rolling-refit driver.

    fit_predict(X_train, y_train, X_future) -> predictions for X_future rows.
    Predictions for t in [r, r + refit_every) come from the model fitted at r,
    whose training rows all have index < r.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    pred = np.full(n, np.nan)
    row_ok = ~np.any(np.isnan(X), axis=1)
    r = int(first_fit)
    while r < n:
        lo = max(0, r - train_window)
        train = np.arange(lo, r)
        train = train[row_ok[train]]
        stop = min(n, r + refit_every)
        future = np.arange(r, stop)
        future = future[row_ok[future]]
        if len(train) >= 10 and len(future) > 0:
            pred[future] = fit_predict(X[train], y[train], X[future])
        r = stop
    return pred


def rolling_ridge(y, X, first_fit, train_window, refit_every, ridge_alpha=1.0):
    """Ridge on standardised columns of X, refit on a rolling window."""
    from sklearn.linear_model import Ridge

    def fit_predict(Xtr, ytr, Xf):
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        model = Ridge(alpha=ridge_alpha)
        model.fit((Xtr - mu) / sd, ytr)
        return model.predict((Xf - mu) / sd)

    return _rolling_refit(y, X, first_fit, train_window, refit_every, fit_predict)


def rolling_xgboost(y, X, first_fit, train_window, refit_every, seed=0,
                    n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=1):
    """Gradient boosting (xgboost) refit on a rolling window.

    Single-threaded by default: with several processes on one machine the
    library's multi-threaded fit can spin-wait and run a hundred times slower.
    """
    from xgboost import XGBRegressor

    def fit_predict(Xtr, ytr, Xf):
        model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0, n_jobs=n_jobs,
        )
        model.fit(Xtr, ytr)
        return model.predict(Xf)

    return _rolling_refit(y, X, first_fit, train_window, refit_every, fit_predict)
