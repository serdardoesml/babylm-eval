"""Tie the front-end frame to the PLS fit: delta log-likelihood per language.

normalized_loglik = logLik(surprisal model) - logLik(baseline model), matching
meco_l2.R / meco_l1.R. scale() is evaluated over the whole fit frame (as in the
lmer formula) and only then are NA-response rows dropped (lmer's na.omit).
"""
import numpy as np
import pandas as pd

from .lmm import PLS, fit_ml
from .frame import RE_TERMS_FULL, RE_TERMS_BASE


def _scale_full(col):
    """R scale() as used inside the lmer formula: centre by mean, divide by
    sample sd (ddof=1). Computed over the whole column but ignoring NaN, so a
    predictor with a few unmatched (NaN) values is not poisoned; those rows are
    then dropped by the complete-case na.omit below (as lmer's na.action does).
    """
    x = np.asarray(col, float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)


def loglik(fit_frame, terms):
    y = fit_frame["firstrun.dur"].to_numpy(float)
    scaled = {t: _scale_full(fit_frame[t]) for t in terms}
    # lmer na.omit: drop rows with NA in the response or ANY model term.
    keep = ~np.isnan(y)
    for t in terms:
        keep &= ~np.isnan(scaled[t])
    X = np.column_stack([np.ones(len(fit_frame))] + [scaled[t] for t in terms])[keep]
    subs = pd.Categorical(fit_frame["subid"].to_numpy()[keep])
    pls = PLS(X, y[keep], X.copy(), subs.codes, len(subs.categories))
    _, dev, _ = fit_ml(pls, X.shape[1])
    return -dev / 2.0


def normalized_loglik(fit_frame):
    ll_full = loglik(fit_frame, RE_TERMS_FULL)
    ll_base = loglik(fit_frame, RE_TERMS_BASE)
    return ll_full, ll_base, ll_full - ll_base
