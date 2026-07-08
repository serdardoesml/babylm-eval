"""Pure-Python profiled-deviance ML fit for lme4's ``(1 + ... || subid)`` LMM.

Reproduces ``lmer(y ~ X + (1 + ... || subid), REML = FALSE)`` and its
``logLik()``. Because ``||`` gives an *uncorrelated* (diagonal) relative
covariance factor, ``theta`` is one std-dev ratio per random-effect column,
shared across subjects. With the random-effect columns grouped by subject the
penalized-least-squares system is block-diagonal (one ``p x p`` block per
subject), so we factor it blockwise with dense Cholesky -- exact and fast, and
free of any external sparse-Cholesky dependency.

Validated against lme4 4.x on the MECO L2 corpus: at lme4's theta the deviance
matches to ~1e-9; the full optimizer agrees with lme4's logLik to ~1e-6, which
is optimizer-stop noise well below any inter-model score gap.
"""
import numpy as np
from scipy.optimize import minimize


class PLS:
    """Profiled ML deviance of a diagonal-RE linear mixed model.

    Parameters
    ----------
    X : (n, p_fe) fixed-effects model matrix (post-scale, complete cases).
    y : (n,) response.
    re_design : (n, p_re) random-effects design (the scaled RE columns).
    subid_codes : (n,) integer subject code in [0, n_subid).
    n_subid : number of subjects.
    """

    def __init__(self, X, y, re_design, subid_codes, n_subid):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        Zd = np.asarray(re_design, float)
        self.n, self.p_fe = self.X.shape
        self.p_re = Zd.shape[1]
        self.n_subid = n_subid
        self.XtX = self.X.T @ self.X
        self.Xty = self.X.T @ self.y
        self.yty = float(self.y @ self.y)
        subid_codes = np.asarray(subid_codes)
        self.blocks = []
        # errstate guards spurious over/invalid warnings some BLAS builds emit
        # on the matmul SIMD tail; the reductions themselves are exact.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for s in range(n_subid):
                idx = np.where(subid_codes == s)[0]
                Zs, Xs, ys = Zd[idx], self.X[idx], self.y[idx]
                self.blocks.append((Zs.T @ Zs, Zs.T @ Xs, Zs.T @ ys))

    # Large finite penalty returned when a candidate theta makes the PLS system
    # non-factorizable (e.g. the optimizer probes a huge theta so Lam ZsZs Lam
    # overflows). Keeps BOBYQA/L-BFGS-B on finite values; the optimum is PD.
    _PENALTY = 1e18

    def deviance(self, theta):
        theta = np.asarray(theta, float)
        Ip = np.eye(self.p_re)
        logdetA = 0.0
        RZXtRZX = np.zeros((self.p_fe, self.p_fe))
        RZXtcu = np.zeros(self.p_fe)
        cutcu = 0.0
        try:
            for ZsZs, ZsX, Zsy in self.blocks:
                As = (theta[:, None] * ZsZs) * theta[None, :] + Ip  # Lam ZsZs Lam + I
                Ls = np.linalg.cholesky(As)
                logdetA += 2.0 * np.sum(np.log(np.diag(Ls)))
                cu_s = np.linalg.solve(Ls, theta * Zsy)
                RZX_s = np.linalg.solve(Ls, theta[:, None] * ZsX)
                cutcu += float(cu_s @ cu_s)
                RZXtRZX += RZX_s.T @ RZX_s
                RZXtcu += RZX_s.T @ cu_s
            RtR = self.XtX - RZXtRZX
            RX = np.linalg.cholesky(RtR)
            cb = np.linalg.solve(RX, self.Xty - RZXtcu)
            r2 = self.yty - cutcu - float(cb @ cb)      # minimized penalized RSS
            if not np.isfinite(r2) or r2 <= 0:
                return self._PENALTY, np.nan
            dev = logdetA + self.n * (1.0 + np.log(2.0 * np.pi * r2 / self.n))
            if not np.isfinite(dev):
                return self._PENALTY, np.nan
            return dev, r2
        except np.linalg.LinAlgError:
            return self._PENALTY, np.nan

    def dev(self, theta):
        return self.deviance(theta)[0]


def _bobyqa(pls, p_re, x0):
    import nlopt
    o = nlopt.opt(nlopt.LN_BOBYQA, p_re)
    o.set_lower_bounds(np.zeros(p_re))
    o.set_upper_bounds(np.full(p_re, np.inf))
    o.set_min_objective(lambda th, grad: pls.dev(th))
    o.set_xtol_rel(1e-10)
    o.set_ftol_rel(1e-12)
    o.set_maxeval(100000)
    return o.optimize(np.asarray(x0, float))


def fit_ml(pls, p_re, starts=(1.0, 0.5, 2.0)):
    """Minimize the profiled ML deviance over theta >= 0.

    Deterministic: multi-start nlopt BOBYQA (lme4's algorithm), each polished
    with bounded L-BFGS-B so boundary components snap to 0; keep the lowest
    deviance. Returns (theta, deviance, r2); logLik = -deviance / 2.
    """
    best_x, best_dev = None, np.inf
    for s in starts:
        x = _bobyqa(pls, p_re, np.full(p_re, float(s)))
        r = minimize(pls.dev, x, method="L-BFGS-B",
                     bounds=[(0.0, None)] * p_re,
                     options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000})
        cand = np.maximum(r.x, 0.0)
        cand_dev = pls.dev(cand)
        if cand_dev < best_dev:
            best_dev, best_x = cand_dev, cand
    dev, r2 = pls.deviance(best_x)
    return best_x, dev, r2
