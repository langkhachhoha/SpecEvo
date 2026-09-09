# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
from typing import Tuple

def circle_packing21() -> np.ndarray:
    """
    Deterministic hybrid global-local optimizer for 21 circles in a rectangle (perimeter 4).
    Pipeline:
      - Global: Differential evolution on a compact staggered-row template with per-row
        horizontal offsets, per-row horizontal skew, per-row alternating vertical jitter,
        and width W. For each candidate, solve an exact LP (HiGHS) to maximize sum radii.
      - Local: Joint nonlinear polish with trust-constr over (W, x, y, r) using sparse
        Jacobians and explicit wall/pairwise constraints.
      - Final: Recompute maximal radii by LP for polished centers to ensure feasibility
        and maximality. Deterministic seeds; graceful SciPy fallback to equal-hex.
    Returns:
        np.ndarray of shape (21, 3): (x, y, r).
    """
    # Lazy imports for environments where SciPy is present
    try:
        from scipy.optimize import linprog, differential_evolution, minimize, Bounds, NonlinearConstraint
        from scipy.sparse import coo_matrix
    except Exception:
        return _equal_hex_baseline()

    n = 21
    rt3 = sqrt(3.0)
    eps_r = 1e-12
    margin = 1e-9

    # ---------- Utilities ----------
    def _equal_hex_baseline():
        r0 = 2.0 / (10.0 + 5.0 * rt3)
        rows = [4, 3, 4, 3, 4, 3]
        pts = []
        for j, k in enumerate(rows):
            y = r0 + j * (rt3 * r0)
            x0 = r0 + (r0 if (j & 1) else 0.0)
            for t in range(k):
                pts.append((x0 + 2.0 * r0 * t, y, r0))
        return np.asarray(pts, float)

    def lp_max_radii(x: np.ndarray, y: np.ndarray, W: float) -> Tuple[np.ndarray, float]:
        """Solve maximize sum r subject to r_i <= walls, r_i + r_j <= d_ij, r_i >= eps via HiGHS."""
        H = 2.0 - W
        if W <= 0.0 or H <= 0.0:
            return None, -np.inf
        nloc = x.size
        if nloc != n or y.size != n:
            return None, -np.inf
        w = np.minimum.reduce([x, W - x, y, H - y])
        if np.any(w <= 0.0):
            return None, -np.inf

        # Build sparse A_ub
        m_pairs = n * (n - 1) // 2
        m = n + m_pairs
        nnz = n + 2 * m_pairs  # walls (n), pairs (2 per pair)
        rows = np.empty(nnz, dtype=int)
        cols = np.empty(nnz, dtype=int)
        data = np.empty(nnz, dtype=float)
        b_ub = np.empty(m, dtype=float)

        # walls
        p = 0
        for i in range(n):
            rows[p] = i
            cols[p] = i
            data[p] = 1.0
            b_ub[i] = float(w[i])
            p += 1
        # pairs
        rix = n
        for i in range(n - 1):
            xi, yi = x[i], y[i]
            for j in range(i + 1, n):
                d = float(np.hypot(xi - x[j], yi - y[j]))
                b_ub[rix] = d
                # r_i + r_j <= d
                rows[p] = rix; cols[p] = i; data[p] = 1.0; p += 1
                rows[p] = rix; cols[p] = j; data[p] = 1.0; p += 1
                rix += 1

        A_ub = coo_matrix((data, (rows, cols)), shape=(m, n))
        c = -np.ones(n)
        bounds = [(eps_r, None)] * n
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return None, -np.inf
        r = res.x
        return r, float(np.sum(r))

    def build_centers_from_template(rows, params):
        """
        Build centers from a staggered-row template with:
          - W: width, vscale: vertical scale in hex units
          - per-row horizontal offset offs[j] (added to base stagger)
          - per-row horizontal skew a[j]: x += a[j]*(t - (k-1)/2)
          - per-row alternating vertical jitter vj[j]*alt_sign(t)
        Then normalize to [margin, W-margin] x [margin, (2-W)-margin].
        """
        R = len(rows)
        W = float(params[0])
        vscale = float(params[1])
        offs = np.asarray(params[2:2 + R], float)
        a = np.asarray(params[2 + R:2 + 2 * R], float)
        vj = np.asarray(params[2 + 2 * R:2 + 3 * R], float)

        pts = []
        for j, k in enumerate(rows):
            base_off = (1.0 if (j & 1) else 0.0) + offs[j]
            cx = (k - 1) * 0.5  # center index for skew
            for t in range(k):
                xx = base_off + 2.0 * t + a[j] * (t - cx)
                sgn = 1.0 if (t & 1) == 0 else -1.0
                yy = j * (vscale * rt3) + vj[j] * sgn
                pts.append((xx, yy))
        P = np.asarray(pts, float)
        P -= P.min(axis=0, keepdims=True)
        span = np.maximum(P.ptp(axis=0), 1e-12)
        H = 2.0 - W
        x = (P[:, 0] / span[0]) * (W - 2 * margin) + margin
        y = (P[:, 1] / span[1]) * (H - 2 * margin) + margin
        return x, y, W

    def de_objective(params, rows):
        x, y, W = build_centers_from_template(rows, params)
        r, s = lp_max_radii(x, y, W)
        return 1e6 if r is None else -s

    def run_de(rows, W_bounds=(0.75, 1.25), v_bounds=(0.90, 1.10),
               off_bounds=(-0.8, 0.8), skew_bounds=(-0.35, 0.35), vj_bounds=(-0.35, 0.35),
               seed=12345, popsize=12, maxiter=45):
        R = len(rows)
        bounds = [W_bounds, v_bounds] + [off_bounds] * R + [skew_bounds] * R + [vj_bounds] * R

        # Compose a deterministic init population with Sobol/LH if available, plus a baseline center
        # Baseline param: W≈1.0, vscale=1.0, offs[j]=0, skew=0, vj=0
        p0 = np.zeros(2 + 3 * R, float)
        p0[0] = 1.0
        p0[1] = 1.0
        init = 'sobol'
        try:
            res = differential_evolution(
                lambda p: de_objective(p, rows),
                bounds=bounds,
                seed=seed,
                strategy='best1bin',
                popsize=popsize,
                tol=1e-6,
                maxiter=maxiter,
                recombination=0.9,
                polish=False,
                updating='deferred',
                workers=1,
                disp=False,
                init=init)
        except TypeError:
            # Older SciPy: no 'init'='sobol'
            res = differential_evolution(
                lambda p: de_objective(p, rows),
                bounds=bounds,
                seed=seed,
                strategy='best1bin',
                popsize=popsize,
                tol=1e-6,
                maxiter=maxiter,
                recombination=0.9,
                polish=False,
                updating='deferred',
                workers=1,
                disp=False)
        p = res.x
        x, y, W = build_centers_from_template(rows, p)
        r, s = lp_max_radii(x, y, W)

        # Two symmetric variants: flip signs of offs/skew/vj deterministically to diversify
        variants = [p, p.copy(), p.copy()]
        variants[1][2:2 + R] *= -1.0
        variants[2][2 + R:2 + 2 * R] *= -1.0
        best_tuple = (s if r is not None else -np.inf, x, y, r if r is not None else None, W)
        for pv in variants[1:]:
            xv, yv, Wv = build_centers_from_template(rows, pv)
            rv, sv = lp_max_radii(xv, yv, Wv)
            if rv is not None and sv > best_tuple[0]:
                best_tuple = (sv, xv, yv, rv, Wv)
        return best_tuple  # (score, x, y, r, W)

    # ---------- Global multi-template search ----------
    templates = [
        [4, 3, 4, 3, 4, 3],
        [3, 4, 3, 4, 3, 4],
        [5, 4, 4, 4, 4],    # 5 rows
        [4, 5, 4, 4, 4],
        [4, 4, 5, 4, 4]
    ]
    candidates = []
    for k, rows in enumerate(templates):
        try:
            cand = run_de(rows, seed=12345 + 7 * k, popsize=12, maxiter=48)
            if cand[3] is not None:
                candidates.append(cand)
        except Exception:
            continue

    # Include equal-hex baseline
    base = _equal_hex_baseline()
    xb, yb, rb = base[:, 0], base[:, 1], base[:, 2]
    Wb = 8.0 * rb[0]
    sb = float(np.sum(rb))
    candidates.append((sb, xb, yb, rb, Wb))

    # ---------- Local NLP polish (trust-constr) ----------
    def nlp_refine(x0, y0, r0, W0):
        # Variables: z = [W, x(21), y(21), r(21)]
        N = n
        z0 = np.concatenate([[W0], x0, y0, r0])

        def fun(z):
            return -np.sum(z[1 + 2 * N:1 + 3 * N])

        def jac(z):
            g = np.zeros_like(z)
            g[1 + 2 * N:1 + 3 * N] = -1.0
            return g

        def wall_fun(z):
            W = z[0]
            H = 2.0 - W
            x = z[1:1 + N]
            y = z[1 + N:1 + 2 * N]
            r = z[1 + 2 * N:1 + 3 * N]
            return np.concatenate([x - r, W - x - r, y - r, H - y - r])

        def wall_jac(z):
            rows = []
            cols = []
            data = []
            off0 = 0
            off1 = off0 + N
            off2 = off1 + N
            off3 = off2 + N
            W_i = 0
            x_i0 = 1
            y_i0 = 1 + N
            r_i0 = 1 + 2 * N
            for i in range(N):
                rows += [off0 + i, off0 + i]
                cols += [x_i0 + i, r_i0 + i]
                data += [1.0, -1.0]
                rows += [off1 + i, off1 + i, off1 + i]
                cols += [W_i, x_i0 + i, r_i0 + i]
                data += [1.0, -1.0, -1.0]
                rows += [off2 + i, off2 + i]
                cols += [y_i0 + i, r_i0 + i]
                data += [1.0, -1.0]
                rows += [off3 + i, off3 + i, off3 + i]
                cols += [W_i, y_i0 + i, r_i0 + i]
                data += [-1.0, -1.0, -1.0]
            J = coo_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))),
                           shape=(4 * N, 1 + 3 * N))
            return J

        pij_i = []
        pij_j = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                pij_i.append(i)
                pij_j.append(j)
        pij_i = np.asarray(pij_i, dtype=int)
        pij_j = np.asarray(pij_j, dtype=int)
        M = pij_i.size

        def pair_fun(z):
            x = z[1:1 + N]
            y = z[1 + N:1 + 2 * N]
            r = z[1 + 2 * N:1 + 3 * N]
            dx = x[pij_i] - x[pij_j]
            dy = y[pij_i] - y[pij_j]
            rr = r[pij_i] + r[pij_j]
            return dx * dx + dy * dy - rr * rr

        def pair_jac(z):
            x = z[1:1 + N]
            y = z[1 + N:1 + 2 * N]
            r = z[1 + 2 * N:1 + 3 * N]
            dx = x[pij_i] - x[pij_j]
            dy = y[pij_i] - y[pij_j]
            rr = r[pij_i] + r[pij_j]
            rows = []
            cols = []
            data = []
            x_i0 = 1
            y_i0 = 1 + N
            r_i0 = 1 + 2 * N
            for k in range(M):
                i = pij_i[k]; j = pij_j[k]
                rows += [k, k]; cols += [x_i0 + i, x_i0 + j]; data += [2.0 * dx[k], -2.0 * dx[k]]
                rows += [k, k]; cols += [y_i0 + i, y_i0 + j]; data += [2.0 * dy[k], -2.0 * dy[k]]
                rows += [k, k]; cols += [r_i0 + i, r_i0 + j]; data += [-2.0 * rr[k], -2.0 * rr[k]]
            J = coo_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))),
                           shape=(M, 1 + 3 * N))
            return J

        lb = np.zeros(1 + 3 * N)
        ub = np.full(1 + 3 * N, 2.0)
        lb[0] = 0.4   # W lower bound
        ub[0] = 1.6   # W upper bound
        lb[1 + 2 * N:1 + 3 * N] = eps_r
        bounds = Bounds(lb, ub)

        wall_c = NonlinearConstraint(wall_fun, 0.0, np.inf, jac=wall_jac)
        pair_c = NonlinearConstraint(pair_fun, 0.0, np.inf, jac=pair_jac)

        res = minimize(fun, z0, method='trust-constr', jac=jac,
                       constraints=[wall_c, pair_c],
                       bounds=bounds,
                       options=dict(maxiter=500, verbose=0,
                                    xtol=1e-12, gtol=1e-10, barrier_tol=1e-12))
        z = res.x if res.success else z0
        W = z[0]
        x = z[1:1 + N]
        y = z[1 + N:1 + 2 * N]
        r, s = lp_max_radii(x, y, W)
        if r is None:
            return x0, y0, r0, W0, float(np.sum(r0))
        return x, y, r, W, s

    # Refine top candidates (including symmetric variants) and take the best
    best_score = -np.inf
    best_pack = None
    for s, x, y, r, W in candidates:
        try:
            xr, yr, rr, Wr, sr = nlp_refine(x, y, r, W)
        except Exception:
            xr, yr, rr, Wr, sr = x, y, r, W, float(np.sum(r))
        if sr > best_score:
            best_score = sr
            best_pack = (xr, yr, rr, Wr)

    # Guard: never worse than baseline
    if best_score + 1e-12 < sb:
        return np.stack([xb, yb, rb], axis=1)

    x_best, y_best, r_best, W_best = best_pack
    return np.stack([x_best, y_best, r_best], axis=1)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    cs = circle_packing21()
    print(f"Radii sum: {np.sum(cs[:,-1])}")