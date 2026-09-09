# EVOLVE-BLOCK-START
"""
Deterministic NLP constructor maximizing sum of radii for 26 circles in the unit square.

Formulation (unchanged):
- Variables: x_i, y_i, r_i, i=1..26
- Maximize sum r_i subject to:
    x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    ||c_i - c_j|| - (r_i + r_j) >= 0 for all i<j, r_i >= 0

Refinements:
- Seed set expanded with geometry-aware staggered hex row templates:
  [5,4,5,4,4,4] (both parities), [5,5,4,4,4,4], and a 7-row shell-like pattern,
  plus previous hex-like/grid seeds. All seeds are explicit and deterministic.
- Two-stage optimization per seed:
  1) trust-constr NLP starting from clearance-feasible radii,
  2) LP radii polish for fixed centers (HiGHS),
  3) short warm re-optimization (trust-constr) now warm-started with LP radii,
  4) final LP polish + tiny safety shrink.
- Analytic Jacobian for constraints to accelerate trust-constr.

Returns:
    centers: (26,2), radii: (26,), sum_radii: float
"""

import numpy as np


def compute_max_radii(centers):
    """Feasible radii from clearance: min(boundary distance, 0.5*nearest-center)."""
    if len(centers) == 0:
        return np.array([], dtype=float)
    x, y = centers[:, 0], centers[:, 1]
    b = np.minimum.reduce((x, 1 - x, y, 1 - y))
    D = centers[:, None, :] - centers[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", D, D)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return np.minimum(b, 0.5 * nn)


def _build_pairs(n):
    """All unordered index pairs (i<j)."""
    return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)


def _lp_radii_for_fixed_centers(centers):
    """
    LP polish for fixed centers:
      maximize sum r_i
      s.t. 0 <= r_i <= boundary_i, r_i + r_j <= d_ij for all i<j
    Uses SciPy HiGHS; returns radii and flag.
    """
    try:
        from scipy.optimize import linprog
        import scipy.sparse as sp
    except Exception:
        return compute_max_radii(centers), False

    n = len(centers)
    x, y = centers[:, 0], centers[:, 1]
    b = np.minimum.reduce((x, 1 - x, y, 1 - y))

    pairs = _build_pairs(n)
    if len(pairs):
        dij = np.hypot(x[pairs[:, 0]] - x[pairs[:, 1]], y[pairs[:, 0]] - y[pairs[:, 1]])
    else:
        dij = np.array([], dtype=float)

    # Objective: minimize -sum r => c=-1
    c = -np.ones(n, dtype=float)

    # Build sparse A_ub and b_ub with boundary and pairwise constraints
    rows, cols, data, b_ub = [], [], [], []

    # r_i <= b_i
    I = np.arange(n)
    rows.extend(I.tolist()); cols.extend(I.tolist()); data.extend([1.0] * n)
    b_ub.extend(b.tolist())

    # r_i + r_j <= d_ij
    if len(pairs):
        base = len(b_ub)
        pr = np.repeat(np.arange(base, base + len(pairs)), 2)
        rows.extend(pr.tolist())
        cols.extend(np.column_stack([pairs[:, 0], pairs[:, 1]]).ravel().tolist())
        data.extend([1.0] * (2 * len(pairs)))
        b_ub.extend(dij.tolist())

    import scipy.sparse as sp
    A_ub = sp.coo_matrix(
        (np.array(data, float), (np.array(rows, int), np.array(cols, int))),
        shape=(len(b_ub), n)
    ).tocsr()
    b_ub = np.array(b_ub, float)

    bounds = [(0.0, 0.5)] * n  # safe absolute bound; boundary rows are tighter anyway

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"presolve": True})
    if res.success and res.x is not None:
        r = np.asarray(res.x, float)
        return np.clip(r, 0.0, b), True
    return compute_max_radii(centers), False


def _nlp_optimize_with_jac(centers0, r_init=None, maxiter=500):
    """
    Trust-constr NLP with analytic Jacobian.
    Inputs:
      - centers0: (n,2) initial centers (feasible)
      - r_init: optional initial radii; will be feasibility-shrunk
      - maxiter: max outer iterations
    Returns:
      - centers, radii (feasible)
    """
    from scipy.optimize import minimize, NonlinearConstraint, Bounds
    import scipy.sparse as sp

    n = len(centers0)
    pairs = _build_pairs(n)

    x0 = np.clip(centers0[:, 0], 0.0, 1.0)
    y0 = np.clip(centers0[:, 1], 0.0, 1.0)
    rc = compute_max_radii(np.stack([x0, y0], axis=1))
    if r_init is not None and len(r_init) == n:
        r0 = np.minimum(rc, 0.985 * np.maximum(r_init, 0.0))
    else:
        r0 = 0.95 * rc
    z0 = np.concatenate([x0, y0, r0])

    lb = np.concatenate([np.zeros(2 * n), np.zeros(n)])
    ub = np.concatenate([np.ones(2 * n), np.full(n, 0.5)])
    bnds = Bounds(lb, ub)

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_jac(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    # g(z) >= 0 vector of constraints
    def gfun(z):
        x = z[:n]; y = z[n:2 * n]; r = z[2 * n:]
        g1 = x - r
        g2 = 1.0 - x - r
        g3 = y - r
        g4 = 1.0 - y - r
        if len(pairs):
            dx = x[pairs[:, 0]] - x[pairs[:, 1]]
            dy = y[pairs[:, 0]] - y[pairs[:, 1]]
            dij = np.hypot(dx, dy)
            gp = dij - (r[pairs[:, 0]] + r[pairs[:, 1]])
            return np.concatenate([g1, g2, g3, g4, gp])
        return np.concatenate([g1, g2, g3, g4])

    off1 = 0
    off2 = off1 + n
    off3 = off2 + n
    off4 = off3 + n
    offp = off4 + n
    m = offp + len(pairs)

    def gjac(z):
        x = z[:n]; y = z[n:2 * n]
        rows, cols, data = [], [], []

        # g1: x_i - r_i
        ridx = off1 + np.arange(n)
        rows += ridx.tolist(); cols += np.arange(n).tolist(); data += [1.0] * n
        rows += ridx.tolist(); cols += (2 * n + np.arange(n)).tolist(); data += [-1.0] * n

        # g2: 1 - x_i - r_i
        ridx = off2 + np.arange(n)
        rows += ridx.tolist(); cols += np.arange(n).tolist(); data += [-1.0] * n
        rows += ridx.tolist(); cols += (2 * n + np.arange(n)).tolist(); data += [-1.0] * n

        # g3: y_i - r_i
        ridx = off3 + np.arange(n)
        rows += ridx.tolist(); cols += (n + np.arange(n)).tolist(); data += [1.0] * n
        rows += ridx.tolist(); cols += (2 * n + np.arange(n)).tolist(); data += [-1.0] * n

        # g4: 1 - y_i - r_i
        ridx = off4 + np.arange(n)
        rows += ridx.tolist(); cols += (n + np.arange(n)).tolist(); data += [-1.0] * n
        rows += ridx.tolist(); cols += (2 * n + np.arange(n)).tolist(); data += [-1.0] * n

        # pairwise
        if len(pairs):
            dx = x[pairs[:, 0]] - x[pairs[:, 1]]
            dy = y[pairs[:, 0]] - y[pairs[:, 1]]
            dij = np.hypot(dx, dy)
            invd = 1.0 / np.maximum(dij, 1e-16)
            ridx = offp + np.arange(len(pairs))

            rows += ridx.tolist(); cols += pairs[:, 0].tolist(); data += (dx * invd).tolist()
            rows += ridx.tolist(); cols += pairs[:, 1].tolist(); data += (-dx * invd).tolist()
            rows += ridx.tolist(); cols += (n + pairs[:, 0]).tolist(); data += (dy * invd).tolist()
            rows += ridx.tolist(); cols += (n + pairs[:, 1]).tolist(); data += (-dy * invd).tolist()
            rows += ridx.tolist(); cols += (2 * n + pairs[:, 0]).tolist(); data += [-1.0] * len(pairs)
            rows += ridx.tolist(); cols += (2 * n + pairs[:, 1]).tolist(); data += [-1.0] * len(pairs)

        J = sp.coo_matrix(
            (np.array(data, float), (np.array(rows, int), np.array(cols, int))),
            shape=(m, 3 * n)
        ).tocsr()
        return J

    cons = NonlinearConstraint(gfun, 0.0, np.inf, jac=gjac)

    res = minimize(
        obj, z0, method="trust-constr", jac=obj_jac, bounds=bnds, constraints=[cons],
        options=dict(maxiter=maxiter, verbose=0, xtol=1e-12, gtol=1e-12, barrier_tol=1e-12)
    )

    if not res.success:
        try:
            # Derivative-based fallback inside SciPy
            res = minimize(
                obj, z0, method="SLSQP", jac=obj_jac, bounds=bnds,
                constraints=[{"type": "ineq", "fun": gfun}],
                options=dict(maxiter=maxiter, ftol=1e-9, disp=False)
            )
        except Exception:
            pass

    z = res.x
    x = np.clip(z[:n], 0.0, 1.0)
    y = np.clip(z[n:2 * n], 0.0, 1.0)
    r = np.maximum(z[2 * n:], 0.0)

    # Global tiny safety shrink from worst slack
    bslack = np.min(np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r]))
    pslack = np.inf
    if len(pairs):
        dij = np.hypot(x[pairs[:, 0]] - x[pairs[:, 1]],
                       y[pairs[:, 0]] - y[pairs[:, 1]])
        pslack = np.min(dij - (r[pairs[:, 0]] + r[pairs[:, 1]]))
    eps = max(0.0, -bslack, -0.5 * pslack) + 1e-12
    if eps > 0:
        r = np.maximum(r - eps, 0.0)

    return np.stack([x, y], axis=1), r


# ---------- Seed constructors (explicit, deterministic) ----------

def _seed_grid_5x5_plus():
    """5x5 grid plus two interstitials near opposite corners."""
    g = np.linspace(0.1, 0.9, 5)
    Gx, Gy = np.meshgrid(g, g, indexing='xy')
    P = np.stack([Gx.ravel(), Gy.ravel()], axis=1).tolist()
    P.append([0.2, 0.2]); P.append([0.8, 0.8])
    return np.asarray(P[:26], float)


def _seed_hex_like(n=26, margin=0.08, sx=0.18):
    """Hexagonal lattice inside margins; pick n closest to the square center."""
    sy = (np.sqrt(3) / 2.0) * sx
    pts = []
    y = margin
    row = 0
    while y <= 1 - margin + 1e-12:
        x0 = margin + ((sx / 2.0) if (row % 2 == 1) else 0.0)
        x = x0
        while x <= 1 - margin + 1e-12:
            pts.append([x, y])
            x += sx
        row += 1
        y += sy
    pts = np.asarray(pts, float)
    c = np.array([0.5, 0.5], float)
    idx = np.argsort(np.sum((pts - c) ** 2, axis=1))[:n]
    return pts[idx]


def _seed_stagger_rows(rows, margin=0.075, alpha=0.992, parity=0, w=None):
    """
    Staggered rows seed:
      - rows: list of counts (e.g., [5,4,5,4,4,4])
      - margin: min boundary gap for centers
      - alpha: horizontal relaxation scaling
      - parity: 0 => even rows phase 0, odd rows phase 0.5; 1 flips
      - w: 5-gap weights summing to 5 for 6-row layouts; for R rows use R-1 weights normalized
    """
    R = len(rows)
    # Horizontal spacing based on the widest row
    kmax = max(rows)
    dx = alpha * (1.0 - 2.0 * margin) / max(1, (kmax - 1))
    # Vertical spacing hex-like
    dy_base = (np.sqrt(3) / 2.0) * dx

    gaps = R - 1
    if w is None:
        wv = np.ones(gaps, float)
    else:
        wv = np.asarray(w, float)
        wv = wv / np.sum(wv) * gaps
    # Scale dy so that rows occupy from margin to 1 - margin
    dy = (1.0 - 2.0 * margin) / np.sum(wv)

    y = np.empty(R, float)
    y[0] = margin
    for i in range(gaps):
        y[i + 1] = y[i] + dy * wv[i]

    # Build centers with parity-based half-column shifts
    centers = []
    for i, k in enumerate(rows):
        phase = 0.5 if ((i % 2) == parity) else 0.0
        # Allow 3-count rows to use 1.5*dx to open interior space
        step = dx * (1.5 if k == 3 else 1.0)
        # Center rows horizontally; add phase in units of dx
        width = (k - 1) * step
        x0 = 0.5 * (1.0 - width) + phase * dx
        # Clip
        x0 = min(max(margin, x0), 1.0 - margin - width)
        for j in range(k):
            centers.append([x0 + j * step, y[i]])
    return np.asarray(centers, float)


def _seed_balanced_hex_545444(parity=0, center_bias=False):
    """
    Closed-form stagger seed via constraints:
      4*dx+2*r=1, 5*dy+2*r=1, sqrt((dx/2)^2 + dy^2)=2*r  => r,dx,dy.
    Then place rows [5,4,5,4,4,4] with parity offset.
    """
    r0 = 1.0 / (2.0 + 80.0 / np.sqrt(89.0))
    dx0 = (1.0 - 2.0 * r0) / 4.0
    dy0 = (1.0 - 2.0 * r0) / 5.0
    rows = [5, 4, 5, 4, 4, 4]
    # Vertical weights: either uniform or center-biased
    if center_bias:
        w = np.array([1.0, 1.15, 1.30, 1.15, 1.0], float)
    else:
        w = np.ones(5, float)
    # Normalize weights to sum=5
    w = 5.0 * w / np.sum(w)

    # Build y coordinates from r0 to 1-r0 using dy0*w
    y = [r0]
    for gi in range(5):
        y.append(y[-1] + dy0 * w[gi])
    y = np.array(y[:6], float)

    centers = []
    for i, k in enumerate(rows):
        off = (dx0 / 2.0) if ((i % 2) == parity) else 0.0
        # left-justify so row fits; then center globally
        width = (k - 1) * dx0
        x0 = 0.5 * (1.0 - width) + off
        x0 = min(max(r0, x0), 1.0 - r0 - width)
        for j in range(k):
            centers.append([x0 + j * dx0, y[i]])
    return np.asarray(centers, float)


def _seed_set():
    """Assemble a compact, diverse, deterministic seed set."""
    seeds = []
    # Prior seeds
    seeds.append(_seed_grid_5x5_plus())
    seeds.append(_seed_hex_like(n=26, margin=0.08, sx=0.175))
    seeds.append(_seed_hex_like(n=26, margin=0.10, sx=0.185))
    seeds.append(_seed_hex_like(n=26, margin=0.075, sx=0.168))
    seeds.append(_seed_dense_grid_centered(n=26, nx=6, ny=5, margin=0.06))

    # Balanced hex-row templates
    seeds.append(_seed_balanced_hex_545444(parity=0, center_bias=False))
    seeds.append(_seed_balanced_hex_545444(parity=1, center_bias=True))

    # Staggered custom row patterns
    seeds.append(_seed_stagger_rows([5, 5, 4, 4, 4, 4], margin=0.07, alpha=0.990, parity=0))
    seeds.append(_seed_stagger_rows([4, 4, 4, 4, 3, 3, 4], margin=0.072, alpha=0.988, parity=1))
    return seeds[:8]  # keep runtime tight: use the 8 most representative


def _seed_dense_grid_centered(n=26, nx=6, ny=5, margin=0.07):
    """Rectangular grid; pick n closest to center."""
    xs = np.linspace(margin, 1 - margin, nx)
    ys = np.linspace(margin, 1 - margin, ny)
    Gx, Gy = np.meshgrid(xs, ys, indexing='xy')
    P = np.stack([Gx.ravel(), Gy.ravel()], axis=1)
    c = np.array([0.5, 0.5])
    idx = np.argsort(np.sum((P - c) ** 2, axis=1))[:n]
    return P[idx]


def _optimize_from_seed(centers0):
    """
    Full pipeline for one seed:
      1) trust-constr NLP, analytic Jacob
      2) LP radii polish (fixed centers)
      3) short warm re-optimization (trust-constr) using LP radii as warmstart
      4) final LP polish + safety clip
    """
    try:
        c1, r1 = _nlp_optimize_with_jac(centers0, r_init=None, maxiter=360)
        r_lp1, ok1 = _lp_radii_for_fixed_centers(c1)
        if ok1:
            r1 = r_lp1
        # Warm re-optimization with LP radii warmstart
        c2, r2 = _nlp_optimize_with_jac(c1, r_init=r1, maxiter=220)
        r_lp2, ok2 = _lp_radii_for_fixed_centers(c2)
        if ok2:
            r2 = r_lp2
        c2 = np.clip(c2, 0.0, 1.0)
        r2 = np.maximum(r2 - 1e-12, 0.0)
        return c2, r2, float(np.sum(r2))
    except Exception:
        # SciPy unavailable -> clearance fallback
        r0 = compute_max_radii(centers0)
        return centers0, r0, float(np.sum(r0))


def construct_packing():
    """
    Deterministic multi-start NLP + LP polish constructor.
    Builds several explicit seeds, optimizes each, and returns the best by sum of radii.
    """
    seeds = _seed_set()
    best = None
    best_sum = -np.inf
    for S in seeds:
        c, r, s = _optimize_from_seed(S)
        if s > best_sum:
            best_sum = s
            best = (c, r, s)
    if best is None:
        # Absolute fallback: a basic seed with clearance radii
        c0 = _seed_grid_5x5_plus()
        r0 = compute_max_radii(c0)
        return c0, r0, float(np.sum(r0))
    return best
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    # visualize(centers, radii)