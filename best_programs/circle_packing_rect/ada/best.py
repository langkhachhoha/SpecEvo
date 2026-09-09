# EVOLVE-BLOCK-START
import numpy as np

def circle_packing21() -> np.ndarray:
    """
    Deterministic NLP (trust-constr) maximizing the sum of 21 circle radii inside a rectangle with W+H<=2.
    Decision vars: z = [x1..x21, y1..y21, r1..r21, W, H].
    Objective: minimize -sum(r).
    Constraints:
      - Non-overlap: ||(xi-xj, yi-yj)|| - (ri + rj) >= 0 for all i<j (NonlinearConstraint with analytic Jacobian).
      - In-rectangle (linear): x_i - r_i >= 0, y_i - r_i >= 0,
                               W - x_i - r_i >= 0, H - y_i - r_i >= 0 for all i (LinearConstraint).
      - Perimeter (linear): W + H <= 2.
    Initialization: closed-form equal-hex 6-row layout then tiny deterministic Halton jitter and 2% deflate.
    Post-process: uniform shrink to strict feasibility (floating-point safety). Robust fallback to equal-hex.
    Returns: np.ndarray shape (21,3) rows (x, y, r).
    """
    n = 21
    try:
        from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
    except Exception:
        return _equal_hex()

    # Halton sequence for deterministic jitter
    def halton(k, b):
        f, r = 1.0, 0.0
        while k > 0:
            f /= b
            r += f * (k % b)
            k //= b
        return r

    # Equal-hex initializer (feasible, W+H=2)
    def init_equal_hex():
        rt3 = np.sqrt(3.0)
        cw, cv = 8.0, 2.0 + 5.0 * rt3
        r = 2.0 / (cw + cv)
        W, H = cw * r, cv * r
        dy = rt3 * r
        counts = [4, 3, 4, 3, 4, 3]
        offs = [0.0, r, 0.0, r, 0.0, r]
        xs, ys, rs = [], [], []
        for i, (c, off) in enumerate(zip(counts, offs)):
            y = r + i * dy
            for j in range(c):
                x = r + off + 2.0 * r * j
                xs.append(x); ys.append(y); rs.append(r)
        x = np.array(xs); y = np.array(ys); r = np.array(rs)
        # Small deterministic jitter to unstick equalities
        amp = 1e-3 * min(W, H)
        for i in range(n):
            x[i] = np.clip(x[i] + (halton(i + 1, 2) - 0.5) * amp, 0.0, W)
            y[i] = np.clip(y[i] + (halton(i + 1, 3) - 0.5) * amp, 0.0, H)
        r *= 0.98  # introduce slack
        return x, y, r, W, H

    # Build pair index arrays once
    IJ = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)
    I = IJ[:, 0]; J = IJ[:, 1]
    m_vars = 3 * n + 2

    # Objective and gradient
    def fun(z):
        return -np.sum(z[2 * n:3 * n])

    def grad(z):
        g = np.zeros_like(z)
        g[2 * n:3 * n] = -1.0
        return g

    # Non-overlap constraints
    def g_pairs(z):
        x = z[:n]; y = z[n:2 * n]; r = z[2 * n:3 * n]
        dx = x[I] - x[J]; dy = y[I] - y[J]
        dij = np.hypot(dx, dy)
        return dij - (r[I] + r[J])

    def jac_pairs(z):
        x = z[:n]; y = z[n:2 * n]; r = z[2 * n:3 * n]
        dx = x[I] - x[J]; dy = y[I] - y[J]
        dij = np.hypot(dx, dy)
        dij = np.maximum(dij, 1e-12)
        ux = dx / dij; uy = dy / dij
        Jmat = np.zeros((len(I), m_vars), dtype=float)
        # d/dx
        Jmat[np.arange(len(I)), I] += ux
        Jmat[np.arange(len(I)), J] -= ux
        # d/dy
        Jmat[np.arange(len(I)), n + I] += uy
        Jmat[np.arange(len(I)), n + J] -= uy
        # d/dr
        Jmat[np.arange(len(I)), 2 * n + I] -= 1.0
        Jmat[np.arange(len(I)), 2 * n + J] -= 1.0
        return Jmat

    # Linear in-rectangle constraints as A*z >= 0
    def linear_inrect():
        A = np.zeros((4 * n, m_vars), dtype=float)
        # x - r >= 0
        for i in range(n):
            A[i, i] = 1.0
            A[i, 2 * n + i] = -1.0
        # y - r >= 0
        for i in range(n):
            A[n + i, n + i] = 1.0
            A[n + i, 2 * n + i] = -1.0
        # W - x - r >= 0
        for i in range(n):
            A[2 * n + i, 3 * n] = 1.0
            A[2 * n + i, i] = -1.0
            A[2 * n + i, 2 * n + i] = -1.0
        # H - y - r >= 0
        for i in range(n):
            A[3 * n + i, 3 * n + 1] = 1.0
            A[3 * n + i, n + i] = -1.0
            A[3 * n + i, 2 * n + i] = -1.0
        lb = np.zeros(4 * n)
        ub = np.full(4 * n, np.inf)
        return LinearConstraint(A, lb, ub)

    # Perimeter W + H <= 2
    def linear_perim():
        A = np.zeros((1, m_vars), dtype=float)
        A[0, 3 * n] = 1.0
        A[0, 3 * n + 1] = 1.0
        lb = -np.inf
        ub = 2.0
        return LinearConstraint(A, lb, ub)

    # Final uniform shrink to strict feasibility inside found W,H
    def project_uniform_shrink(x, y, r, W, H):
        # walls
        with np.errstate(divide="ignore", invalid="ignore"):
            wall = np.minimum.reduce([x / np.maximum(r, 1e-18),
                                      (W - x) / np.maximum(r, 1e-18),
                                      y / np.maximum(r, 1e-18),
                                      (H - y) / np.maximum(r, 1e-18)])
        wall = np.minimum(wall, np.inf)
        # pairs
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        D = np.hypot(dx, dy)
        np.fill_diagonal(D, np.inf)
        rij = r[:, None] + r[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            pair = np.min(D / np.maximum(rij, 1e-18), axis=1)
        gamma = float(np.minimum(1.0, np.minimum(np.min(wall), np.min(pair))))
        if not np.isfinite(gamma) or gamma <= 0.0:
            gamma = 0.0
        return np.maximum(1e-9, 0.999 * gamma) * r

    # Build problem
    x0, y0, r0, W0, H0 = init_equal_hex()
    z0 = np.concatenate([x0, y0, r0, [W0, H0]])

    # Bounds
    rmin = 1e-8
    lb = np.concatenate([np.zeros(n), np.zeros(n), np.full(n, rmin), np.full(2, rmin)])
    ub = np.full(m_vars, 2.0)
    bnds = Bounds(lb, ub)

    # Constraints
    c_pairs = NonlinearConstraint(g_pairs, 0.0, np.inf, jac=jac_pairs)
    c_in = linear_inrect()
    c_per = linear_perim()

    # Solve
    try:
        res = minimize(fun, z0, method='trust-constr', jac=grad, hess=None,
                       constraints=[c_pairs, c_in, c_per], bounds=bnds,
                       options=dict(maxiter=1000, gtol=1e-9, xtol=1e-12, verbose=0))
        z = res.x
        x, y, r, W, H = z[:n], z[n:2 * n], z[2 * n:3 * n], z[3 * n], z[3 * n + 1]
        # Safety repair
        r = project_uniform_shrink(x, y, np.maximum(r, rmin), max(W, rmin), max(H, rmin))
        # Clip to box just in case of tiny drift
        x = np.clip(x, r, max(W, rmin) - r)
        y = np.clip(y, r, max(H, rmin) - r)
        return np.column_stack([x, y, r])
    except Exception:
        return _equal_hex()

def _equal_hex():
    # Closed-form equal hexagonal 6-row packing (deterministic, W+H=2)
    rt3 = np.sqrt(3.0)
    cw, cv = 8.0, 2.0 + 5.0 * rt3
    r = 2.0 / (cw + cv)
    W, H = cw * r, cv * r
    dy = rt3 * r
    counts = [4, 3, 4, 3, 4, 3]
    offs = [0.0, r, 0.0, r, 0.0, r]
    C = []
    for i, (c, off) in enumerate(zip(counts, offs)):
        y = r + i * dy
        for j in range(c):
            x = r + off + 2.0 * r * j
            C.append([x, y, r])
    return np.array(C, dtype=float)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    cs = circle_packing21()
    print(f"sum_radii={np.sum(cs[:,2]):.12f}, n={cs.shape[0]}")