# EVOLVE-BLOCK-START
import numpy as np

# Exactly N=21 circles; container is a rectangle with perimeter 4 (W+H=2).
# Deterministic multi-start + alternating LP radii solves and center relaxation,
# with a compact, contact-aware polish and an expanded seed bank.
# Returns array (N,3): (x,y,r) for non-overlapping circles fully contained in the chosen rectangle.
N = 21

def _pairwise_dist(c):
    """Compute pairwise deltas and distances for centers c (n,2).
    Returns (D, dx, dy) with D_ii=inf. D_ij = ||c_i - c_j||; dx,dy signed deltas."""
    dx = c[:, None, 0] - c[None, :, 0]
    dy = c[:, None, 1] - c[None, :, 1]
    D = np.hypot(dx, dy)
    np.fill_diagonal(D, np.inf)
    return D, dx, dy

def _solve_radii_gs(c, W, H, sweeps=720, tol=1e-12, r0=None):
    """Exact LP in radii at fixed centers via monotone Gauss–Seidel fixed point:
       maximize sum r_i s.t. 0<=r_i<=b_i=min(x_i,W-x_i,y_i,H-y_i), r_i+r_j<=d_ij.
       Update: r_i <- max(0, min(b_i, min_j(d_ij - r_j)) - eps).
       Final strict projection ensures feasibility."""
    n = c.shape[0]
    b = np.minimum.reduce([c[:, 0], W - c[:, 0], c[:, 1], H - c[:, 1]])
    D, _, _ = _pairwise_dist(c)
    r = np.clip(b if r0 is None else r0, 0.0, b).copy()
    eps = 1e-12
    order = np.arange(n)
    for s in range(sweeps):
        if s & 1:
            order = order[::-1]
        delta = 0.0
        for i in order:
            mi = float(np.min(D[i] - r))
            m = b[i] if b[i] < mi else mi
            new = 0.0 if m <= 0.0 else (m - eps)
            di = abs(new - r[i])
            if di > delta:
                delta = di
            r[i] = new
        if delta < tol:
            break
    # Strict projection pass
    for i in range(n):
        r[i] = max(0.0, min(b[i], float(np.min(D[i] - r)) - eps))
    return r

def _seed_layout(counts, W, H, rng, phase_scale=0.22, transpose=False, weighted_y=False, jitter_amp=0.0020):
    """Hex-stagger seeding for per-row (or per-column if transpose) 'counts' (sum=N).
       Rows at uniform y or weighted by counts; each row uniform in x with alternating tiny phase.
       Small jitter breaks symmetry; if transpose=True, seed in (H,W) then swap axes."""
    if transpose:
        Wt, Ht = H, W
    else:
        Wt, Ht = W, H
    cj = np.array([max(int(v), 1) for v in counts], dtype=int)
    R = len(cj)
    if weighted_y and R > 1:
        w = 0.60 + 0.40 * (cj / float(np.max(cj)))
        edges = np.concatenate([[0.0], np.cumsum(w)])
        edges /= edges[-1]
        ys = 0.5 * (edges[:-1] + edges[1:]) * Ht
    else:
        ys = (np.arange(R) + 0.5) / max(R, 1) * Ht
    pts = []
    for j, Nj in enumerate(cj):
        Nj = max(int(Nj), 1)
        xs = (np.arange(Nj) + 0.5) / Nj * Wt
        phase = phase_scale * Wt / max(Nj, 1)
        xs = np.clip(xs + (phase if (j & 1) else -phase), 0.0, Wt)
        for x in xs:
            pts.append([x, ys[j]])
    c = np.asarray(pts, dtype=float)
    amp = jitter_amp * min(Wt, Ht)
    jitter = (rng.random(c.shape) - 0.5) * amp
    c[:, 0] = np.clip(c[:, 0] + jitter[:, 0], 0.0, Wt)
    c[:, 1] = np.clip(c[:, 1] + jitter[:, 1], 0.0, Ht)
    if transpose:
        c = c[:, ::-1]
    return c

def _crit_relax(c_init, r, W, H, steps=88, step0=0.026, rng=None,
                topk_hi=10, topk_lo=7, band=0.012, tau0=0.022, cool=0.015,
                step_cool=0.010, noise_period=38, noise_amp=0.0011,
                use_jacobi=True, momentum=True):
    """Critical-constraint guided relaxation of centers at fixed radii r:
       - Build gaps: g_ij = d_ij - (r_i + r_j), boundary gaps {x-r_i, W-x-r_i, y-r_i, H-y-r_i}.
       - Move along an annealed soft-min gradient over the k smallest gaps plus a near-tight band.
       - Optional Jacobi radii refresh aligns gradient with near-current LP contacts.
       - Optional momentum smooths directions. Project to [r_i, W-r_i]×[r_i, H-r_i] each step."""
    if rng is None:
        rng = np.random.default_rng(123456)
    c = c_init.copy()
    n = c.shape[0]
    eps = 1e-12
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    for t in range(steps):
        D, dx, dy = _pairwise_dist(c)
        with np.errstate(divide="ignore", invalid="ignore"):
            ux = np.where(np.isfinite(D), dx / D, 0.0)
            uy = np.where(np.isfinite(D), dy / D, 0.0)
        # Radii refresh (Jacobi blend)
        if use_jacobi:
            b = np.minimum.reduce([c[:, 0], W - c[:, 0], c[:, 1], H - c[:, 1]])
            r_jac = np.empty_like(r)
            for i in range(n):
                mi = float(np.min(D[i] - r))  # proxy wrt current r
                mi = min(mi, b[i])
                r_jac[i] = 0.0 if mi <= 0.0 else (mi - eps)
            alpha = 0.68 / (1.0 + 0.015 * t)
            r_use = alpha * r + (1.0 - alpha) * r_jac
        else:
            r_use = r
        # Pair and boundary gaps (only depend on r_use)
        with np.errstate(invalid="ignore"):
            G = D - (r_use[:, None] + r_use[None, :])
        np.fill_diagonal(G, np.inf)
        band_abs = band * min(W, H)
        tau = tau0 * min(W, H) / (1.0 + cool * t)
        k_now = int(round(topk_hi - (topk_hi - topk_lo) * (t / max(1, steps - 1))))
        k_now = max(min(k_now, n - 1), 3)
        gx = np.zeros(n, dtype=float)
        gy = np.zeros(n, dtype=float)
        for i in range(n):
            # Select k smallest pair gaps + near-tight band
            gi = G[i]
            idx = np.argpartition(gi, k_now - 1)[:k_now]
            gmin = float(np.min(gi[idx]))
            near = np.where(gi < (gmin + band_abs))[0]
            sel = np.unique(np.concatenate([idx, near]))
            sel = sel[sel != i]
            # Boundary gaps for i at r_use
            gl = c[i, 0] - r_use[i]
            gr = (W - r_use[i]) - c[i, 0]
            gb = c[i, 1] - r_use[i]
            gt = (H - r_use[i]) - c[i, 1]
            g_bnd = np.array([gl, gr, gb, gt], dtype=float)
            # Soft-min weights
            g_sel = np.concatenate([gi[sel], g_bnd])
            gm = float(np.min(g_sel))
            wp = np.exp(-(gi[sel] - gm) / max(tau, 1e-6))
            wb = np.exp(-(g_bnd - gm) / max(tau, 1e-6))
            wsum = float(np.sum(wp) + np.sum(wb)) + 1e-12
            wp /= wsum
            wb /= wsum
            # Direction: away from near contacts and from tight boundaries
            gi_x = (np.sum(wp * ux[i, sel]) if len(sel) else 0.0) + wb[0] * (+1.0) + wb[1] * (-1.0)
            gi_y = (np.sum(wp * uy[i, sel]) if len(sel) else 0.0) + wb[2] * (+1.0) + wb[3] * (-1.0)
            gx[i] = gi_x
            gy[i] = gi_y
        # Momentum and step
        if momentum:
            beta = 0.76 - 0.26 * (t / max(1, steps - 1))
            vx = beta * vx + (1.0 - beta) * gx
            vy = beta * vy + (1.0 - beta) * gy
            mx, my = vx, vy
        else:
            mx, my = gx, gy
        eta = step0 / (1.0 + step_cool * t)
        c[:, 0] = np.clip(c[:, 0] + eta * mx, r_use, W - r_use)
        c[:, 1] = np.clip(c[:, 1] + eta * my, r_use, H - r_use)
        # Tiny early noise to break plateaus
        if (t % max(1, noise_period)) == 0 and t < steps // 2:
            amp = noise_amp * (1.0 - t / max(1, steps)) * min(W, H)
            noise = (rng.random(c.shape) - 0.5) * amp
            c[:, 0] = np.clip(c[:, 0] + noise[:, 0], r_use, W - r_use)
            c[:, 1] = np.clip(c[:, 1] + noise[:, 1], r_use, H - r_use)
    return c

def _contact_equalize(c, r, W, H, steps=24, step0=0.010):
    """Very small-step 'tight-contact equalization' at fixed radii r:
       - Build exact pair slacks s_ij = d_ij - (r_i + r_j) and boundary slacks.
       - Push centers away from near-tight constraints using an exponential soft-min.
       - Pure projection onto [r_i,W-r_i]×[r_i,H-r_i]; no noise, tiny steps."""
    n = len(r)
    c = c.copy()
    tau = 0.005 * min(W, H)
    for t in range(steps):
        D, dx, dy = _pairwise_dist(c)
        with np.errstate(divide="ignore", invalid="ignore"):
            ux = np.where(np.isfinite(D), dx / D, 0.0)
            uy = np.where(np.isfinite(D), dy / D, 0.0)
        G = D - (r[:, None] + r[None, :])
        np.fill_diagonal(G, np.inf)
        gx = np.zeros(n)
        gy = np.zeros(n)
        for i in range(n):
            gi = G[i]
            # focus only on near-tight pairs and boundaries
            gmin = float(np.min(gi))
            sel = np.where(gi < gmin + 0.010 * min(W, H))[0]
            wp = np.exp(-(gi[sel] - gmin) / max(tau, 1e-9))
            # boundary slacks
            gl = c[i, 0] - r[i]; gr = (W - r[i]) - c[i, 0]
            gb = c[i, 1] - r[i]; gt = (H - r[i]) - c[i, 1]
            g_bnd = np.array([gl, gr, gb, gt])
            gm = min(float(np.min(g_bnd)), gmin)
            wb = np.exp(-(g_bnd - gm) / max(tau, 1e-9))
            wsum = float(np.sum(wp) + np.sum(wb)) + 1e-12
            wp /= wsum; wb /= wsum
            gi_x = (np.sum(wp * ux[i, sel]) if len(sel) else 0.0) + wb[0] * (+1.0) + wb[1] * (-1.0)
            gi_y = (np.sum(wp * uy[i, sel]) if len(sel) else 0.0) + wb[2] * (+1.0) + wb[3] * (-1.0)
            gx[i] = gi_x; gy[i] = gi_y
        eta = step0 / (1.0 + 0.05 * t)
        c[:, 0] = np.clip(c[:, 0] + eta * gx, r, W - r)
        c[:, 1] = np.clip(c[:, 1] + eta * gy, r, H - r)
    return c

def _contact_polish(c, W, H, iters=3, rng=None):
    """Alternate exact LP radii solves with tiny-step relax and contact-equalize passes to stabilize contacts."""
    if rng is None:
        rng = np.random.default_rng(999)
    r = None
    for _ in range(iters):
        r = _solve_radii_gs(c, W, H, sweeps=760, tol=1e-12, r0=r)
        c = _crit_relax(c, r, W, H, steps=28, step0=0.012, rng=rng,
                        topk_hi=10, topk_lo=8, band=0.010, tau0=0.020, cool=0.015,
                        step_cool=0.010, noise_period=42, noise_amp=0.0011,
                        use_jacobi=False, momentum=False)
        c = _contact_equalize(c, r, W, H, steps=20, step0=0.010)
    r = _solve_radii_gs(c, W, H, sweeps=820, tol=1e-12, r0=r)
    return c, r

def _alt_opt(c0, W, H, rng, rounds=8, base_sweeps=560, step0=0.026):
    """Alternating optimization from a seed:
       Repeat: solve radii (warm-started GS) then critical relax (annealed soft-min ascent).
       Keep best-by-sum(r) across rounds; finalize via compact contact polish."""
    c = c0.copy()
    best_c, best_r, best_s = None, None, -1.0
    r = None
    for k in range(rounds):
        r = _solve_radii_gs(c, W, H, sweeps=base_sweeps + 24 * k, tol=1e-12, r0=r)
        s = float(np.sum(r))
        if s > best_s:
            best_c, best_r, best_s = c.copy(), r.copy(), s
        steps = 64 if k < rounds - 1 else 52
        c = _crit_relax(c, r, W, H, steps=steps,
                        step0=max(0.016, step0 - 0.003 * min(k, 7)),
                        rng=rng, topk_hi=10, topk_lo=7, band=0.012, tau0=0.022, cool=0.015,
                        step_cool=0.010, noise_period=38, noise_amp=0.0011,
                        use_jacobi=True, momentum=True)
    c_fin, r_fin = _contact_polish(best_c, W, H, iters=3, rng=rng)
    s_fin = float(np.sum(r_fin))
    if s_fin > best_s:
        best_c, best_r, best_s = c_fin, r_fin, s_fin
    else:
        best_r = _solve_radii_gs(best_c, W, H, sweeps=820, tol=1e-12, r0=best_r)
        best_s = float(np.sum(best_r))
    return best_c, best_r, best_s

def _score_seed(c0, W, H, rng):
    """Cheap deterministic scoring for a seed: two short alt cycles + final LP pass; return (sum_r, c, r)."""
    c = c0.copy()
    r = None
    for k in range(2):
        r = _solve_radii_gs(c, W, H, sweeps=380 + 20 * k, tol=1e-12, r0=r)
        c = _crit_relax(c, r, W, H, steps=42, step0=0.022, rng=rng,
                        topk_hi=9, topk_lo=7, band=0.013, tau0=0.020, cool=0.015,
                        step_cool=0.010, noise_period=38, noise_amp=0.0011,
                        use_jacobi=True, momentum=True)
    r = _solve_radii_gs(c, W, H, sweeps=520, tol=1e-12, r0=r)
    return float(np.sum(r)), c, r

def _retarget_aspect(c, W, H, Wp, rng):
    """Affine-retarget centers from (W,H) to (Wp,Hp=2-Wp); brief relax+solve polish; return (c,r,sum)."""
    Hp = 2.0 - Wp
    if Wp <= 0.0 or Hp <= 0.0:
        return c, np.zeros(len(c)), -1.0
    sx, sy = (Wp / W), (Hp / H)
    cp = c.copy()
    cp[:, 0] = np.clip(cp[:, 0] * sx, 0.0, Wp)
    cp[:, 1] = np.clip(cp[:, 1] * sy, 0.0, Hp)
    r = _solve_radii_gs(cp, Wp, Hp, sweeps=520, tol=1e-12)
    cp = _crit_relax(cp, r, Wp, Hp, steps=26, step0=0.016, rng=rng,
                     topk_hi=10, topk_lo=8, band=0.012, tau0=0.022, cool=0.015,
                     step_cool=0.010, noise_period=42, noise_amp=0.0011,
                     use_jacobi=True, momentum=True)
    r = _solve_radii_gs(cp, Wp, Hp, sweeps=620, tol=1e-12, r0=r)
    return cp, r, float(np.sum(r))

def _aspect_line_search(c, r, W, H, rng, span=0.22, iters=6):
    """Golden-section search over width in [W-span,W+span]∩[0.64,1.36], using _retarget_aspect."""
    lo = max(0.64, W - span)
    hi = min(1.36, W + span)
    if hi - lo < 1e-6:
        return c, r, float(np.sum(r)), (W, H)
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    best_c, best_r, best_s, best_WH = c, r, float(np.sum(r)), (W, H)
    c1, r1, s1 = _retarget_aspect(c, W, H, x1, rng)
    c2, r2, s2 = _retarget_aspect(c, W, H, x2, rng)
    for _ in range(max(2, iters)):
        if s1 > s2:
            if s1 > best_s:
                best_c, best_r, best_s, best_WH = c1, r1, s1, (x1, 2.0 - x1)
            hi, c2, r2, s2 = x2, c1, r1, s1
            x2 = x1
            x1 = hi - phi * (hi - lo)
            c1, r1, s1 = _retarget_aspect(c, W, H, x1, rng)
        else:
            if s2 > best_s:
                best_c, best_r, best_s, best_WH = c2, r2, s2, (x2, 2.0 - x2)
            lo, c1, r1, s1 = x1, c2, r2, s2
            x1 = x2
            x2 = lo + phi * (hi - lo)
            c2, r2, s2 = _retarget_aspect(c, W, H, x2, rng)
    return best_c, best_r, best_s, best_WH

def circle_packing21() -> np.ndarray:
    """Deterministic constructor maximizing sum of 21 radii within a rectangle (W+H=2).
       Approach:
         - Expanded multi-start over aspects and curated row/column patterns with several phases.
         - Quick alt-cycles to rank seeds; intensive alternating optimization on finalists.
         - Golden-section micro-search over aspect; compact contact-aware polish.
       Returns (x,y,r) in the chosen rectangle; circles are non-overlapping and contained."""
    rng = np.random.default_rng(20250111)

    # Aspect candidates (H = 2 - W). Dense near square + a few wider/taller extremes.
    widths = sorted(set([
        0.70, 0.74, 0.78, 0.82, 0.86, 0.90, 0.94, 0.98, 1.02, 1.06, 1.10, 1.14, 1.18, 1.22, 1.26, 1.30
    ]))

    # Curated row-count patterns summing to 21. Broadened to include skewed 3-row and balanced 4/5/6-row layouts.
    bank = [
        # 3-row (wide)
        [11, 5, 5], [10, 6, 5], [9, 7, 5], [9, 6, 6], [8, 8, 5], [8, 7, 6], [7, 7, 7],
        # 4-row (moderate)
        [8, 5, 4, 4], [7, 6, 4, 4], [7, 5, 5, 4], [6, 6, 5, 4], [6, 5, 5, 5], [5, 5, 5, 6],
        # 5-row (near-square)
        [6, 4, 4, 4, 3], [5, 5, 4, 4, 3], [5, 4, 4, 4, 4], [4, 5, 4, 4, 4], [4, 4, 5, 4, 4], [4, 4, 4, 5, 4],
        # 6-row and tall
        [4, 4, 4, 3, 3, 3], [3, 4, 3, 4, 3, 4], [4, 3, 4, 3, 4, 3],
        # 7-row (very tall)
        [3, 3, 3, 3, 3, 3, 3],
    ]
    phases = [0.00, 0.15, 0.22, 0.30]
    orientations = [False, True]

    coarse_pool = []
    for W in widths:
        H = 2.0 - W
        if H <= 0.0 or W <= 0.0:
            continue
        # Aspect-aware subset to focus computation
        if W > H + 0.14:  # wide: emphasize 3/4-row
            sel = [[11, 5, 5], [10, 6, 5], [9, 6, 6], [8, 8, 5], [8, 7, 6], [7, 7, 7],
                   [8, 5, 4, 4], [7, 6, 4, 4], [6, 6, 5, 4], [6, 5, 5, 5]]
        elif H > W + 0.14:  # tall: emphasize 5/6/7-row
            sel = [[4, 5, 4, 4, 4], [5, 4, 4, 4, 4], [5, 5, 4, 4, 3],
                   [4, 4, 4, 3, 3, 3], [3, 4, 3, 4, 3, 4], [3, 3, 3, 3, 3, 3, 3]]
        else:  # near square: balanced 4/5-row + a couple 3-row
            sel = [[5, 4, 4, 4, 4], [4, 5, 4, 4, 4], [4, 4, 5, 4, 4], [4, 4, 4, 5, 4],
                   [6, 4, 4, 4, 3], [5, 5, 4, 4, 3], [6, 6, 5, 4], [7, 5, 5, 4], [8, 7, 6], [7, 7, 7]]
        for counts in sel:
            if sum(counts) != N:
                continue
            for phase_scale in phases:
                for transpose in orientations:
                    for rev in (False, True):  # reverse rows/cols to alter y-spacing under weighting
                        cc = counts[::-1] if rev else counts
                        weighted_y = (not transpose)
                        c0 = _seed_layout(cc, W, H, rng, phase_scale=phase_scale,
                                          transpose=transpose, weighted_y=weighted_y)
                        s, c, r = _score_seed(c0, W, H, rng)
                        coarse_pool.append((s, c, r, cc, transpose, W, H, phase_scale))

    if not coarse_pool:
        # Fallback: simple 7x3 grid with conservative radii in a square
        W, H = 1.0, 1.0
        xs = (np.arange(7) + 0.5) / 7.0 * W
        ys = (np.arange(3) + 0.5) / 3.0 * H
        grid = np.array([[x, y] for y in ys for x in xs], float)
        r = np.full(N, 0.07 * min(W, H))
        out = np.zeros((N, 3), float)
        out[:, 0:2] = grid
        out[:, 2] = r
        return out

    # Keep the best coarse candidates for intensive refinement
    coarse_pool.sort(key=lambda t: t[0], reverse=True)
    finalists = coarse_pool[:min(10, len(coarse_pool))]

    # Intensive refinement on finalists
    best_s = -1.0
    best_c = None
    best_r = None
    best_WH = (1.0, 1.0)
    for _, _, _, counts, transpose, W, H, phase_scale in finalists:
        c0 = _seed_layout(counts, W, H, rng, phase_scale=phase_scale,
                          transpose=transpose, weighted_y=(not transpose))
        c, r, s = _alt_opt(c0, W, H, rng, rounds=8, base_sweeps=560, step0=0.026)
        if s > best_s:
            best_s, best_c, best_r, best_WH = s, c, r, (W, H)

    # Aspect golden-section micro-search; brief local re-optimization if improved
    W, H = best_WH
    c_as, r_as, s_as, WH_as = _aspect_line_search(best_c, best_r, W, H, rng, span=0.22, iters=6)
    if s_as > best_s:
        best_c, best_r, best_s, best_WH = c_as, r_as, s_as, WH_as
        W, H = best_WH
        c_ref, r_ref, s_ref = _alt_opt(best_c, W, H, rng, rounds=3, base_sweeps=540, step0=0.024)
        if s_ref > best_s:
            best_c, best_r, best_s = c_ref, r_ref, s_ref

    # Deterministic tiny jitter probes to escape shallow locks
    probe_amp = 0.0010 * min(W, H)
    for _ in range(2):
        noise = (rng.random(best_c.shape) - 0.5) * probe_amp
        cp = np.empty_like(best_c)
        cp[:, 0] = np.clip(best_c[:, 0] + noise[:, 0], 0.0, W)
        cp[:, 1] = np.clip(best_c[:, 1] + noise[:, 1], 0.0, H)
        c_try, r_try, s_try = _alt_opt(cp, W, H, rng, rounds=2, base_sweeps=520, step0=0.024)
        if s_try > best_s:
            best_c, best_r, best_s = c_try, r_try, s_try

    # Final micro-polish and exact LP solve
    c_mp, r_mp = _contact_polish(best_c, W, H, iters=3, rng=rng)
    if float(np.sum(r_mp)) > best_s:
        best_c, best_r = c_mp, r_mp

    out = np.zeros((N, 3), dtype=float)
    out[:, 0:2] = best_c
    out[:, 2] = np.maximum(0.0, best_r)
    return out

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    arr = circle_packing21()
    print(f"Radii sum: {np.sum(arr[:, -1]):.10f}")