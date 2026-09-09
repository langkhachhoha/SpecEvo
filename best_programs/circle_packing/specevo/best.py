import numpy as np
import time 
import heapq
from scipy.spatial import cKDTree
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    '''
    Construct a packing of 26 circles in the unit square.

    Returns:
        centers: numpy array of shape (26, 2)
        radii: numpy array of shape (26,)
        sum_radii: objective value (sum of radii)
    '''
    rng = np.random.default_rng(1234567)
    n = 26
    max_time = 600.0
    t0 = time.time()

    # Smooth utilities
    def softplus_beta(z, beta):
        bz = beta * z
        return (np.maximum(bz, 0.0) + np.log1p(np.exp(-np.abs(bz)))) / beta

    def sigmoid_beta(z, beta):
        bz = beta * z
        out = np.empty_like(bz)
        pos = bz >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-bz[pos]))
        expbz = np.exp(bz[~pos])
        out[~pos] = expbz / (1.0 + expbz)
        return out

    def smooth_pos(a, beta):
        return softplus_beta(a, beta) - (np.log(2.0) / beta)

    def smooth_pos_grad(a, beta):
        return sigmoid_beta(a, beta)

    # Farthest-point augmentation with tighter margin
    def make_seed_centers(n_target=26, margin=0.12):
        grid = []
        grid_size = 5
        gx = np.linspace(margin, 1.0 - margin, grid_size)
        gy = np.linspace(margin, 1.0 - margin, grid_size)
        for i, xi in enumerate(gx):
            for j, yj in enumerate(gy):
                if i == 2 and j == 2:
                    continue
                grid.append([xi, yj])
        centers = np.array(grid, dtype=float)
        def clearance(pt, C):
            x, y = pt
            dC = np.min(np.linalg.norm(C - pt, axis=1)) if C.size > 0 else np.inf
            dW = min(x, 1.0 - x, y, 1.0 - y)
            return min(dW, 0.4 * dC)
        while centers.shape[0] < n_target:
            cand = rng.uniform(0.05, 0.95, size=(2000, 2))
            scores = np.array([clearance(cand[i], centers) for i in range(cand.shape[0])])
            best_idx = int(np.argmax(scores))
            centers = np.vstack([centers, cand[best_idx]])
        return centers[:n_target].copy()

    # Optimized prioritized overlap push with adaptive step
    def prioritized_push(centers, radii, topk=2 * n):
        x = centers[:, 0]
        y = centers[:, 1]
        r = radii
        tree = cKDTree(centers)
        pairs = tree.query_pairs(r.max() + r.max() + 1e-6)
        heap = []
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = (dx * dx + dy * dy) ** 0.5 + 1e-12
            gap = (r[i] + r[j]) - d
            if gap > 0:
                heapq.heappush(heap, (-gap, i, j))
        steps = min(topk, len(heap))
        for _ in range(steps):
            neg_gap, a, b = heapq.heappop(heap)
            gap = -neg_gap
            dx_ab = x[a] - x[b]
            dy_ab = y[a] - y[b]
            d_ab = (dx_ab * dx_ab + dy_ab * dy_ab) ** 0.5 + 1e-12
            nx = dx_ab / d_ab
            ny = dy_ab / d_ab
            step = 0.3 * gap
            x[a] += step * nx
            y[a] += step * ny
            x[b] -= step * nx
            y[b] -= step * ny
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        centers[:, 0] = x
        centers[:, 1] = y
        return centers

    # Geometric repair with stronger iterative enforcement
    def project_feasible(centers, radii, iters=14, tol=5e-13):
        x = centers[:, 0].copy()
        y = centers[:, 1].copy()
        r = radii.copy()
        for _ in range(iters):
            vL = r - x
            vR = x + r - 1.0
            vB = r - y
            vT = y + r - 1.0
            if np.any(vL > tol):
                x[vL > tol] += vL[vL > tol]
            if np.any(vR > tol):
                x[vR > tol] -= vR[vR > tol]
            if np.any(vB > tol):
                y[vB > tol] += vB[vB > tol]
            if np.any(vT > tol):
                y[vT > tol] -= vT[vT > tol]
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            ctmp = np.stack([x, y], axis=1)
            ctmp = prioritized_push(ctmp, r, topk=8 * len(r))
            x, y = ctmp[:, 0], ctmp[:, 1]
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.hypot(dx, dy) + 1e-12
            v = (r[:, None] + r[None, :]) - d
            if np.max(v[np.triu_indices(len(r), 1)]) <= tol:
                break
            r = np.minimum(r, x)
            r = np.minimum(r, 1.0 - x)
            r = np.minimum(r, y)
            r = np.minimum(r, 1.0 - y)
            r = np.maximum(r, 0.0)
        r *= 0.9995
        return np.stack([x, y], axis=1), r

    # Hybrid radii polish: heap-based contraction + uniform expansion cycles
    def polish_radii_hybrid(centers, radii, passes=80, tol=1e-13):
        x = centers[:, 0]
        y = centers[:, 1]
        nloc = centers.shape[0]
        r = radii.copy()
        b = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
        r = np.minimum(np.maximum(r, 0.0), b)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        D = np.hypot(dx, dy) + 1e-12
        iu, ju = np.triu_indices(nloc, 1)

        for _ in range(passes):
            viol = (r[:, None] + r[None, :]) - D
            vmax = np.max(viol[iu, ju])
            if vmax > tol:
                heap = []
                for i, j in zip(iu, ju):
                    gap = viol[i, j]
                    if gap > tol:
                        heapq.heappush(heap, (-gap, i, j))
                steps = min(40, len(heap))
                for _ in range(steps):
                    neg_gap, i, j = heapq.heappop(heap)
                    gap = -neg_gap
                    cut = 0.5 * gap
                    di = min(cut, r[i])
                    dj = min(cut, r[j])
                    r[i] -= di
                    r[j] -= dj
                r = np.minimum(np.maximum(r, 0.0), b)
                continue

            pair_slack = D - (r[:, None] + r[None, :])
            ps = pair_slack[iu, ju]
            min_pair_slack = np.min(ps) if ps.size > 0 else np.inf
            bound_slack = np.min(b - r)
            delta = min(bound_slack, 0.5 * min_pair_slack)
            if not np.isfinite(delta) or delta <= 1e-13:
                break
            r += 0.96 * delta
            r = np.minimum(r, b)

        for _ in range(3):
            viol = (r[:, None] + r[None, :]) - D
            for i in range(nloc):
                for j in range(i + 1, nloc):
                    s = viol[i, j]
                    if s > 0:
                        cut = 0.4 * s
                        di = min(cut, r[i])
                        dj = min(cut, r[j])
                        r[i] -= di
                        r[j] -= dj
            r = np.clip(r, 0.0, b)
        return r

    # Greedy fixed-centers expansion: allocate residual pair/wall slack
    def greedy_expand_radii(centers, radii, steps=180, safety=0.95):
        x = centers[:, 0]
        y = centers[:, 1]
        r = radii.copy()
        b = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
        r = np.minimum(np.maximum(r, 0.0), b)
        nloc = r.size
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        D = np.hypot(dx, dy) + 1e-12

        for _ in range(steps):
            S = D - (r[:, None] + r[None, :])
            np.fill_diagonal(S, np.inf)
            allowed = np.minimum(b - r, np.min(S, axis=1))
            k = int(np.argmax(allowed))
            amax = allowed[k]
            if not np.isfinite(amax) or amax <= 1e-13:
                break
            delta = safety * amax
            r[k] += delta
            r = np.minimum(r, b)

        S = D - (r[:, None] + r[None, :])
        iu, ju = np.triu_indices(nloc, 1)
        viol = -np.minimum(S[iu, ju], 0.0)
        if viol.size > 0 and np.max(viol) > 0:
            for i, j, gap in zip(iu, ju, viol):
                if gap > 0:
                    cut = 0.51 * gap
                    di = min(cut, r[i])
                    dj = min(cut, r[j])
                    r[i] -= di
                    r[j] -= dj
            r = np.clip(r, 0.0, b)
        return r

    # NEW: Exact radii via LP for fixed centers (donor component)
    iu, ju = np.triu_indices(n, 1)
    m_pairs = iu.size
    A_pairs = np.zeros((m_pairs, n), dtype=float)
    for k, (i, j) in enumerate(zip(iu, ju)):
        A_pairs[k, i] = 1.0
        A_pairs[k, j] = 1.0

    def wall_bounds(centers):
        x = centers[:, 0]
        y = centers[:, 1]
        return np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])

    def pair_distances(centers):
        x = centers[:, 0]
        y = centers[:, 1]
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        d = np.hypot(dx, dy)
        return d

    def solve_radii_lp(centers):
        b = wall_bounds(centers)
        b = np.maximum(b, 0.0)
        d = pair_distances(centers)
        c = -np.ones(n, dtype=float)
        bounds = [(0.0, float(bi)) for bi in b]
        res = linprog(c=c, A_ub=A_pairs, b_ub=d, bounds=bounds, method='highs')
        if res.success and res.x is not None:
            r = res.x.astype(float)
            r = np.clip(r, 0.0, b)
            viol = (r[iu] + r[ju]) - d
            if np.max(viol) > 5e-11:
                over = viol > 0.0
                cuts = 0.55 * viol[over]
                for k, ci in enumerate(np.where(over)[0]):
                    i = int(iu[ci])
                    j = int(ju[ci])
                    r[i] -= min(cuts[k], r[i])
                    r[j] -= min(cuts[k], r[j])
                r = np.clip(r, 0.0, b)
            return r, float(np.sum(r))
        r_fix = np.minimum.reduce([centers[:, 0], 1.0 - centers[:, 0], centers[:, 1], 1.0 - centers[:, 1]])
        s_fix = float(np.sum(r_fix))
        return r_fix, s_fix

    def solve_radii_lp_with_duals(centers):
        b = wall_bounds(centers)
        b = np.maximum(b, 0.0)
        d = pair_distances(centers)
        c = -np.ones(n, dtype=float)
        bounds = [(0.0, float(bi)) for bi in b]
        res = linprog(c=c, A_ub=A_pairs, b_ub=d, bounds=bounds, method='highs')
        duals = None
        if res.success and res.x is not None:
            try:
                duals = np.asarray(res.ineqlin.marginals, dtype=float)
            except Exception:
                duals = None
            r = res.x.astype(float)
            r = np.clip(r, 0.0, b)
            return r, float(np.sum(r)), duals
        r_fix = np.minimum.reduce([centers[:, 0], 1.0 - centers[:, 0], centers[:, 1], 1.0 - centers[:, 1]])
        return r_fix, float(np.sum(r_fix)), duals

    def lp_dual_center_polish(centers, steps=15, base_step=0.004, max_trials=5):
        C = centers.copy().astype(float)
        r0, s_best, duals = solve_radii_lp_with_duals(C)
        best_C = C.copy()
        def pair_weights(C_local, r_local, duals_local):
            dloc = pair_distances(C_local)
            if duals_local is not None and np.all(np.isfinite(duals_local)) and duals_local.size == dloc.size:
                w = np.maximum(duals_local, 0.0)
                sw = np.sum(w)
                return w / sw if sw > 0 else w
            slack = dloc - (r_local[iu] + r_local[ju])
            slack = np.maximum(slack, 0.0)
            w = 1.0 / (1e-6 + slack)
            sw = np.sum(w)
            return w / sw if sw > 0 else w

        for _ in range(steps):
            w = pair_weights(C, r0, duals)
            x = C[:, 0].copy()
            y = C[:, 1].copy()
            dxp = x[iu] - x[ju]
            dyp = y[iu] - y[ju]
            d = np.hypot(dxp, dyp) + 1e-12
            nx = dxp / d
            ny = dyp / d
            gx = np.zeros(n, dtype=float)
            gy = np.zeros(n, dtype=float)
            np.add.at(gx, iu, w * nx)
            np.add.at(gx, ju, -w * nx)
            np.add.at(gy, iu, w * ny)
            np.add.at(gy, ju, -w * ny)
            gscale = max(np.max(np.abs(gx)), np.max(np.abs(gy)), 1e-12)
            step_try = base_step
            accepted = False
            for _t in range(max_trials):
                Xn = np.clip(x + (step_try * gx / gscale), 0.0, 1.0)
                Yn = np.clip(y + (step_try * gy / gscale), 0.0, 1.0)
                Cn = np.stack([Xn, Yn], axis=1)
                rn, sn, duals_n = solve_radii_lp_with_duals(Cn)
                if sn > s_best + 1e-12:
                    C = Cn
                    r0 = rn
                    duals = duals_n
                    s_best = sn
                    best_C = C.copy()
                    accepted = True
                    break
                step_try *= 0.5
                if step_try < 1e-5:
                    break
            if not accepted:
                break
        return best_C

    # Optimization core with adaptive learning rate decay and dynamic beta_r
    def optimize_once(init_centers, iters=3500, beta_pen=40.0, lr_init=0.025, mu0=1.0, mu1=5e2, push_every=5):
        x = init_centers[:, 0].astype(float).copy()
        y = init_centers[:, 1].astype(float).copy()
        x = np.clip(x + rng.normal(0, 5e-4, size=x.shape), 0.0, 1.0)
        y = np.clip(y + rng.normal(0, 5e-4, size=y.shape), 0.0, 1.0)
        target_r0 = 0.024
        s0 = np.log(np.expm1(15.0 * (target_r0 + np.log(2.0) / 15.0))) / 15.0
        s = np.full(x.shape, s0)
        m = np.zeros(3 * n)
        v = np.zeros(3 * n)
        eps_adam = 1e-8
        b1 = 0.9
        b2 = 0.999
        best_cent = None
        best_r = None
        best_sum = -1.0
        grad_norm_history = []
        min_grad_norm = 1e-4
        min_improvement = 1e-6
        patience = 200

        def current_r(ss, beta_r):
            return softplus_beta(ss, beta_r) - (np.log(2.0) / beta_r)

        for t in range(1, iters + 1):
            beta_r = 15.0 + 35.0 * (t / max(1, iters))  # Adaptive β increasing over time
            r = current_r(s, beta_r)
            dr_ds = sigmoid_beta(s, beta_r)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.hypot(dx, dy) + 1e-12

            vL = r - x
            vR = x + r - 1.0
            vB = r - y
            vT = y + r - 1.0
            pL = smooth_pos(vL, beta_pen)
            pR = smooth_pos(vR, beta_pen)
            pB = smooth_pos(vB, beta_pen)
            pT = smooth_pos(vT, beta_pen)
            gL = smooth_pos_grad(vL, beta_pen)
            gR = smooth_pos_grad(vR, beta_pen)
            gB = smooth_pos_grad(vB, beta_pen)
            gT = smooth_pos_grad(vT, beta_pen)

            Vij = (r[:, None] + r[None, :]) - d
            iu2, ju2 = np.triu_indices(n, k=1)
            Vij_u = Vij[iu2, ju2]
            pU = smooth_pos(Vij_u, beta_pen)
            gU = smooth_pos_grad(Vij_u, beta_pen)

            mu = mu0 + (mu1 - mu0) * (t / max(1, iters))
            gx = np.zeros(n)
            gy = np.zeros(n)
            gs = np.zeros(n)

            cL = 2.0 * mu * pL * gL
            cR = 2.0 * mu * pR * gR
            cB = 2.0 * mu * pB * gB
            cT = 2.0 * mu * pT * gT
            gx -= cL
            gx += cR
            gy -= cB
            gy += cT
            gs += (cL + cR + cB + cT) * dr_ds

            if iu2.size > 0:
                cU = 2.0 * mu * pU * gU
                np.add.at(gs, iu2, cU * dr_ds[iu2])
                np.add.at(gs, ju2, cU * dr_ds[ju2])
                ddx = dx[iu2, ju2] / d[iu2, ju2]
                ddy = dy[iu2, ju2] / d[iu2, ju2]
                np.add.at(gx, iu2, -cU * ddx)
                np.add.at(gx, ju2, cU * ddx)
                np.add.at(gy, iu2, -cU * ddy)
                np.add.at(gy, ju2, cU * ddy)

            gs += -dr_ds

            g = np.concatenate([gx, gy, gs])
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            mhat = m / (1.0 - b1 ** t)
            vhat = v / (1.0 - b2 ** t)
            grad_norm = np.linalg.norm(g)
            grad_norm_history.append(grad_norm)
            if len(grad_norm_history) > 10:
                grad_norm_history.pop(0)
            
            current_lr = lr_init * (1.0 if t < 500 else 
                                    0.5 if t < 2000 else 
                                    0.25)
            
            if grad_norm < min_grad_norm and len(grad_norm_history) >= 10:
                grad_avg = np.mean(grad_norm_history[-10:])
                if grad_avg < min_grad_norm * 1.5:
                    if best_sum > -1.0 and (t - 1) % 100 == 0:
                        r_now = current_r(s, beta_r)
                        c_now = np.stack([x, y], axis=1)
                        c_now = prioritized_push(c_now, r_now, topk=5 * n)
                        x, y = c_now[:, 0], c_now[:, 1]
                        r_now = current_r(s, beta_r)
                        c_now = np.stack([x, y], axis=1)
                        c_fix, _ = project_feasible(c_now, r_now, iters=14)
                        try:
                            r_sc, _ = solve_radii_lp(c_fix)
                        except Exception:
                            r_fix_dummy = np.minimum.reduce([c_fix[:, 0], 1.0 - c_fix[:, 0], c_fix[:, 1], 1.0 - c_fix[:, 1]])
                            r_sc = polish_radii_hybrid(c_fix, r_fix_dummy, passes=18, tol=1e-13)
                            r_sc = greedy_expand_radii(c_fix, r_sc, steps=80, safety=0.95)
                        sr = float(np.sum(r_sc))
                        if sr <= best_sum + min_improvement:
                            current_lr *= 0.5
            step = current_lr * mhat / (np.sqrt(vhat) + eps_adam)

            x -= step[0:n]
            y -= step[n:2 * n]
            s -= step[2 * n:3 * n]
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)

            if (t % 5 == 0) or (t % 25 == 0):
                r_now = current_r(s, beta_r)
                c_now = np.stack([x, y], axis=1)
                c_now = prioritized_push(c_now, r_now, topk=5 * n)
                x, y = c_now[:, 0], c_now[:, 1]

            if (t % 100 == 0) or (t == iters):
                r_now = current_r(s, beta_r)
                centers_now = np.stack([x, y], axis=1)
                c_fix, _ = project_feasible(centers_now, r_now, iters=14)
                try:
                    r_sc, _ = solve_radii_lp(c_fix)
                except Exception:
                    r_fix_dummy = np.minimum.reduce([c_fix[:, 0], 1.0 - c_fix[:, 0], c_fix[:, 1], 1.0 - c_fix[:, 1]])
                    r_sc = polish_radii_hybrid(c_fix, r_fix_dummy, passes=18, tol=1e-13)
                    r_sc = greedy_expand_radii(c_fix, r_sc, steps=80, safety=0.95)
                sr = float(np.sum(r_sc))
                if sr > best_sum:
                    best_sum = sr
                    best_cent = c_fix.copy()
                    best_r = r_sc.copy()

        if best_cent is None:
            r_now = current_r(s, beta_r)
            centers_now = np.stack([x, y], axis=1)
            c_fix, _ = project_feasible(centers_now, r_now, iters=14)
            try:
                r_sc, _ = solve_radii_lp(c_fix)
            except Exception:
                r_fix_dummy = np.minimum.reduce([c_fix[:, 0], 1.0 - c_fix[:, 0], c_fix[:, 1], 1.0 - c_fix[:, 1]])
                r_sc = polish_radii_hybrid(c_fix, r_fix_dummy, passes=18, tol=1e-13)
                r_sc = greedy_expand_radii(c_fix, r_sc, steps=80, safety=0.95)
            best_cent = c_fix
            best_r = r_sc
            best_sum = float(np.sum(best_r))
        return best_cent, best_r, best_sum

    seed = make_seed_centers(n)
    best_overall_sum = -1.0
    best_overall_centers = None
    best_overall_radii = None
    restart = 0
    improvement_window = []
    window_size = 10
    min_improvement = 1e-6

    while time.time() - t0 < max_time * 0.97:
        if restart == 0:
            init = seed.copy()
        elif best_overall_centers is not None:
            init = best_overall_centers + rng.normal(0, 2e-3, size=best_overall_centers.shape)
            init = np.clip(init, 0.0, 1.0)
        else:
            init = make_seed_centers(n)

        iters = 3500 if restart < 4 else 2000
        centers, radii, ssum = optimize_once(
            init,
            iters=iters,
            beta_pen=40.0,
            lr_init=0.025 if restart < 4 else 0.02,
            mu0=1.0,
            mu1=5e2,
            push_every=5
        )

        improvement_window.append(ssum)
        if len(improvement_window) > window_size:
            improvement_window.pop(0)

        if len(improvement_window) == window_size:
            avg_improvement = np.mean(np.diff(improvement_window))
            if avg_improvement < min_improvement:
                pass

        if ssum > best_overall_sum:
            best_overall_sum = ssum
            best_overall_centers = centers
            best_overall_radii = radii

        restart += 1
        if restart > 30:
            break

    centers = best_overall_centers
    radii = best_overall_radii
    centers, _ = project_feasible(centers, radii, iters=14)
    centers = lp_dual_center_polish(centers, steps=15, base_step=0.004, max_trials=5)
    try:
        radii, sum_radii = solve_radii_lp(centers)
    except Exception:
        r_fix_dummy = np.minimum.reduce([centers[:, 0], 1.0 - centers[:, 0], centers[:, 1], 1.0 - centers[:, 1]])
        radii = polish_radii_hybrid(centers, r_fix_dummy, passes=80, tol=1e-13)
        radii = greedy_expand_radii(centers, radii, steps=180, safety=0.95)
        sum_radii = float(np.sum(radii))

    centers = centers[:n].astype(float)
    radii = radii[:n].astype(float)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii