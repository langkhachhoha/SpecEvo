import numpy as np
import time

def circle_packing21() -> np.ndarray:
    '''
    Places 21 non-overlapping circles inside a rectangle of
    perimeter 4, maximizing the sum of their radii.

    Returns:
        circles: numpy array of shape (21, 3), where each row
            stores (x, y, radius).
    '''
    rng = np.random.default_rng(42)
    n = 21
    time_limit = 585.0
    t0 = time.time()

    # ---- Initialization: 7x3 hexagonal lattice with slightly expanded spacing ----
    xmin = 0.0
    ymin = 0.0
    W0 = 1.08
    H0 = 0.88  # W0 + H0 = 1.96 < 2
    xmax = xmin + W0
    ymax = ymin + H0

    cols, rows = 7, 3
    mx = 0.10
    my = 0.10
    xs = np.linspace(xmin + mx, xmax - mx, cols)
    ys = np.linspace(ymin + my, ymax - my, rows)

    x_list, y_list = [], []
    dx_col = (xmax - mx - (xmin + mx)) / (cols - 1) if cols > 1 else 0.0
    for r_idx, yy in enumerate(ys):
        offset = 0.5 * dx_col if (r_idx % 2 == 1 and cols > 1) else 0.0
        row_xs = xs + offset
        row_xs = np.clip(row_xs, xmin + mx, xmax - mx)
        for xx in row_xs:
            x_list.append(xx)
            y_list.append(yy)
    x = np.array(x_list[:n], dtype=float)
    y = np.array(y_list[:n], dtype=float)
    x += rng.uniform(-0.004, 0.004, size=n)
    y += rng.uniform(-0.004, 0.004, size=n)
    r = np.full(n, 0.022, dtype=float)

    I, J = np.triu_indices(n, k=1)

    def compute_radii(x, y, r, xmin, xmax, ymin, ymax):
        for i in range(n):
            dL = x[i] - xmin
            dR = xmax - x[i]
            dB = y[i] - ymin
            dT = ymax - y[i]
            d_border = min(dL, dR, dB, dT)
            d_circ = np.inf
            for j in range(n):
                if i == j:
                    continue
                d_circ = min(d_circ, np.hypot(x[i] - x[j], y[i] - y[j]))
            r[i] = max(1e-6, min(d_border, d_circ / 2.05))
        return r

    r = compute_radii(x, y, r, xmin, xmax, ymin, ymax)

    def pack(x, y, r, xmin, xmax, ymin, ymax):
        return np.concatenate([x, y, r, np.array([xmin, xmax, ymin, ymax], dtype=float)])

    def unpack(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        xmin, xmax, ymin, ymax = v[3*n:3*n+4]
        return x, y, r, xmin, xmax, ymin, ymax

    def barrier_value_grad(x, y, r, xmin, xmax, ymin, ymax, t_weight, return_slacks=False):
        dxp = x[I] - x[J]
        dyp = y[I] - y[J]
        d = np.hypot(dxp, dyp)
        s_sep = d - r[I] - r[J]

        sL = x - r - xmin
        sR = xmax - x - r
        sB = y - r - ymin
        sT = ymax - y - r
        s_pos = r
        sp = 2.0 - ((xmax - xmin) + (ymax - ymin))

        min_slack = min(
            np.min(s_sep) if s_sep.size else np.inf,
            np.min(sL), np.min(sR), np.min(sB), np.min(sT),
            np.min(s_pos), sp
        )
        if not np.isfinite(min_slack) or min_slack <= 0.0:
            return -np.inf, None, min_slack, None

        val = t_weight * np.sum(r)
        val += np.sum(np.log(s_sep))
        val += np.sum(np.log(sL)) + np.sum(np.log(sR)) + np.sum(np.log(sB)) + np.sum(np.log(sT))
        val += np.sum(np.log(s_pos))
        val += np.log(sp)

        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        gb = np.zeros(4)

        inv_s = 1.0 / s_sep
        inv_d = 1.0 / (d + 1e-12)
        u = inv_s * inv_d

        np.add.at(gx, I, u * dxp)
        np.add.at(gy, I, u * dyp)
        np.add.at(gx, J, -u * dxp)
        np.add.at(gy, J, -u * dyp)
        np.add.at(gr, I, -inv_s)
        np.add.at(gr, J, -inv_s)

        inv_sL = 1.0 / sL
        inv_sR = 1.0 / sR
        inv_sB = 1.0 / sB
        inv_sT = 1.0 / sT

        gx += inv_sL
        gr -= inv_sL
        gb[0] -= np.sum(inv_sL)

        gx -= inv_sR
        gr -= inv_sR
        gb[1] += np.sum(inv_sR)

        gy += inv_sB
        gr -= inv_sB
        gb[2] -= np.sum(inv_sB)

        gy -= inv_sT
        gr -= inv_sT
        gb[3] += np.sum(inv_sT)

        inv_pos = 1.0 / s_pos
        gr += inv_pos

        inv_sp = 1.0 / sp
        gb[0] += inv_sp
        gb[1] -= inv_sp
        gb[2] += inv_sp
        gb[3] -= inv_sp

        gr += t_weight

        g = np.concatenate([gx, gy, gr, gb])

        slacks = None
        if return_slacks:
            slacks = {
                'I': I, 'J': J,
                's_sep': s_sep,
                'sL': sL, 'sR': sR, 'sB': sB, 'sT': sT,
                'sp': sp
            }
        return val, g, min_slack, slacks

    def clearance_scales(slacks):
        I = slacks['I']; J = slacks['J']
        s_sep = slacks['s_sep']
        sL, sR, sB, sT = slacks['sL'], slacks['sR'], slacks['sB'], slacks['sT']
        # Replace full matrix with vectorized reduction using np.minimum.reduceat
        s_pair_min = np.full(n, np.inf)
        # Aggregate minimum over all pairs using reduceat
        np.minimum.at(s_pair_min, I, s_sep)
        np.minimum.at(s_pair_min, J, s_sep)
        # Now combine with border constraints
        c = np.minimum(s_pair_min, sL)
        c = np.minimum(c, sR)
        c = np.minimum(c, sB)
        c = np.minimum(c, sT)
        eps = 1e-6
        sxy = 1.0 + 0.30 / (eps + c)
        sr = 1.0 + 0.20 / (eps + c)
        sxy = np.clip(sxy, 1.0, 3.5)
        sr = np.clip(sr, 1.0, 2.5)
        return sxy, sr

    def pack(x, y, r, xmin, xmax, ymin, ymax):
        return np.concatenate([x, y, r, np.array([xmin, xmax, ymin, ymax], dtype=float)])

    def unpack(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        xmin, xmax, ymin, ymax = v[3*n:3*n+4]
        return x, y, r, xmin, xmax, ymin, ymax

    v = pack(x, y, r, xmin, xmax, ymin, ymax)
    beta = 0.96
    m = np.zeros_like(v)
    step_base = 0.045
    t_weight = 1.0
    t_mult = 2.7
    max_stages = 13
    steps_per_stage = 550
    min_slack_tol = 1e-10

    best_v = v.copy()
    best_score = -np.inf

    for stage in range(max_stages):
        if time.time() - t0 > time_limit:
            break
        m[:] = 0.0
        alpha = step_base

        bv = barrier_value_grad(*unpack(v), t_weight, return_slacks=True)
        val, g, ms, slacks = bv
        if not np.isfinite(val) or ms <= 0:
            break

        for k in range(steps_per_stage):
            if time.time() - t0 > time_limit:
                break

            sxy, sr = clearance_scales(slacks)
            gx = g[0:n] * sxy
            gy = g[n:2*n] * sxy
            grv = g[2*n:3*n] * sr
            gb = g[3*n:3*n+4]
            g_pre = np.concatenate([gx, gy, grv, gb])

            m = beta * m + g_pre
            delta = alpha * m

            if k % 160 == 0:
                alpha *= 0.91
                if alpha < 1e-6:
                    break

            tau = 1.0
            accepted = False
            while tau > 1e-8:
                v_new = v + tau * delta
                xN, yN, rN, xminN, xmaxN, yminN, ymaxN = unpack(v_new)
                if not (xminN < xmaxN and yminN < ymaxN):
                    tau *= 0.5
                    continue
                val_new, g_new, ms_new, sl_new = barrier_value_grad(xN, yN, rN, xminN, xmaxN, yminN, ymaxN, t_weight, return_slacks=True)
                if np.isfinite(val_new) and ms_new > min_slack_tol and val_new >= val - 1e-10:
                    v = v_new
                    val = val_new
                    g = g_new
                    slacks = sl_new
                    accepted = True
                    break
                tau *= 0.5

            if not accepted:
                alpha *= 0.53
                if alpha < 3e-6:
                    break
                continue

            xC, yC, rC, xminC, xmaxC, yminC, ymaxC = unpack(v)
            score = float(np.sum(rC))
            if score > best_score:
                best_score = score
                best_v = v.copy()

        t_weight *= t_mult
        step_base = min(0.14, step_base * 1.07)

    x, y, r, xmin, xmax, ymin, ymax = unpack(best_v)

    left = np.min(x - r)
    right = np.max(x + r)
    bottom = np.min(y - r)
    top = np.max(y + r)

    width = right - left
    height = top - bottom
    if np.isfinite(width + height) and (width + height) > 0:
        scale = (2.0 - 1e-10) / (width + height)
        cx = 0.5 * (left + right)
        cy = 0.5 * (bottom + top)
        x = (x - cx) * scale + cx
        y = (y - cy) * scale + cy
        r = r * scale

    left = np.min(x - r)
    right = np.max(x + r)
    bottom = np.min(y - r)
    top = np.max(y + r)

    def barrier_fixedbox_value_grad(x, y, r, left, right, bottom, top, t_weight_fb):
        dxp = x[I] - x[J]
        dyp = y[I] - y[J]
        d = np.hypot(dxp, dyp)
        s_sep = d - r[I] - r[J]

        sL = x - r - left
        sR = right - x - r
        sB = y - r - bottom
        sT = top - y - r
        s_pos = r

        min_slack = min(
            np.min(s_sep) if s_sep.size else np.inf,
            np.min(sL), np.min(sR), np.min(sB), np.min(sT),
            np.min(s_pos)
        )
        if not np.isfinite(min_slack) or min_slack <= 0.0:
            return -np.inf, None, min_slack, None

        val = t_weight_fb * np.sum(r)
        val += np.sum(np.log(s_sep))
        val += np.sum(np.log(sL)) + np.sum(np.log(sR)) + np.sum(np.log(sB)) + np.sum(np.log(sT))
        val += np.sum(np.log(s_pos))

        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)

        inv_s = 1.0 / s_sep
        inv_d = 1.0 / (d + 1e-12)
        u = inv_s * inv_d

        np.add.at(gx, I, u * dxp)
        np.add.at(gy, I, u * dyp)
        np.add.at(gx, J, -u * dxp)
        np.add.at(gy, J, -u * dyp)
        np.add.at(gr, I, -inv_s)
        np.add.at(gr, J, -inv_s)

        inv_sL = 1.0 / sL
        inv_sR = 1.0 / sR
        inv_sB = 1.0 / sB
        inv_sT = 1.0 / sT

        gx += inv_sL
        gr -= inv_sL

        gx -= inv_sR
        gr -= inv_sR

        gy += inv_sB
        gr -= inv_sB

        gy -= inv_sT
        gr -= inv_sT

        inv_pos = 1.0 / s_pos
        gr += inv_pos

        slacks = {
            's_sep': s_sep,
            'sL': sL, 'sR': sR, 'sB': sB, 'sT': sT
        }
        g = np.concatenate([gx, gy, gr])
        return val, g, min_slack, slacks

    def clearance_scales_fixedbox(slacks):
        s_sep = slacks['s_sep']
        sL, sR, sB, sT = slacks['sL'], slacks['sR'], slacks['sB'], slacks['sT']
        # Replace full matrix with vectorized reduction using np.minimum.reduceat
        s_pair_min = np.full(n, np.inf)
        np.minimum.at(s_pair_min, I, s_sep)
        np.minimum.at(s_pair_min, J, s_sep)
        c = np.minimum(s_pair_min, sL)
        c = np.minimum(c, sR)
        c = np.minimum(c, sB)
        c = np.minimum(c, sT)
        eps = 1e-6
        sxy = 1.0 + 0.30 / (eps + c)
        sr = 1.0 + 0.20 / (eps + c)
        sxy = np.clip(sxy, 1.0, 3.4)
        sr = np.clip(sr, 1.0, 2.4)
        return sxy, sr

    def pack_fb(x, y, r):
        return np.concatenate([x, y, r])

    def unpack_fb(v):
        return v[0:n], v[n:2*n], v[2*n:3*n]

    vfb = pack_fb(x, y, r)
    mfb = np.zeros_like(vfb)
    beta_fb = 0.96
    step_fb = 0.055
    t_weight_fb = 1.0
    t_mult_fb = 2.6
    max_stages_fb = 5
    steps_per_stage_fb = 230
    min_slack_tol_fb = 1e-11

    for stage in range(max_stages_fb):
        if time.time() - t0 > time_limit:
            break
        mfb[:] = 0.0
        alpha = step_fb
        val, g, ms, slacks = barrier_fixedbox_value_grad(x, y, r, left, right, bottom, top, t_weight_fb)
        if not np.isfinite(val) or ms <= 0:
            break

        for k in range(steps_per_stage_fb):
            if time.time() - t0 > time_limit:
                break

            sxy, srsc = clearance_scales_fixedbox(slacks)
            gx = g[0:n] * sxy
            gy = g[n:2*n] * sxy
            grv = g[2*n:3*n] * srsc
            g_pre = np.concatenate([gx, gy, grv])

            mfb = beta_fb * mfb + g_pre
            delta = alpha * mfb

            if k % 130 == 0:
                alpha *= 0.92
                if alpha < 1e-6:
                    break

            tau = 1.0
            accepted = False
            while tau > 1e-8:
                vnew = vfb + tau * delta
                xN, yN, rN = unpack_fb(vnew)
                val_new, g_new, ms_new, sl_new = barrier_fixedbox_value_grad(xN, yN, rN, left, right, bottom, top, t_weight_fb)
                if np.isfinite(val_new) and ms_new > min_slack_tol_fb and val_new >= val - 1e-10:
                    vfb = vnew
                    x, y, r = xN, yN, rN
                    val = val_new
                    g = g_new
                    slacks = sl_new
                    accepted = True
                    break
                tau *= 0.5

            if not accepted:
                alpha *= 0.54
                if alpha < 3e-6:
                    break
                continue

        t_weight_fb *= t_mult_fb
        step_fb = min(0.13, step_fb * 1.07)

    # Replace base parent's expand_radii_fixed_bbox with donor version
    def expand_radii_fixed_bbox(x, y, r, left, right, bottom, top, sweeps=16):
        r = r.copy()
        for sweep in range(sweeps):
            rmax = np.empty_like(r)
            for i in range(n):
                lim = min(x[i] - left, right - x[i], y[i] - bottom, top - y[i])
                for j in range(n):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j]) - r[j]
                    if d < lim:
                        lim = d
                        if lim <= 0:
                            break
                rmax[i] = max(1e-9, lim)
            order = np.argsort(rmax - r)[::-1]
            for i in order:
                lim = min(x[i] - left, right - x[i], y[i] - bottom, top - y[i])
                for j in range(n):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j]) - r[j]
                    if d < lim:
                        lim = d
                        if lim <= 0:
                            break
                lim = max(1e-9, lim)
                # Adaptive factor: increases faster in early sweeps, converges to higher final
                factor = 1.0 - np.exp(-sweep * 1.0)
                r[i] = min(lim, r[i] + factor * (lim - r[i]))
        return r

    r = expand_radii_fixed_bbox(x, y, r, left, right, bottom, top, sweeps=16)

    left = np.min(x - r)
    right = np.max(x + r)
    bottom = np.min(y - r)
    top = np.max(y + r)
    width = right - left
    height = top - bottom
    total_span = width + height
    if np.isfinite(total_span) and total_span > 0:
        scale = (2.0 - 1e-10) / total_span
        cx = 0.5 * (left + right)
        cy = 0.5 * (bottom + top)
        x = (x - cx) * scale + cx
        y = (y - cy) * scale + cy
        r = r * scale

    circles = np.column_stack([x, y, r])

    ok = True
    if not np.all(np.isfinite(circles)) or circles.shape != (n, 3):
        ok = False
    else:
        I_chk, J_chk = np.triu_indices(n, 1)
        dx_chk = circles[I_chk, 0] - circles[J_chk, 0]
        dy_chk = circles[I_chk, 1] - circles[J_chk, 1]
        dist = np.hypot(dx_chk, dy_chk)
        if np.any(dist < (circles[I_chk, 2] + circles[J_chk, 2]) - 1e-9):
            ok = False
        left = np.min(circles[:,0] - circles[:,2])
        right = np.max(circles[:,0] + circles[:,2])
        bottom = np.min(circles[:,1] - circles[:,2])
        top = np.max(circles[:,1] + circles[:,2])
        if (right - left) + (top - bottom) > 2.0 + 1e-8:
            ok = False

    if not ok:
        xmin, ymin = 0.0, 0.0
        xmax, ymax = 0.98, 0.98
        xs = np.linspace(xmin + 0.08, xmax - 0.08, 7)
        ys = np.linspace(ymin + 0.08, ymax - 0.08, 3)
        xx = np.repeat(xs, 3)
        yy = np.tile(ys, 7)
        rr = np.full(n, 0.0225)
        circles = np.column_stack([xx, yy, rr])

    return circles