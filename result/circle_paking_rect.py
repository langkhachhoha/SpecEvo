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
    rng = np.random.default_rng()
    N = 21
    start_time = time.time()
    time_budget = 600.0

    # Utilities: smooth transforms
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def softplus(x, beta=4.0):
        return np.log1p(np.exp(beta * x)) / beta

    def softplus_grad(x, beta=4.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    # Feasibility and polishing
    def is_feasible(xy, r, W, H, tol=1e-9):
        if np.any(r < -tol) or np.any(r > W + H):
            return False
        if np.any(xy[:,0] - r < -tol) or np.any(xy[:,0] + r > W + tol):
            return False
        if np.any(xy[:,1] - r < -tol) or np.any(xy[:,1] + r > H + tol):
            return False
        dx = xy[:, None, 0] - xy[None, :, 0]
        dy = xy[:, None, 1] - xy[None, :, 1]
        D2 = dx * dx + dy * dy
        iu = np.triu_indices(N, 1)
        R = r[:, None] + r[None, :]
        return np.all(D2[iu] + tol >= (R[iu]) ** 2)

    def final_polish(xy, r, W, H):
        # Keep within walls
        b = np.minimum.reduce([xy[:,0], W - xy[:,0], xy[:,1], H - xy[:,1]])
        r = np.maximum(0.0, np.minimum(r, b))
        # Resolve pairwise overlaps
        for _ in range(10):
            dx = xy[:, None, 0] - xy[None, :, 0]
            dy = xy[:, None, 1] - xy[None, :, 1]
            D = np.sqrt(np.maximum(0.0, dx * dx + dy * dy))
            iu = np.triu_indices(N, 1)
            Rsum = r[:, None] + r[None, :]
            overlap = Rsum[iu] - D[iu]
            mask = overlap > 1e-12
            if not np.any(mask):
                break
            I = iu[0][mask]; J = iu[1][mask]
            for k in range(I.size):
                i = I[k]; j = J[k]
                dij = max(D[i, j], 1e-12)
                ov = (r[i] + r[j]) - dij
                if ov > 0.0:
                    tot = r[i] + r[j] + 1e-12
                    fi = r[i] / tot
                    r[i] = max(0.0, r[i] - 0.5 * ov * fi)
                    r[j] = max(0.0, r[j] - 0.5 * ov * (1.0 - fi))
            b = np.minimum.reduce([xy[:,0], W - xy[:,0], xy[:,1], H - xy[:,1]])
            r = np.maximum(0.0, np.minimum(r, b))
        # Translate to origin
        minx = float(np.min(xy[:,0] - r))
        miny = float(np.min(xy[:,1] - r))
        xy[:,0] -= min(0.0, minx)
        xy[:,1] -= min(0.0, miny)
        # Compute bounding box
        left = float(np.min(xy[:,0] - r))
        right = float(np.max(xy[:,0] + r))
        bottom = float(np.min(xy[:,1] - r))
        top = float(np.max(xy[:,1] + r))
        Wb = right - left
        Hb = top - bottom
        xy[:,0] -= left
        xy[:,1] -= bottom
        # Scale to perimeter 4
        if Wb + Hb > 2.0:
            s = 2.0 / (Wb + Hb + 1e-12)
            xy *= s
            r *= s
        # Final safety
        dx = xy[:, None, 0] - xy[None, :, 0]
        dy = xy[:, None, 1] - xy[None, :, 1]
        D = np.sqrt(np.maximum(0.0, dx * dx + dy * dy))
        iu = np.triu_indices(N, 1)
        for k in range(len(iu[0])):
            i = iu[0][k]; j = iu[1][k]
            if r[i] + r[j] > D[i, j] + 1e-12:
                ov = r[i] + r[j] - D[i, j]
                tot = r[i] + r[j] + 1e-12
                fi = r[i] / tot
                r[i] = max(0.0, r[i] - ov * fi * 1.01)
                r[j] = max(0.0, r[j] - ov * (1.0 - fi) * 1.01)
        return np.hstack([xy, r[:, None]])

    # Core optimization
    def optimize_once(W_init=None, runtime=70.0):
        t0 = time.time()
        if W_init is None:
            s0 = rng.uniform(0.2, 1.8) / 2.0
        else:
            s0 = np.clip(W_init / 2.0, 0.05, 0.95)
        u = np.log(s0 / (1.0 - s0))

        px = rng.normal(0.0, 0.8, size=N)
        py = rng.normal(0.0, 0.8, size=N)
        q = rng.normal(-2.8, 0.3, size=N)

        m_px = np.zeros_like(px); v_px = np.zeros_like(px)
        m_py = np.zeros_like(py); v_py = np.zeros_like(py)
        m_q = np.zeros_like(q); v_q = np.zeros_like(q)
        m_u = 0.0; v_u = 0.0

        beta1 = 0.9; beta2 = 0.999; eps_adam = 1e-8
        mu_list = [1.0, 4.0, 16.0, 64.0, 256.0]
        iters_per_mu = 450
        base_lr = 0.03

        best_sum = -1.0
        best_out = None

        step_global = 0
        for mu in mu_list:
            if time.time() - t0 > runtime:
                break
            # Fixed decay schedule: high LR in early stages, reduce sharply after mid-phase
            if mu <= 4.0:
                lr = base_lr
            elif mu <= 64.0:
                lr = base_lr * 0.5
            else:
                lr = base_lr * 0.1
            lr = max(lr, 1e-5)  # Prevent vanishing gradients

            for it in range(iters_per_mu):
                step_global += 1
                if time.time() - t0 > runtime:
                    break

                su = sigmoid(u)
                w = 2.0 * su
                h = 2.0 - w
                dwdu = 2.0 * su * (1.0 - su)
                dhdu = -dwdu

                sx = sigmoid(px)
                sy = sigmoid(py)
                xi = w * sx
                yi = h * sy

                ri = softplus(q, beta=4.0)
                dri_dq = softplus_grad(q, beta=4.0)

                g_x = np.zeros(N)
                g_y = np.zeros(N)
                g_r = -np.ones(N)
                g_w_exp = 0.0
                g_h_exp = 0.0

                # Border penalties
                b1 = ri - xi
                m1 = b1 > 0.0
                if np.any(m1):
                    c1 = 2.0 * mu * b1[m1]
                    g_r[m1] += c1
                    g_x[m1] -= c1

                b2 = ri - w + xi
                m2 = b2 > 0.0
                if np.any(m2):
                    c2 = 2.0 * mu * b2[m2]
                    g_r[m2] += c2
                    g_x[m2] += c2
                    g_w_exp += -float(np.sum(c2))

                b3 = ri - yi
                m3 = b3 > 0.0
                if np.any(m3):
                    c3 = 2.0 * mu * b3[m3]
                    g_r[m3] += c3
                    g_y[m3] -= c3

                b4 = ri - h + yi
                m4 = b4 > 0.0
                if np.any(m4):
                    c4 = 2.0 * mu * b4[m4]
                    g_r[m4] += c4
                    g_y[m4] += c4
                    g_h_exp += -float(np.sum(c4))

                # Pairwise penalties
                dx = xi[:, None] - xi[None, :]
                dy = yi[:, None] - yi[None, :]
                D = np.sqrt(dx * dx + dy * dy + 1e-12)
                Rsum = ri[:, None] + ri[None, :]
                gij = Rsum - D
                iu = np.triu_indices(N, 1)
                gij_u = gij[iu]
                act = gij_u > 0.0
                if np.any(act):
                    S = np.zeros((N, N))
                    val = 2.0 * mu * gij_u[act]
                    I = iu[0][act]; J = iu[1][act]
                    S[I, J] = val
                    S += S.T
                    Ux = dx / D
                    Uy = dy / D
                    g_x += -np.sum(S * Ux, axis=1)
                    g_y += -np.sum(S * Uy, axis=1)
                    g_r += np.sum(S, axis=1)

                # Chain rule
                dL_dw = g_w_exp + float(np.sum(g_x * sx))
                dL_dh = g_h_exp + float(np.sum(g_y * sy))
                dpx = g_x * (w * sx * (1.0 - sx))
                dpy = g_y * (h * sy * (1.0 - sy))
                dq = g_r * dri_dq
                du = dL_dw * dwdu + dL_dh * dhdu

                b1t = beta1
                b2t = beta2

                m_px = b1t * m_px + (1 - b1t) * dpx
                v_px = b2t * v_px + (1 - b2t) * (dpx * dpx)
                m_hat_px = m_px / (1 - b1t ** step_global)
                v_hat_px = v_px / (1 - b2t ** step_global)
                px -= lr * m_hat_px / (np.sqrt(v_hat_px) + eps_adam)

                m_py = b1t * m_py + (1 - b1t) * dpy
                v_py = b2t * v_py + (1 - b2t) * (dpy * dpy)
                m_hat_py = m_py / (1 - b1t ** step_global)
                v_hat_py = v_py / (1 - b2t ** step_global)
                py -= lr * m_hat_py / (np.sqrt(v_hat_py) + eps_adam)

                m_q = b1t * m_q + (1 - b1t) * dq
                v_q = b2t * v_q + (1 - b2t) * (dq * dq)
                m_hat_q = m_q / (1 - b1t ** step_global)
                v_hat_q = v_q / (1 - b2t ** step_global)
                q -= lr * m_hat_q / (np.sqrt(v_hat_q) + eps_adam)

                m_u = b1t * m_u + (1 - b1t) * du
                v_u = b2t * v_u + (1 - b2t) * (du * du)
                m_hat_u = m_u / (1 - b1t ** step_global)
                v_hat_u = v_u / (1 - b2t ** step_global)
                u -= 0.6 * lr * m_hat_u / (np.sqrt(v_hat_u) + eps_adam)

                # Snapshot
                if it % 60 == 0 or it == iters_per_mu - 1:
                    su = sigmoid(u)
                    w = 2.0 * su
                    h = 2.0 - w
                    sx = sigmoid(px)
                    sy = sigmoid(py)
                    xi = w * sx
                    yi = h * sy
                    ri = softplus(q, beta=4.0)
                    xy = np.stack([xi, yi], axis=1)
                    b = np.minimum.reduce([xy[:,0], w - xy[:,0], xy[:,1], h - xy[:,1]])
                    ri = np.maximum(0.0, np.minimum(ri, b))
                    C = final_polish(xy, ri, w, h)
                    ssum = float(np.sum(C[:,2]))
                    if np.all(np.isfinite(C)) and ssum > best_sum + 1e-12:
                        best_sum = ssum
                        best_out = C

            if time.time() - t0 > runtime:
                break

        if best_out is None:
            su = sigmoid(u)
            w = 2.0 * su
            h = 2.0 - w
            sx = sigmoid(px)
            sy = sigmoid(py)
            xi = w * sx
            yi = h * sy
            ri = softplus(q, beta=4.0)
            xy = np.stack([xi, yi], axis=1)
            b = np.minimum.reduce([xy[:,0], w - xy[:,0], xy[:,1], h - xy[:,1]])
            ri = np.maximum(0.0, np.minimum(ri, b))
            best_out = final_polish(xy, ri, w, h)
        return best_out

    # Multi-start
    best_global = None
    best_sum_global = -1.0
    width_candidates = [0.6, 0.8, 1.0, 1.2, 1.4]
    per_run_time = max(4.0, (time_budget - 5.0) / (len(width_candidates) * 2))

    for W0 in width_candidates:
        for _ in range(2):
            if time.time() - start_time > time_budget:
                break
            C = optimize_once(W_init=W0, runtime=per_run_time)
            if C is None or C.shape != (N, 3) or not np.all(np.isfinite(C)):
                continue
            ssum = float(np.sum(C[:,2]))
            if ssum > best_sum_global:
                best_sum_global = ssum
                best_global = C

    # Fallback
    if best_global is None or best_global.shape != (N, 3) or not np.all(np.isfinite(best_global)):
        W = H = 1.0
        nx, ny = 7, 3
        xs = (np.arange(nx) + 0.5) * (W / nx)
        ys = (np.arange(ny) + 0.5) * (H / ny)
        pts = np.array([(x, y) for y in ys for x in xs])[:N]
        r = np.full(N, 0.03)
        return np.hstack([pts, r[:, None]])

    # Final normalization
    C = best_global.copy()
    left = float(np.min(C[:,0] - C[:,2]))
    right = float(np.max(C[:,0] + C[:,2]))
    bottom = float(np.min(C[:,1] - C[:,2]))
    top = float(np.max(C[:,1] + C[:,2]))
    Wb = right - left
    Hb = top - bottom
    C[:,0] -= left
    C[:,1] -= bottom
    if Wb + Hb > 2.0:
        s = 2.0 / (Wb + Hb + 1e-12)
        C[:, :2] *= s
        C[:, 2] *= s
    dx = C[:, None, 0] - C[None, :, 0]
    dy = C[:, None, 1] - C[None, :, 1]
    D = np.sqrt(np.maximum(0.0, dx * dx + dy * dy))
    iu = np.triu_indices(N, 1)
    for k in range(len(iu[0])):
        i = iu[0][k]; j = iu[1][k]
        if C[i,2] + C[j,2] > D[i,j] + 1e-12:
            ov = C[i,2] + C[j,2] - D[i,j]
            tot = C[i,2] + C[j,2] + 1e-12
            fi = C[i,2] / tot
            C[i,2] = max(0.0, C[i,2] - ov * fi * 1.01)
            C[j,2] = max(0.0, C[j,2] - ov * (1.0 - fi) * 1.01)
    return C