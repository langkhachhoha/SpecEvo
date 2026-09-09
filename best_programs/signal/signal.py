import numpy as np
import time
import random
import math

def run_signal_processing(noisy_signal, window_size):
    '''
    Filter a 1D noisy signal using a sliding window of the given size.

    Args:
        noisy_signal: 1D numpy array of real-valued samples.
        window_size: int sliding-window length.

    Returns:
        A dict with key ``filtered_signal`` containing a 1D array of length
        ``len(noisy_signal) - window_size + 1``.
    '''
    x = np.asarray(noisy_signal, dtype=float).ravel()
    L = len(x)
    W = int(window_size)
    if W <= 0 or L < W:
        return {"filtered_signal": np.array([], dtype=float)}

    out_len = L - W + 1
    y_out = np.empty(out_len, dtype=float)

    t_win = np.arange(W, dtype=float) / float(max(W - 1, 1))
    t_star = 1.0
    s1 = float(W)
    st = float(np.sum(t_win))
    stt = float(np.sum(t_win * t_win))

    x_hat = np.zeros(2, dtype=float)
    P = np.eye(2, dtype=float) * 1e3
    nis_ewma = 1.0

    def robust_linear_trend_irls(t, y, s1, st, stt, iters=3):
        sy = float(np.sum(y))
        sty = float(np.sum(t * y))
        det = s1 * stt - st * st
        if abs(det) < 1e-12:
            a = sy / max(s1, 1.0)
            b = 0.0
        else:
            a = (stt * sy - st * sty) / det
            b = (s1 * sty - st * sy) / det
        for _ in range(iters):
            resid = y - (a + b * t)
            s = 1.4826 * (np.median(np.abs(resid)) + 1e-12)
            if not np.isfinite(s) or s <= 0:
                break
            k = 1.3
            r_abs = np.abs(resid)
            w = np.ones_like(resid)
            idx = r_abs > (k * s)
            w[idx] = (k * s) / (r_abs[idx] + 1e-12)
            ws = float(np.sum(w))
            wst = float(np.sum(w * t))
            wstt = float(np.sum(w * t * t))
            wsy = float(np.sum(w * y))
            wsty = float(np.sum(w * t * y))
            wdet = ws * wstt - wst * wst
            if abs(wdet) < 1e-12:
                break
            a = (wstt * wsy - wst * wsty) / wdet
            b = (ws * wsty - wst * wsy) / wdet
        return float(a), float(b)

    def periodic_features_strength(u):
        n = u.size
        if n < 4:
            return 0.0, 0.0
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / float(max(n - 1, 1)))
        v = u * hann
        spec = np.abs(np.fft.rfft(v))
        if spec.size <= 1:
            return 0.0, 0.0
        spec[0] = 0.0
        start_bin = 2 if spec.size > 2 else 1
        peak_idx = start_bin + int(np.argmax(spec[start_bin:]))
        peak = float(spec[peak_idx])
        total = float(np.sum(spec) + 1e-12)
        f_cyc_per_win = float(peak_idx) / float(n)
        strength = peak / total
        if f_cyc_per_win < 0.05:
            strength *= 0.35
        return float(np.clip(strength, 0.0, 1.0)), f_cyc_per_win

    def robust_innovation_variance(w, trend):
        resid = w - trend
        mad = 1.4826 * (np.median(np.abs(resid)) + 1e-12)
        var_est = mad * mad
        return max(1e-6, min(var_est, 10.0))

    def kalman_predict_update(x_hat, P, z, Q_s, R_s):
        F = np.array([[1.0, 1.0],
                      [0.0, 1.0]], dtype=float)
        H = np.array([1.0, 0.0], dtype=float)
        Qm = Q_s * np.eye(2, dtype=float)

        x_pred = F @ x_hat
        P_pred = F @ P @ F.T + Qm
        mu_pred = float(H @ x_pred)

        y_innov = float(z - mu_pred)
        S = float(H @ P_pred @ H.T + R_s)
        S = max(S, 1e-12)
        K = (P_pred @ H[:, None]) / S
        x_new = x_pred + (K.flatten() * y_innov)
        P_new = (np.eye(2) - K @ H[None, :]) @ P_pred

        # Replace log-sigmoid with tanh-based robust innovation adjustment
        innov_scale = 1.4826 * (np.median(np.abs(y_innov)) + 1e-12)
        delta_scaled = y_innov / max(1e-12, innov_scale)
        # Apply tanh-based soft clipping to limit outlier impact
        clipped_delta = np.tanh(0.5 * np.clip(np.abs(delta_scaled), 0.0, 8.0)) * np.sign(delta_scaled)
        delta_adj = y_innov * clipped_delta

        mu_post = float(x_new[0] + K.flatten()[0] * delta_adj)

        return x_new, P_new, mu_pred, S, y_innov, mu_post

    # Precompute distance from the window's right end (causal, one-sided)
    dist_template = (W - 1) - np.arange(W, dtype=float)
    u_template = -dist_template

    # Candidate bandwidths (in samples), clipped to window and ensuring coverage
    base_scales = np.array([1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 14.0, 18.0], dtype=float)
    max_scale = max(1.2, 0.95 * W)
    scales = np.unique(np.clip(base_scales, 1.2, max_scale))
    if scales.size < 3:
        extra = np.linspace(1.5, max(3.0, 0.6 * W), 3)
        scales = np.unique(np.clip(np.concatenate([scales, extra]), 1.2, max_scale))
    scales.sort()

    prev1 = None
    prev2 = None

    eps = 1e-12

    for i in range(out_len):
        w = x[i:i + W]

        d = np.diff(w)
        med_d = np.median(np.abs(d)) if d.size else 0.0
        sigma_d = (med_d / 0.6745) / math.sqrt(2.0) if med_d > 0 else 1e-12
        sigma = max(1e-6, sigma_d)

        last_jump = abs(w[-1] - w[-2]) if W >= 2 else 0.0
        thr = 1.2 * sigma
        step_strength = 0.0 if thr <= 0 else float(np.clip((last_jump - thr) / (2.0 * thr), 0.0, 1.0))

        tail_len = int(max(3, min(6, W)))
        tail_med = float(np.median(w[-tail_len:]))

        step_capped_max = max(2.5, min(max_scale, 5.0 + 6.0 * (1.0 - step_strength)))
        usable_scales = np.clip(scales, 1.2, step_capped_max)

        yhat_list = []
        loglik_list = []
        s_list = []
        for s in usable_scales:
            dist = dist_template
            u = u_template
            w_space = np.exp(-0.5 * (dist / (s + eps)) ** 2)

            w_range = np.exp(-0.5 * ((w - tail_med) / (1.2 - 0.7 * step_strength) * sigma + eps) ** 2)

            wt = w_space * w_range
            wt[-1] = max(wt[-1], 1e-3)

            S0 = float(np.sum(wt))
            if not np.isfinite(S0) or S0 <= 1e-12:
                yhat = float(w[-1])
                yhat_list.append(yhat)
                loglik_list.append(-1e6)
                s_list.append(float(s))
                continue

            S1 = float(np.sum(wt * u))
            S2 = float(np.sum(wt * u * u))
            T0 = float(np.sum(wt * w))
            T1 = float(np.sum(wt * w * u))

            denom = S0 * S2 - S1 * S1
            if not np.isfinite(denom) or denom < 1e-10:
                yhat = T0 / (S0 + eps)
                eff = S0
            else:
                a_hat = (S2 * T0 - S1 * T1) / (denom + eps)
                yhat = a_hat
                eff = S0

            resid = w - yhat
            var_est = float(np.mean(resid * resid))
            loglik = -0.5 * (W * math.log(2.0 * math.pi * var_est) + W)
            yhat_list.append(float(yhat))
            loglik_list.append(loglik)
            s_list.append(float(s))

        yhat_arr = np.array(yhat_list, dtype=float)
        loglik_arr = np.array(loglik_list, dtype=float)
        s_arr = np.array(s_list, dtype=float)

        # Apply trimmed median fusion instead of Bayesian model averaging
        loglik_arr = loglik_arr - np.max(loglik_arr)
        weights = np.exp(loglik_arr)
        total_weight = float(np.sum(weights) + 1e-12)
        normalized_weights = weights / total_weight
        clipped_weights = np.clip(normalized_weights, 0.0, 0.3)

        # If all weights are zero, default to last observation
        if total_weight < 1e-12:
            yi = float(w[-1])
        else:
            # Trimmed median fusion: sort by value, cumulative weight, pick median
            combined = sorted(zip(yhat_arr, clipped_weights), key=lambda x: x[0])
            cum_weight = 0.0
            median_idx = 0.5
            for val, wgt in combined:
                cum_weight += wgt
                if cum_weight >= median_idx:
                    yi = val
                    break

        a, b = robust_linear_trend_irls(t_win, w, s1, st, stt)
        trend = a + b * t_win
        r = w - trend
        per_strength, f_cyc = periodic_features_strength(r)
        period = 1.0 / f_cyc if f_cyc > 0.0 else W * 12.0

        R_base = robust_innovation_variance(w, trend)
        # Replaced EWMA with direct Huber-scale estimator for innovation variance
        # Compute robust innovation scale directly from innovation residuals
        innov_resid = np.diff(np.concatenate([np.array([float(a + b * t_win[-1])]), w[-1:]]))  # Predicted vs actual
        innov_mad = 1.4826 * (np.median(np.abs(innov_resid)) + 1e-12)
        R_eff = max(1e-6, min(innov_mad * innov_mad, 2.0))  # Robust innovation variance estimate

        Q_s = 1e-5 * (1.0 + 1000.0 * float(np.var(w) + 1e-12))

        z = float(w[-1])
        x_hat, P, mu_pred, S_innov, y_innov, mu_kf = kalman_predict_update(x_hat, P, z, Q_s, R_eff)
        innov_diffs = np.diff(np.concatenate([np.array([mu_pred]), [z]]))
        nis_mad = 1.4826 * (np.median(np.abs(innov_diffs)) + 1e-12) if len(innov_diffs) > 1 else 1.0
        nis_cur = float(np.clip(nis_mad, 0.0, 8.0))
        nis_ewma = 0.9 * nis_ewma + 0.1 * nis_cur

        dist = (W - 1) - np.arange(W, dtype=float)
        center_val = np.median(w[-min(6, W):])
        dr = w - center_val
        w_range = np.exp(-0.5 * (dr / (sigma + 1e-12)) ** 2)

        s_fine = 2.5
        s_period = float(np.clip(0.5 * period, 2.0, 1.8 * W))
        s_coarse = 1.0 * W
        sigma_s_list = [s_fine, s_period, s_coarse]

        y_candidates = []
        weights = []
        for s in sigma_s_list:
            w_space = np.exp(-0.5 * (dist / (s + 1e-12)) ** 2)
            ww = w_space * w_range
            sw = float(np.sum(ww))
            if sw <= 1e-12:
                yhat = float(w[-1])
                weight = 0.0
            else:
                yhat = float(np.dot(ww, w) / sw)
                weight = float(sw)
            y_candidates.append(yhat)
            weights.append(weight)

        y_trend = float(a + b * t_star)
        y_candidates.append(y_trend)
        weights.append(1.0 + 10.0 * abs(b) * W)

        y_candidates.append(mu_kf)
        weights.append(5.0 * (1.0 - step_strength) * (0.6 + 0.4 * per_strength))

        total_w = float(np.sum(weights) + 1e-12)
        normalized_weights = [w / total_w for w in weights]
        clipped_weights = [min(w, 0.3) for w in normalized_weights]

        if len(y_candidates) == 1:
            yi0 = y_candidates[0]
        else:
            # Trimmed median fusion: sort by value, use cumulative weight
            combined = sorted(zip(y_candidates, clipped_weights), key=lambda x: x[0])
            cum_weight = 0.0
            median_idx = 0.5
            for val, wgt in combined:
                cum_weight += wgt
                if cum_weight >= median_idx:
                    yi0 = val
                    break

        kf_w = 0.3 * (1.0 - step_strength) * (0.6 + 0.4 * per_strength)
        yi_blend = (1.0 - kf_w) * yi0 + kf_w * mu_kf

        yi = yi_blend
        if prev1 is not None:
            delta = abs(yi - prev1)
            noise_level = 1.5 * sigma
            trend_strength = abs(b) * float(W)
            threshold = noise_level * (1.0 + 0.5 * min(1.0, trend_strength / (1e-6 + np.max([sigma, 1e-6]))))
            if prev2 is not None and delta < threshold:
                if abs(yi - prev1) < 2 * noise_level and abs(prev1 - prev2) < 2 * noise_level:
                    yi = prev1
            elif step_strength > 0.6:
                yi = float(w[-1])

        w_min, w_max = float(np.min(w)), float(np.max(w))
        pad = 1.6 * (sigma + 1e-12)
        yi = float(np.clip(yi, w_min - pad, w_max + pad))
        if not np.isfinite(yi):
            yi = float(w[-1])

        y_out[i] = yi
        prev2 = prev1
        prev1 = yi

    return {"filtered_signal": y_out}