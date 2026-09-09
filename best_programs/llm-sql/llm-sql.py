import pandas as pd
import numpy as np
from typing import List, Optional
import time
import random
import heapq

def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    '''
    Reorder DataFrame columns to maximize prefix hit rate.

    Args:
        df: Input DataFrame (raw, columns NOT pre-merged)
        early_stop: Early stopping threshold
        row_stop: Row stopping threshold
        col_stop: Column stopping threshold
        col_merge: List of column groups to merge
        one_way_dep: One-way dependencies (unused)
        distinct_value_threshold: Threshold for distinct values
        parallel: Whether to use parallel processing

    Returns:
        DataFrame with merged columns, reordered columns and rows
    '''
    # Defensive copy
    df = df.copy()

    # 1) Apply explicit column merging
    if col_merge:
        for cols_to_merge in col_merge:
            valid = [c for c in cols_to_merge if c in df.columns]
            if len(valid) == len(cols_to_merge) and len(valid) > 0:
                merged_name = "_".join(cols_to_merge)
                df[merged_name] = df[valid].apply(
                    lambda x: "".join([f"{val}" for val in x]), axis=1
                )
                df = df.drop(columns=valid)

    # Trivial cases
    if df.shape[1] == 0:
        return df
    if df.shape[1] == 1:
        return df.sort_values(by=list(df.columns), kind="mergesort")
    if df.shape[0] <= 1:
        return df

    random.seed(42)
    np.random.seed(42)
    cols: List[str] = list(df.columns)
    m = len(cols)
    n = len(df)

    # 2) Factorize columns to integer codes + compute weights by mean string length
    col_codes: List[np.ndarray] = []
    avg_len: List[float] = []
    for c in cols:
        s = df[c].astype(str)
        codes, _ = pd.factorize(s, sort=False)
        col_codes.append(codes.astype(np.int32, copy=False))
        L = s.str.len()
        avg_len.append(max(1.0, float(L.mean()) if len(L) else 1.0))
    weights = np.array(avg_len, dtype=np.float64)

    # 3) Sample rows to evaluate surrogate quickly
    rng = np.random.RandomState(42)
    samp_n_eval = min(n, max(6000, min(18000, early_stop)))
    if samp_n_eval < n:
        eval_idx = np.sort(rng.choice(n, samp_n_eval, replace=False))
    else:
        eval_idx = np.arange(n, dtype=np.int64)
    S = int(len(eval_idx))
    if S == 0:
        return df

    sample_codes = [arr[eval_idx].astype(np.uint64, copy=False) for arr in col_codes]
    sample_nunique = np.array([np.unique(sc).size for sc in sample_codes], dtype=np.int64)
    global_nunique = np.array([int(pd.unique(col_codes[i]).size) for i in range(m)], dtype=np.int64)

    # Distinctness-guided depth hints
    base_k = int(min(row_stop, m, 8))
    depth_per_col = np.full(m, max(1, base_k), dtype=np.int32)
    max_samp = max(1, int(sample_nunique.max()))
    max_glob = max(1, int(global_nunique.max()))
    for i in range(m):
        r_samp = sample_nunique[i] / max_samp
        r_glob = global_nunique[i] / max_glob
        if max(r_samp, r_glob) > distinct_value_threshold:
            depth_per_col[i] = 2

    # Rolling hash constants
    MUL = np.uint64(1469598103934665603)
    GOLD = np.uint64(0x9e3779b97f4a7c15)

    # Bounded caches
    hash_cache = {}   # (tuple(order_prefix), depth) -> (hash_vec, unique_count)
    fitness_cache = {}  # (tuple(order), depth_cap) -> score

    def count_unique_from_hash(h: np.ndarray) -> int:
        if h.size == 0:
            return 0
        sorted_h = np.sort(h)
        return int(np.count_nonzero(np.diff(sorted_h)) + 1)

    def get_hash_and_unique(order_prefix: List[int], depth: int):
        key = (tuple(order_prefix), depth)
        cached = hash_cache.get(key)
        if cached is not None:
            return cached
        if depth == 1:
            h = sample_codes[order_prefix[0]] + GOLD
        else:
            prev_h, _ = get_hash_and_unique(order_prefix[:-1], depth - 1)
            h = (prev_h * MUL) ^ (sample_codes[order_prefix[-1]] + GOLD)
        u = count_unique_from_hash(h)
        if len(hash_cache) > 18000:
            for k in list(hash_cache.keys())[::3]:
                del hash_cache[k]
        hash_cache[key] = (h, u)
        return h, u

    # 5) Surrogate fitness: weighted sum of collisions across first k prefixes
    def fitness(order_idx: List[int]) -> float:
        if not order_idx:
            return 0.0
        used_k = min(row_stop, len(order_idx))
        key = (tuple(order_idx), used_k)
        cached = fitness_cache.get(key)
        if cached is not None:
            return cached
        score = 0.0
        depth = min(depth_per_col[order_idx[0]], used_k)
        for d in range(1, depth + 1):
            _, u = get_hash_and_unique(order_idx[:d], d)
            score += weights[order_idx[d - 1]] * (S - u)
        val = float(score)
        if len(fitness_cache) > 3500:
            for k in list(fitness_cache.keys())[::5]:
                del fitness_cache[k]
        fitness_cache[key] = val
        return val

    # 6) Pairwise synergy matrix (downsampled)
    S2 = min(S, 3584)
    if S2 < S:
        sub_idx = np.sort(rng.choice(S, S2, replace=False))
    else:
        sub_idx = np.arange(S2, dtype=np.int64)
    synergy_codes = [sc[sub_idx] for sc in sample_codes]

    def unique_count(arr: np.ndarray) -> int:
        if arr.size == 0:
            return 0
        sorted_arr = np.sort(arr)
        return int(np.count_nonzero(np.diff(sorted_arr)) + 1)

    base_collisions = np.zeros(m, dtype=np.float64)
    for i in range(m):
        u = unique_count(synergy_codes[i] + GOLD)
        base_collisions[i] = S2 - u

    synergy = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        hi = synergy_codes[i] + GOLD
        for j in range(m):
            if i == j:
                synergy[i, j] = base_collisions[j]
            else:
                hij = (hi * MUL) ^ (synergy_codes[j] + GOLD)
                uij = unique_count(hij)
                synergy[i, j] = S2 - uij

    # 7) Seeds: cardinality, length, greedy, synergy-chain
    order_low_card = list(np.argsort(sample_nunique).astype(int))
    order_len_desc = list(np.argsort([-weights[i] for i in range(m)]).astype(int))

    # Greedy seed from B — better incremental hash and score propagation
    def greedy_seed() -> List[int]:
        remaining = set(range(m))
        best_first = None
        best_score_local = -1.0
        for i in remaining:
            u = sample_nunique[i]
            sc = weights[i] * (S - u)
            if sc > best_score_local:
                best_score_local = sc
                best_first = i
        order = []
        if best_first is None:
            return list(range(m))
        order.append(best_first)
        remaining.remove(best_first)
        code = sample_codes[best_first] + GOLD

        while remaining:
            best_next = None
            best_gain = -1.0
            best_code_next = None
            for j in remaining:
                cand_code = (code * MUL) ^ (sample_codes[j] + GOLD)
                sorted_code = np.sort(cand_code)
                u = np.count_nonzero(np.diff(sorted_code)) + 1 if S > 0 else 0
                sc = weights[j] * (S - u)
                if sc > best_gain:
                    best_gain = sc
                    best_next = j
                    best_code_next = cand_code
            order.append(best_next)
            remaining.remove(best_next)
            code = best_code_next
        return order

    greedy_order = greedy_seed()
    greedy_rev = list(reversed(greedy_order))

    population: List[List[int]] = []
    seeds = [order_low_card, order_len_desc, greedy_order, greedy_rev, list(range(m))]
    for base in seeds:
        krot = min(5, m)
        for r in range(krot):
            population.append(base[r:] + base[:r])

    # Targeted random candidates
    low_card_indices = [i for i in range(m) if sample_nunique[i] <= sample_nunique.max() * 0.35]
    high_weight_indices = [i for i in range(m) if weights[i] >= weights.mean() * 0.97]
    remaining_idx = [i for i in range(m) if i not in low_card_indices and i not in high_weight_indices]
    target_pop = 32
    for _ in range(max(0, target_pop - len(population))):
        cand = []
        if high_weight_indices:
            k1 = min(6, len(high_weight_indices))
            cand.extend(random.sample(high_weight_indices, k1))
        if low_card_indices:
            k2 = min(6, len(low_card_indices))
            cand.extend(random.sample(low_card_indices, k2))
        if remaining_idx:
            k3 = min(4, len(remaining_idx))
            if k3 > 0:
                cand.extend(random.sample(remaining_idx, k3))
        random.shuffle(cand)
        for i in range(m):
            if i not in cand:
                cand.append(i)
        population.append(cand[:m])

    # Deduplicate
    seen_pop = set()
    uniq_pop = []
    for p in population:
        tp = tuple(p)
        if tp not in seen_pop:
            seen_pop.add(tp)
            uniq_pop.append(p)
    population = uniq_pop[:target_pop]

    # Score seeds
    scores = [fitness(p) for p in population]
    best_idx = int(np.argmax(scores))
    best_order = population[best_idx]
    best_score = float(scores[best_idx])

    # 8) Beam search with synergy-pruned moves and adaptive depth (from A)
    t_start = time.time()
    time_budget = min(2.8, max(1.1, 0.11 * m + 0.9))
    beam_cap = 100
    max_evals = 550

    heap = []
    counter = 0
    for p, sc in zip(population, scores):
        heapq.heappush(heap, (-sc, counter, p))
        counter += 1

    memo_best = {}
    stall = 0
    metropolis_on = False

    def candidate_positions_for_move(order: List[int], i: int, col: int, limit: int) -> List[int]:
        mlen = len(order)
        base_positions = [0, mlen]
        strong_partners = np.argsort(-synergy[:, col])[:min(5, mlen)]
        for sp in strong_partners:
            try:
                pos = order.index(sp) + 1
                base_positions.append(pos)
            except ValueError:
                continue
        base_positions = sorted(set([max(0, min(mlen, p)) for p in base_positions]))
        order_wo = order[:i] + order[i+1:]
        deltas = []
        for p in base_positions:
            p_adj = p if p <= i else p - 1
            left = order_wo[p_adj - 1] if p_adj - 1 >= 0 else None
            right = order_wo[p_adj] if p_adj < len(order_wo) else None
            gain = 0.0
            if left is not None:
                gain += synergy[left, col]
            else:
                gain += base_collisions[col]
            if right is not None:
                gain += synergy[col, right]
            if left is not None and right is not None:
                gain -= synergy[left, right]
            deltas.append((gain, p))
        deltas.sort(reverse=True)
        return sorted(set([p for _, p in deltas[:max(1, limit)]]))

    evals = 0
    depth_cap = int(min(row_stop, max(2, min(row_stop, 4))))
    max_depth_cap = int(min(row_stop, m))

    while heap and (time.time() - t_start) < time_budget and evals < max_evals:
        neg_sc, _, order = heapq.heappop(heap)
        cur_sc = -neg_sc

        key = tuple(order)
        prev_best = memo_best.get(key)
        if prev_best is not None and prev_best >= cur_sc:
            continue
        memo_best[key] = cur_sc

        improved_local = False

        for i in range(m):
            col_idx = order[i]
            try_positions = candidate_positions_for_move(order, i, col_idx, max(1, col_stop))
            for p in try_positions:
                if p == i or p == i + 1:
                    continue
                cand = order[:]
                val = cand.pop(i)
                insert_at = p if p <= i else p - 1
                cand.insert(insert_at, val)
                sc = fitness(cand)
                evals += 1

                temp = max(0.01, 0.5 * (1.0 - (time.time() - t_start) / time_budget))
                if sc > cur_sc:
                    accept = True
                else:
                    delta_score = sc - cur_sc
                    accept_prob = np.exp(delta_score / temp) if temp > 0 else 0.0
                    accept = random.random() < accept_prob

                if accept:
                    if sc > best_score:
                        best_score = sc
                        best_order = cand
                        improved_local = True
                    heapq.heappush(heap, (-sc, counter, cand))
                    counter += 1
                    if len(heap) > beam_cap:
                        heapq.heappop(heap)
                if (time.time() - t_start) >= time_budget or evals >= max_evals:
                    break
            if (time.time() - t_start) >= time_budget or evals >= max_evals:
                break

        if improved_local:
            stall = 0
        else:
            stall += 1
            if stall >= 3:
                metropolis_on = True

    # 9) Polishing: adjacent swaps, relocations, reversals (from A)
    start_polish = time.time()
    polish_budget = min(0.9, max(0.2, 0.04 * m))
    current = best_order[:]
    current_score = best_score

    def try_swap(cur, i, j):
        cand = cur[:]
        cand[i], cand[j] = cand[j], cand[i]
        return fitness(cand), cand

    def try_relocate_best_window(cur, idx):
        col = cur[idx]
        card = max(1, int(sample_nunique[col]))
        max_card = max(1, int(sample_nunique.max()))
        ratio = card / (max_card + 1e-8)
        window = max(4, min(max(6, m // 5), 4 + int((1 - ratio) * 5.0)))
        j_lo = max(0, idx - window)
        j_hi = min(m, idx + window + 1)
        best_local = -np.inf
        best_cand = None
        for j in range(j_lo, j_hi):
            if j == idx:
                continue
            cand = cur[:]
            val = cand.pop(idx)
            ins = j if j <= idx else j - 1
            cand.insert(ins, val)
            sc = fitness(cand)
            if sc > best_local:
                best_local, best_cand = sc, cand
        return best_local, best_cand

    def try_segment_reverse(cur, i, j):
        if i >= j:
            return -np.inf, cur
        cand = cur[:i] + list(reversed(cur[i:j+1])) + cur[j+1:]
        return fitness(cand), cand

    improved = True
    while improved and (time.time() - start_polish) < polish_budget:
        improved = False
        for i in range(m - 1):
            if (time.time() - start_polish) >= polish_budget:
                break
            sc, cand = try_swap(current, i, i + 1)
            if sc > current_score:
                current, current_score = cand, sc
                improved = True
                break
        if (time.time() - start_polish) >= polish_budget:
            break
        if not improved:
            best_local = current_score
            best_cand = None
            for i in range(m):
                if (time.time() - start_polish) >= polish_budget:
                    break
                sc, cand = try_relocate_best_window(current, i)
                if cand is not None and sc > best_local:
                    best_local, best_cand = sc, cand
            if best_cand is not None and best_local > current_score:
                current, current_score = best_cand, best_local
                improved = True
        if (time.time() - start_polish) >= polish_budget:
            break
        if not improved:
            limit_pairs = min(10, m * (m - 1) // 2)
            for _ in range(limit_pairs):
                i = random.randint(0, m - 2)
                j = random.randint(i + 1, min(m - 1, i + 5))
                sc, cand = try_segment_reverse(current, i, j)
                if sc > current_score:
                    current, current_score = cand, sc
                    improved = True
                    break
        if not improved and (time.time() - start_polish) < polish_budget * 0.85:
            tries = min(5, max(1, m // 5))
            for _ in range(tries):
                i, j = sorted(random.sample(range(m), 2))
                sc, cand = try_swap(current, i, j)
                if sc > current_score:
                    current, current_score = cand, sc
                    improved = True
                    break

    final_order = current
    final_cols = [cols[i] for i in final_order]
    df_sorted = df.sort_values(by=final_cols, kind="mergesort")
    return df_sorted.loc[:, final_cols]