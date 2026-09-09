import heapq
from collections import defaultdict

def solve(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    costs = kwargs["costs"]
    row_cover = kwargs["row_cover"]

    if m == 0:
        return {"selected_columns": []}

    # Build column-to-row mapping (1-indexed)
    col_rows = [set() for _ in range(n + 1)]
    for r in range(m):
        for c in row_cover[r]:
            if 1 <= c <= n:
                col_rows[c].add(r)
    
    # Early feasibility check
    for r in range(m):
        if not row_cover[r]:
            return {"selected_columns": []}

    # Convert to bitsets for efficient operations
    row_mask = [0] * m
    for r in range(m):
        for c in row_cover[r]:
            row_mask[r] |= (1 << (c - 1))
    
    # Precompute column coverage bitmasks
    col_mask = [0] * (n + 1)
    for c in range(1, n + 1):
        for r in col_rows[c]:
            col_mask[c] |= (1 << r)
    
    # Initial uncovered mask
    uncovered = (1 << m) - 1
    
    # Priority queue: (lower_bound, cost, selected, uncovered_mask)
    heap = [(0, 0, [], uncovered)]
    best_cost = float('inf')
    best_sel = None

    # Dual lower bound via adaptive scaling
    def dual_lower_bound(uncovered_mask, allowed_cols):
        if uncovered_mask == 0:
            return 0.0
        duals = [0.0] * m
        temp_allowed = set(allowed_cols)
        temp_uncovered = set()
        for r in range(m):
            if (uncovered_mask >> r) & 1:
                temp_uncovered.add(r)
        
        while temp_uncovered and temp_allowed:
            best_row = -1
            best_ratio = float('inf')
            for r in temp_uncovered:
                candidates = [c for c in row_cover[r] if 1 <= c <= n and c in temp_allowed]
                if not candidates:
                    continue
                min_cost = min(costs[c - 1] for c in candidates)
                coverage = len(candidates)
                ratio = min_cost / coverage if coverage > 0 else float('inf')
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_row = r
            if best_row == -1:
                break
            min_cost = min(costs[c - 1] for c in row_cover[best_row] if 1 <= c <= n and c in temp_allowed)
            scaling = 0.8 * 0.9 ** (len(temp_uncovered) // 2)
            dual_value = min_cost * scaling
            duals[best_row] = dual_value
            for c in row_cover[best_row]:
                if 1 <= c <= n and c in temp_allowed:
                    costs[c - 1] -= dual_value
                    if costs[c - 1] <= 0:
                        temp_allowed.discard(c)
            temp_uncovered.remove(best_row)
        
        return sum(duals)

    while heap:
        lb, cost, selected, uncovered_mask = heapq.heappop(heap)
        if cost >= best_cost:
            continue
        if uncovered_mask == 0:
            if cost < best_cost:
                best_cost = cost
                best_sel = selected[:]
            continue

        # Greedy selection: pick column with best marginal cost per uncovered row
        best_marginal_cost = float('inf')
        best_col = None
        best_covered = 0
        for c in range(1, n + 1):
            if c in selected or (col_mask[c] & uncovered_mask) == 0:
                continue
            covered = col_mask[c] & uncovered_mask
            rows_covered = bin(covered).count('1')
            marginal_cost = costs[c - 1] / rows_covered
            if marginal_cost < best_marginal_cost:
                best_marginal_cost = marginal_cost
                best_col = c
                best_covered = covered

        if best_col is None:
            continue

        # Branch 1: include best_col
        new_selected = selected + [best_col]
        new_cost = cost + costs[best_col - 1]
        new_uncovered = uncovered_mask & ~best_covered
        new_lb = dual_lower_bound(new_uncovered, [c for c in range(1, n + 1) if c != best_col and c not in selected])
        if new_lb + new_cost < best_cost:
            heapq.heappush(heap, (new_lb, new_cost, new_selected, new_uncovered))

        # Greedy completion to avoid deep search
        completed = []
        work_uncovered = new_uncovered
        work_allowed = set(c for c in range(1, n + 1) if c not in selected + [best_col])
        while work_uncovered and work_allowed:
            best_ratio = float('-inf')
            best_c = None
            best_gain = 0
            for c in work_allowed:
                covered_by_c = [r for r in col_rows[c] if (work_uncovered >> r) & 1]
                gain = len(covered_by_c)
                if gain == 0:
                    continue
                ratio = gain / costs[c - 1]
                if ratio > best_ratio or (ratio == best_ratio and gain > best_gain):
                    best_ratio = ratio
                    best_c = c
                    best_gain = gain
            if best_c is None:
                break
            completed.append(best_c)
            for r in col_rows[best_c]:
                if (work_uncovered >> r) & 1:
                    work_uncovered &= ~(1 << r)
            work_allowed.discard(best_c)
        if not work_uncovered:
            final_selected = new_selected + completed
            final_cost = sum(costs[c - 1] for c in final_selected)
            if final_cost < best_cost:
                best_cost = final_cost
                best_sel = final_selected

        # Branch 2: exclude best_col (pruned if feasible)
        work_allowed = set(c for c in range(1, n + 1) if c != best_col and c not in selected)
        feasible = True
        for r in range(m):
            if (uncovered_mask >> r) & 1:
                if not any(c in work_allowed for c in row_cover[r]):
                    feasible = False
                    break
        if feasible:
            new_lb2 = dual_lower_bound(uncovered_mask, work_allowed)
            if new_lb2 + cost < best_cost:
                heapq.heappush(heap, (new_lb2, cost, selected, uncovered_mask))

    # Final repair: remove redundant columns
    if best_sel is None:
        return {"selected_columns": []}
    
    selected_set = set(best_sel)
    cover_count = [0] * m
    for c in selected_set:
        for r in col_rows[c]:
            cover_count[r] += 1

    removed = True
    while removed:
        removed = False
        candidates = sorted(selected_set, key=lambda c: costs[c - 1] / sum(1 for r in range(m) if (col_mask[c] >> r) & 1), reverse=True)
        for c in candidates:
            if c not in selected_set:
                continue
            can_remove = True
            for r in col_rows[c]:
                if cover_count[r] <= 1:
                    can_remove = False
                    break
            if can_remove:
                selected_set.remove(c)
                for r in col_rows[c]:
                    cover_count[r] -= 1
                removed = True

    # Verify coverage
    covered = [False] * m
    for c in selected_set:
        for r in col_rows[c]:
            covered[r] = True
    if not all(covered):
        return {"selected_columns": []}

    return {"selected_columns": sorted(selected_set)}