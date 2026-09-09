def solve(**kwargs):
    """
    Solve the Maximum Independent Set problem for a given test case.

   Input:
        kwargs (dict): A dictionary with the following keys:
            - graph (networkx.Graph): The graph to solve

    Returns:
        dict: A solution dictionary containing:
            - mis_nodes (list): List of node indices in the maximum independent set
    """
    import random
    from collections import deque
    import heapq

    G = kwargs["graph"]
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {"mis_nodes": []}

    seed = kwargs.get("seed", None)
    rng = random.Random(seed)

    # Build adjacency sets and bitsets
    nbrs = {u: set(G.neighbors(u)) for u in nodes}
    idx = {u: i for i, u in enumerate(nodes)}
    full_mask = (1 << n) - 1
    adj_bits = [0] * n
    for u in nodes:
        i = idx[u]
        for v in nbrs[u]:
            j = idx[v]
            adj_bits[i] |= 1 << j

    # Greedy coloring for upper bound (in complement graph) - reused efficiently
    def compute_color_bound(P_mask):
        if P_mask == 0:
            return 0
        P = P_mask
        color = 0
        while P:
            color += 1
            best_deg = float('inf')
            best_v = -1
            v_bit = P
            while v_bit:
                bit = v_bit & -v_bit
                i = bit.bit_length() - 1
                deg = (adj_bits[i] & P).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = i
                v_bit &= v_bit - 1
            if best_v == -1:
                break
            P &= ~(1 << best_v)
            P &= ~(adj_bits[best_v] | (1 << best_v))
        return color

    # Heuristic: prioritize high-degree vertices to prune faster
    deg_order = sorted(nodes, key=lambda u: G.degree(u), reverse=True)

    best_set = set()
    best_size = 0

    # Stack for iterative DFS with bound pruning
    stack = [(full_mask, 0, 0)]  # (candidate_mask, current_set_mask, depth)
    visited = set()

    # Exact MIS on small induced subgraph with memoized recursion
    def exact_mis_on_subgraph(allowed_nodes):
        if not allowed_nodes:
            return set()
        idx_map = {v: i for i, v in enumerate(allowed_nodes)}
        adj_bits_sub = [0] * len(allowed_nodes)
        for i, v in enumerate(allowed_nodes):
            for u in nbrs[v]:
                j = idx_map.get(u)
                if j is not None:
                    adj_bits_sub[i] |= (1 << j)
        memo = {}

        def dp(S):
            if S == 0:
                return 0
            if S in memo:
                return memo[S]
            i = 0
            max_deg = -1
            temp = S
            while temp:
                bit = temp & -temp
                j = bit.bit_length() - 1
                d = (adj_bits_sub[j] & S).bit_count()
                if d > max_deg:
                    max_deg = d
                    i = j
                temp ^= bit
            excl = dp(S & ~(1 << i))
            incl = dp(S & ~(1 << i) & ~adj_bits_sub[i]) | (1 << i)
            res = incl if incl.bit_count() >= excl.bit_count() else excl
            memo[S] = res
            return res

        mask = dp((1 << len(allowed_nodes)) - 1)
        return {allowed_nodes[i] for i in range(len(allowed_nodes)) if (mask >> i) & 1}

    # Feedback-driven region selection
    feedback_centers = [v for v in nodes if G.degree(v) <= 3 or rng.random() < 0.3]
    rng.shuffle(feedback_centers)
    region_cap = min(32, max(8, n // 10))
    max_iters = max(10, min(200, 4 * (n // max(1, region_cap))))

    # Main loop: adaptive LNS with feedback
    for it in range(max_iters):
        center = feedback_centers[it % len(feedback_centers)] if feedback_centers else rng.choice(nodes)
        visited_region = set()
        queue = deque([center])
        visited_region.add(center)
        region = set()
        while queue and len(region) < region_cap:
            v = queue.popleft()
            region.add(v)
            candidates = [u for u in nbrs[v] if u not in visited_region]
            candidates.sort(key=lambda x: (G.degree(x), rng.random()))
            for u in candidates:
                if u not in visited_region:
                    visited_region.add(u)
                    queue.append(u)
                if len(region) >= region_cap:
                    break
        Rset = set(region)

        # Find allowed nodes: not adjacent to outside MIS
        forbidden = set()
        for u in best_set - Rset:
            forbidden |= (nbrs[u] & Rset)
        allowed = [v for v in Rset if v not in forbidden]

        # Solve exactly on allowed region
        new_in_R = exact_mis_on_subgraph(allowed)

        # Combine and improve
        cand_set = (best_set - Rset) | new_in_R
        # Greedy augmentation
        cand_set = set(cand_set)
        for v in sorted(nodes, key=lambda x: (G.degree(x), rng.random())):
            if v in cand_set or (nbrs[v] & cand_set):
                continue
            cand_set.add(v)

        if len(cand_set) > best_size:
            best_size = len(cand_set)
            best_set = set(cand_set)
            # Expand feedback centers
            for v in cand_set - best_set:
                if v not in feedback_centers and G.degree(v) > 1:
                    feedback_centers.append(v)

        # Periodic perturbation
        if (it + 1) % 25 == 0 and rng.random() < 0.3:
            to_drop = rng.randint(1, min(2, len(best_set) // 5))
            drops = rng.sample(list(best_set), to_drop)
            best_set = set(best_set - set(drops))

    # Final branch-and-bound with pruning — key change: eliminate heap, use direct scan
    stack = [(full_mask, 0, 0)]
    while stack:
        P_mask, S_mask, depth = stack.pop()
        size = S_mask.bit_count()

        # Compute upper bound via greedy coloring in complement graph
        if P_mask == 0:
            upper_bound = 0
        else:
            P = P_mask
            color = 0
            while P:
                color += 1
                best_deg = float('inf')
                best_v = -1
                v_bit = P
                while v_bit:
                    bit = v_bit & -v_bit
                    i = bit.bit_length() - 1
                    deg = (adj_bits[i] & P).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = i
                    v_bit &= v_bit - 1
                if best_v == -1:
                    break
                P &= ~(1 << best_v)
                P &= ~(adj_bits[best_v] | (1 << best_v))
            upper_bound = size + color

        # Prune if current path cannot beat best solution
        if size + upper_bound <= best_size:
            continue

        # Check if this is a complete partial solution
        if P_mask == 0:
            if size > best_size:
                best_size = size
                best_set = {nodes[i] for i in range(n) if (S_mask >> i) & 1}
            continue

        # Use direct O(n) scan to find min-degree vertex (replaces heap)
        best_deg = float('inf')
        best_v = -1
        v_bit = P_mask
        while v_bit:
            bit = v_bit & -v_bit
            i = bit.bit_length() - 1
            deg = (adj_bits[i] & P_mask).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = i
            v_bit &= v_bit - 1

        if best_v == -1:
            continue

        # Branch: include best_v
        v_bit = 1 << best_v
        new_P = P_mask & ~(adj_bits[best_v] | v_bit)
        new_S = S_mask | v_bit
        stack.append((new_P, new_S, depth + 1))

        # Apply local search immediately after including a vertex
        I = {nodes[i] for i in range(n) if (new_S >> i) & 1}
        I = I.copy()
        for u in nodes:
            if u in I:
                continue
            if not any(v in I for v in nbrs[u]):
                I.add(u)
        if len(I) > best_size:
            best_size = len(I)
            best_set = I

        # Branch: exclude best_v — only if it might improve
        new_P_excl = P_mask & ~v_bit
        # Compute upper bound for exclude branch
        P = new_P_excl
        color_excl = 0
        while P:
            color_excl += 1
            best_deg = float('inf')
            best_v = -1
            v_bit = P
            while v_bit:
                bit = v_bit & -v_bit
                i = bit.bit_length() - 1
                deg = (adj_bits[i] & P).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = i
                v_bit &= v_bit - 1
            if best_v == -1:
                break
            P &= ~(1 << best_v)
            P &= ~(adj_bits[best_v] | (1 << best_v))
        upper_bound_excl = size + color_excl
        if upper_bound_excl > best_size:
            stack.append((new_P_excl, S_mask, depth + 1))

    return {"mis_nodes": sorted(best_set)}