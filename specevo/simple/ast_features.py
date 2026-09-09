"""14-dimensional AST count features for the SpecEvo behavior signature.

This is a direct port of the framework's hand-crafted feature set (see
``specevo.behavior.features``). The bigram histogram from earlier SpecEvo
versions is gone — it was a finer-grained signal than the search
actually needed, and its high dimension (1600) confused downstream
clustering once the embedding half was added. 14 counts, log1p-damped,
gives us a stable, low-noise structural fingerprint that pairs
cleanly with the PCA-reduced description embedding.

Layout (kept stable; the archive stores the vectors and may compare
across re-cluster events):

    0  ast_depth                 max depth of the AST
    1  cyclomatic_complexity     McCabe (1 + branches + bool-op fanout)
    2  loop_count                For / While nodes
    3  loop_nesting_max          deepest nested loop
    4  branch_count              If nodes
    5  function_def_count        FunctionDef nodes
    6  comprehension_count       ListComp / DictComp / SetComp / GeneratorExp
    7  call_count                Call nodes
    8  comparison_count          Compare nodes
    9  subscript_count           Subscript nodes
    10 numeric_literal_count     int / float constants
    11 math_op_count             BinOp + UnaryOp nodes
    12 import_count              Import + ImportFrom nodes
    13 code_length               len(source) — coarse size signal
"""

from __future__ import annotations

import ast
import logging

import numpy as np

logger = logging.getLogger(__name__)

N_AST_FEATURES = 14


def compute_ast_features(code: str) -> np.ndarray:
    """Return the 14-d log1p-damped count vector for *code*.

    Returns a zero vector if *code* cannot be parsed — callers treat that
    as 'no structural information' rather than crashing the worker.
    """
    if not code:
        return np.zeros(N_AST_FEATURES, dtype=np.float32)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.debug("compute_ast_features: parse failed; returning zero vector")
        return np.zeros(N_AST_FEATURES, dtype=np.float32)

    ast_depth = _ast_depth(tree)
    cyclomatic = 1
    loop_count = 0
    branch_count = 0
    fn_def_count = 0
    comp_count = 0
    call_count = 0
    cmp_count = 0
    sub_count = 0
    num_count = 0
    math_op_count = 0
    import_count = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            cyclomatic += 1
        elif isinstance(node, ast.BoolOp):
            cyclomatic += len(node.values) - 1
        if isinstance(node, (ast.For, ast.While)):
            loop_count += 1
        if isinstance(node, ast.If):
            branch_count += 1
        if isinstance(node, ast.FunctionDef):
            fn_def_count += 1
        if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            comp_count += 1
        if isinstance(node, ast.Call):
            call_count += 1
        if isinstance(node, ast.Compare):
            cmp_count += 1
        if isinstance(node, ast.Subscript):
            sub_count += 1
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            num_count += 1
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            math_op_count += 1
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_count += 1

    loop_nesting = _max_loop_nesting(tree)

    raw = np.array(
        [
            ast_depth, cyclomatic, loop_count, loop_nesting, branch_count,
            fn_def_count, comp_count, call_count, cmp_count, sub_count,
            num_count, math_op_count, import_count, len(code),
        ],
        dtype=np.float32,
    )
    return np.log1p(raw)


def _ast_depth(tree: ast.AST) -> int:
    def _walk(node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(_walk(c) for c in children)

    return _walk(tree)


def _max_loop_nesting(tree: ast.AST) -> int:
    def _walk(node: ast.AST, current: int) -> int:
        best = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                best = max(best, _walk(child, current + 1))
            else:
                best = max(best, _walk(child, current))
        return best

    return _walk(tree, 0)
