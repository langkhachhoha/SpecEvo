#!/usr/bin/env python3
"""Fetch and materialise the LLM-SRBench **LSR-Synth** dataset for this repo.

LSR-Synth (Shojaee et al., ICML 2025 — arXiv:2504.10415) is the discovery-driven
half of LLM-SRBench: 129 synthetic equation-discovery problems across four
scientific domains, each combining well-known scientific terms with novel
synthetic ones so the target cannot be recited from memory.

    chem_react       36 problems   dA_dt = f(t, A)          reaction kinetics
    bio_pop_growth   24 problems   dP_dt = f(t, P)          population growth
    phys_osc         44 problems   dv_dt = f(x, t, v)       damped oscillators
    matsci           25 problems   sigma = f(epsilon, T)    stress / strain

Each problem ships three disjoint sample sets (paper App. A.2): 4000 in-domain
**train** points, 500 in-domain **id_test** points, and 500 held-out
**ood_test** points taken from beyond the training range (the last time points
for dynamical systems, the highest temperatures for stress-strain). Reporting
both ID and OOD is the point of the benchmark, so both are kept.

Provenance
----------
The canonical upload ``nnheui/llm-srbench`` is a *gated* HuggingFace dataset
(access must be granted per-account), which makes unattended server setup
impossible. We therefore read by default from the ungated mirror
``pkuHaowei/llm-srbench``, which republishes the same problems with the
train / id_test / ood_test splits already separated. The ground-truth
expressions and sample values were spot-checked against the paper's own
figures (Fig. 14 BPG0, Fig. 16 MatSci0, Fig. 17 PO0) and Table 4 before this
mirror was adopted. Set ``--repo nnheui/llm-srbench`` to use the official gated
copy instead if your HuggingFace account has been granted access.

Outputs (all under ``benchmarks/llm_srbench/data/``, gitignored)::

    problems.json                 manifest: per-domain variables + problem list
    <domain>/<problem_id>.npz     train / id_test / ood_test float64 arrays

Column 0 of every array is the target variable; columns 1.. are the inputs in
``input_vars`` order. This is the same layout the reference implementation uses
(``samples[:, 0]`` is the output, ``samples[:, 1:]`` the inputs).

Usage::

    python benchmarks/llm_srbench/prepare_data.py             # all domains
    python benchmarks/llm_srbench/prepare_data.py --limit 10  # first 10 each
    python benchmarks/llm_srbench/prepare_data.py --check     # verify only
    python benchmarks/llm_srbench/prepare_data.py --from-local /path/to/parquet
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"

MIRROR_REPO = "pkuHaowei/llm-srbench"
OFFICIAL_REPO = "nnheui/llm-srbench"

# Domain -> (HuggingFace split/config name, problem-id prefix used by the paper).
DOMAINS: dict[str, tuple[str, str]] = {
    "chem_react": ("lsr_synth_chem_react", "crk"),
    "bio_pop_growth": ("lsr_synth_bio_pop_growth", "bpg"),
    "phys_osc": ("lsr_synth_phys_osc", "po"),
    "matsci": ("lsr_synth_matsci", "matsci"),
}

# Expected problem counts (paper Sec. 2.2 / Table 4). phys_osc holds 44 rows in
# the released data although the paper text says 43; we assert the released
# count so a truncated download is caught.
EXPECTED_COUNTS = {
    "chem_react": 36,
    "bio_pop_growth": 24,
    "phys_osc": 44,
    "matsci": 25,
}

SPLITS = ("train", "id_test", "ood_test")

# Mirror column names -> our split names.
_MIRROR_COLS = {
    "train": ("train_input", "train_output"),
    "id_test": ("test_input", "test_output"),
    "ood_test": ("ood_input", "ood_output"),
}


# --------------------------------------------------------------------------- #
# Description parsing
# --------------------------------------------------------------------------- #
_OUT_RE = re.compile(r"^Output:\s*(\S+)\s*-\s*(.+)$")
_IN_RE = re.compile(r"^Input\s*\d+:\s*(\S+)\s*-\s*(.+)$")


def _parse_description(text: str) -> tuple[list[str], list[str]]:
    """Pull ``(symbols, symbol_descs)`` out of a mirror ``description`` blob.

    The blob looks like::

        Discover the mathematical equation relating the following variables:
        Output: dA_dt - Rate of change of concentration in chemistry ...
        Input 1: t - Time
        Input 2: A - Concentration at time t

    Returns output-first lists, matching the reference ``Equation.symbols`` /
    ``Equation.symbol_descs`` ordering.
    """
    symbols: list[str] = []
    descs: list[str] = []
    out: tuple[str, str] | None = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        m = _OUT_RE.match(line)
        if m:
            out = (m.group(1).strip(), m.group(2).strip())
            continue
        m = _IN_RE.match(line)
        if m:
            symbols.append(m.group(1).strip())
            descs.append(m.group(2).strip())
    if out is None:
        raise ValueError(f"could not find an 'Output:' line in description:\n{text!r}")
    return [out[0]] + symbols, [out[1]] + descs


# --------------------------------------------------------------------------- #
# Download / read
# --------------------------------------------------------------------------- #
def _snapshot(repo: str, local: Path | None) -> Path:
    if local is not None:
        path = Path(local).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"--from-local: not a directory: {path}")
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required to download LSR-Synth.\n"
            "  pip install huggingface_hub pyarrow\n"
            "or pass --from-local <dir> pointing at a directory that already "
            "holds the lsr_synth_*/ parquet files."
        ) from exc
    return Path(
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=["lsr_synth_*/*.parquet", "data/lsr_synth_*", "README.md"],
        )
    )


def _find_parquets(root: Path, split_name: str) -> list[Path]:
    """Locate the parquet shard(s) for one domain inside a snapshot dir."""
    hits = sorted(root.glob(f"{split_name}/*.parquet"))
    if hits:
        return hits
    # Official layout: data/lsr_synth_chem_react-00000-of-00001.parquet
    hits = sorted(root.glob(f"data/{split_name}-*.parquet"))
    if hits:
        return hits
    hits = sorted(root.glob(f"**/{split_name}*.parquet"))
    return hits


def _read_rows(paths: list[Path]) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyarrow is required to read the parquet files: pip install pyarrow") from exc
    rows: list[dict] = []
    for p in paths:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def _sort_key(instance_id: str, prefix: str) -> tuple[int, str]:
    """Order problems by their numeric suffix (crk0, crk1, ... crk10)."""
    m = re.search(rf"{re.escape(prefix)}(\d+)$", instance_id)
    return (int(m.group(1)) if m else 1 << 30, instance_id)


# --------------------------------------------------------------------------- #
# Materialise
# --------------------------------------------------------------------------- #
def _to_matrix(inputs: list[list[float]], outputs: list[list[float]], n_inputs: int):
    import numpy as np

    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(outputs, dtype=np.float64)
    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(f"expected a single output column, got shape {y.shape}")
        y = y[:, 0]
    if x.ndim != 2 or x.shape[1] != n_inputs:
        raise ValueError(f"expected inputs of shape (n, {n_inputs}), got {x.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"input/output row mismatch: {x.shape[0]} vs {y.shape[0]}")
    return np.column_stack([y, x])


@contextlib.contextmanager
def _manifest_lock():
    """Serialise writers so two domains prepared at once cannot lose an update."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = DATA_DIR / ".prepare.lock"
    with open(lock_path, "w") as fh:
        try:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass  # No flock (Windows, exotic filesystem) — the atomic write still holds.
        yield


def _write_manifest(manifest: dict) -> None:
    """Atomically replace problems.json so a concurrent reader never sees a torn file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "problems.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)


def _load_manifest() -> dict:
    path = DATA_DIR / "problems.json"
    if path.exists():
        try:
            man = json.loads(path.read_text())
            if isinstance(man.get("domains"), dict):
                return man
        except Exception:  # noqa: BLE001 — a corrupt manifest is simply rebuilt
            pass
    return {"source_repo": None, "splits": list(SPLITS), "domains": {}}


def prepare(
    *,
    repo: str = MIRROR_REPO,
    local: Path | None = None,
    limit: int | None = None,
    domains: list[str] | None = None,
    force: bool = False,
) -> dict:
    import numpy as np

    root = _snapshot(repo, local)
    wanted = domains or list(DOMAINS)
    # Merge into whatever is already on disk: preparing one domain must not
    # erase the others' entries, or a run of a second domain would find an
    # empty manifest.
    manifest = _load_manifest()
    manifest["source_repo"] = repo if local is None else f"local:{root}"
    manifest["splits"] = list(SPLITS)

    for domain in wanted:
        split_name, prefix = DOMAINS[domain]
        paths = _find_parquets(root, split_name)
        if not paths:
            raise SystemExit(f"no parquet found for {domain} ({split_name}) under {root}")
        rows = _read_rows(paths)
        if domain in EXPECTED_COUNTS and len(rows) != EXPECTED_COUNTS[domain]:
            raise SystemExit(
                f"{domain}: expected {EXPECTED_COUNTS[domain]} problems, "
                f"the download has {len(rows)} — refusing to write a partial dataset"
            )
        rows.sort(key=lambda r: _sort_key(r["instance_id"], prefix))

        out_dir = DATA_DIR / domain
        out_dir.mkdir(parents=True, exist_ok=True)

        # Variables are recorded per problem, not per domain. Most problems in a
        # domain share one signature, but LSR-Synth only hands the discoverer the
        # variables that actually appear in the target equation (the reference
        # loader keeps symbols whose properties contain "V"). phys_osc therefore
        # mixes (x, t, v), (x, t) and (t, v) problems.
        problems = []
        signatures: dict[tuple[str, ...], int] = {}
        for row in rows[: limit or len(rows)]:
            pid = row["instance_id"]
            short = pid.split("_")[-1]
            symbols, descs = _parse_description(row["description"])
            in_vars = list(row["input_vars"])
            out_vars = list(row["output_vars"])
            if symbols != out_vars + in_vars:
                raise SystemExit(
                    f"{pid}: description variables {symbols} disagree with "
                    f"input_vars/output_vars {out_vars + in_vars}"
                )

            arrays = {}
            for split, (icol, ocol) in _MIRROR_COLS.items():
                arrays[split] = _to_matrix(row[icol], row[ocol], len(in_vars))

            npz_path = out_dir / f"{short}.npz"
            if force or not npz_path.exists():
                np.savez_compressed(npz_path, **arrays)

            signatures[tuple(symbols)] = signatures.get(tuple(symbols), 0) + 1
            problems.append(
                {
                    "id": short,
                    "instance_id": pid,
                    "file": f"{domain}/{short}.npz",
                    "symbols": symbols,
                    "symbol_descs": descs,
                    "gt_expression": row["gt_expression"],
                    "n_train": int(arrays["train"].shape[0]),
                    "n_id_test": int(arrays["id_test"].shape[0]),
                    "n_ood_test": int(arrays["ood_test"].shape[0]),
                    "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest()[:16],
                }
            )

        manifest["domains"][domain] = {
            "hf_split": split_name,
            "prefix": prefix,
            "n_available": len(rows),
            "problems": problems,
        }
        shapes = ", ".join(f"({', '.join(s[1:])})x{n}" for s, n in signatures.items())
        print(f"  {domain:16s} {len(problems):3d} problems  inputs: {shapes}")

    _write_manifest(manifest)
    return manifest


def check(domains: list[str] | None = None, limit: int | None = None, quiet: bool = False) -> int:
    """Verify that the materialised data for the requested domains is usable."""
    import numpy as np

    man_path = DATA_DIR / "problems.json"
    if not man_path.exists():
        if not quiet:
            print(f"MISSING: {man_path} — run prepare_data.py first", file=sys.stderr)
        return 1
    try:
        man = json.loads(man_path.read_text())
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"UNREADABLE: {man_path}: {exc}", file=sys.stderr)
        return 1

    wanted = domains or list(DOMAINS)
    bad = 0
    for domain in wanted:
        info = man.get("domains", {}).get(domain)
        if info is None:
            if not quiet:
                print(f"MISSING: domain '{domain}' is not in {man_path}", file=sys.stderr)
            bad += 1
            continue
        if limit and len(info["problems"]) < limit:
            if not quiet:
                print(
                    f"INCOMPLETE: domain '{domain}' has {len(info['problems'])} problems, "
                    f"{limit} requested",
                    file=sys.stderr,
                )
            bad += 1
            continue
        for prob in info["problems"][: limit or len(info["problems"])]:
            path = DATA_DIR / prob["file"]
            if not path.exists():
                if not quiet:
                    print(f"MISSING: {path}", file=sys.stderr)
                bad += 1
                continue
            n_cols = len(prob["symbols"])
            try:
                with np.load(path) as z:
                    for split in SPLITS:
                        if split not in z:
                            if not quiet:
                                print(f"{path}: missing split '{split}'", file=sys.stderr)
                            bad += 1
                            continue
                        arr = z[split]
                        if arr.ndim != 2 or arr.shape[1] != n_cols:
                            if not quiet:
                                print(f"{path}[{split}]: bad shape {arr.shape}", file=sys.stderr)
                            bad += 1
                        if not np.all(np.isfinite(arr)):
                            if not quiet:
                                print(f"{path}[{split}]: non-finite values", file=sys.stderr)
                            bad += 1
            except Exception as exc:  # noqa: BLE001 — truncated / half-written file
                if not quiet:
                    print(f"UNREADABLE: {path}: {exc}", file=sys.stderr)
                bad += 1
        n = len(info["problems"]) if not limit else min(limit, len(info["problems"]))
        if not quiet:
            print(f"  {domain:16s} {n:3d} problems OK  ({info['n_available']} available)")
    if bad:
        if not quiet:
            print(f"FAILED: {bad} problem(s)", file=sys.stderr)
        return 1
    if not quiet:
        print("LSR-Synth data OK.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=MIRROR_REPO,
                    help=f"HuggingFace dataset repo (default: {MIRROR_REPO}; "
                         f"official gated copy: {OFFICIAL_REPO})")
    ap.add_argument("--from-local", default=None, metavar="DIR",
                    help="Read parquet from an already-downloaded directory instead of HuggingFace.")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Only materialise the first N problems per domain (default: all).")
    ap.add_argument("--domain", action="append", choices=list(DOMAINS), default=None,
                    help="Restrict to one domain (repeatable).")
    ap.add_argument("--force", action="store_true", help="Rewrite .npz files that already exist.")
    ap.add_argument("--check", action="store_true", help="Verify existing data and exit.")
    args = ap.parse_args()

    if args.check:
        return check(args.domain, args.limit)

    # Idempotent fast path: the run scripts call this before every run, and four
    # domains may be launched at once. If the requested data is already complete
    # there is nothing to download and — importantly — nothing to write, so
    # concurrent launches cannot race on problems.json.
    if not args.force and check(args.domain, args.limit, quiet=True) == 0:
        print(f"LSR-Synth data already prepared in {DATA_DIR} — nothing to do.")
        return check(args.domain, args.limit)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _manifest_lock():
        # Another process may have finished preparing while we waited for the lock.
        if not args.force and check(args.domain, args.limit, quiet=True) == 0:
            print(f"LSR-Synth data already prepared in {DATA_DIR} — nothing to do.")
            return check(args.domain, args.limit)
        print(f"Preparing LSR-Synth into {DATA_DIR}")
        prepare(
            repo=args.repo,
            local=Path(args.from_local) if args.from_local else None,
            limit=args.limit,
            domains=args.domain,
            force=args.force,
        )
    return check(args.domain, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
