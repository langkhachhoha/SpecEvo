# Reproduction scripts

Each script runs one benchmark suite end to end. They are thin loops over
`scripts/run_specevo.py` (SpecEvo) and `scripts/run_levi.py` / the
`specevo-baselines-run` CLI (baselines) — read them before launching, they
spend real API credit.

| Script | Suite | Tasks |
|---|---|---|
| `math.sh`      | Mathematical discovery | 7 |
| `adrs.sh`      | Systems optimization (ADRS) | 4 |
| `co_bench.sh`  | CO-Bench combinatorial optimization | 36 |
| `lsr_synth.sh` | LSR-Synth equation discovery | 129 |

Every script honours the same environment overrides:

```bash
METHOD=specevo        # specevo | levi | openevolve_native | gepa_native | adaevolve | evox | relayevolve | ...
SPECULATOR_MODEL=...  # lightweight model  (default: qwen3-30b-a3b-instruct-2507)
NAVIGATOR_MODEL=...   # frontier model     (default: gpt-5)
EVALS=500             # evaluation budget
DOLLARS=10            # API-cost budget in USD
SECONDS_CAP=10800     # wall-clock budget (3h)
WORKERS=4             # concurrent Speculators
OUT=outputs/repro     # output root
```
