"""Offline regression checks for public launch and reproduction commands."""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from specevo_baselines.cli import _run_with_time_budget, parse_args

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("runner_name", ["run_specevo", "run_levi"])
def test_task_runner_reaches_engine_with_seed_and_lazy_inputs(runner_name, monkeypatch, tmp_path):
    import specevo

    module = load_script(runner_name)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-for-offline-test")
    problem = SimpleNamespace(
        PROBLEM_DESCRIPTION="test",
        FUNCTION_SIGNATURE="def solve(x):",
        score_fn=lambda fn, inputs: {"score": 1},
        INPUTS=None,
        get_lazy_inputs=lambda: [1, 2],
    )
    monkeypatch.setitem(sys.modules, "release_test_problem", problem)
    out = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            runner_name,
            "--task-dir",
            str(tmp_path),
            "--problem-module",
            "release_test_problem",
            "--output-dir",
            str(out),
            "--seed",
            "17",
            "--evals",
            "1",
        ],
    )
    if runner_name == "run_levi":
        sys.argv.append("--prompt-bank")
    result = SimpleNamespace(
        best_score=1.0,
        best_metrics={},
        best_program="def solve(x): return x",
        total_evaluations=1,
        total_cost=0,
        archive_size=1,
        runtime_seconds=0,
        total_llm_calls=0,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        init_usage={},
        navigator_trials=[],
    )
    engine_name = "run_specevo" if runner_name == "run_specevo" else "evolve_code"
    engine = Mock(return_value=result)
    monkeypatch.setattr(specevo, engine_name, engine)
    assert module.main() == 0
    assert engine.call_args.kwargs["inputs"] == [1, 2]
    if runner_name == "run_specevo":
        assert engine.call_args.kwargs["seed"] == 17
    assert json.loads((out / "summary.json").read_text())["seed"] == 17


@pytest.mark.parametrize("method", ["specevo", "levi", "topk", "relayevolve"])
def test_reproduction_forwards_controls_and_honors_python(method, tmp_path):
    wrapper = tmp_path / "fake-python"
    captured = tmp_path / "args.json"
    wrapper.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "open(os.environ['CAPTURE_ARGS'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    wrapper.chmod(0o755)
    env = {
        **os.environ,
        "PY": str(wrapper),
        "METHOD": method,
        "EVALS": "11",
        "DOLLARS": "0.7",
        "SECONDS_CAP": "21",
        "WORKERS": "3",
        "SEED": "17",
        "OPENROUTER_API_KEY": "fake",
        "OUT": str(tmp_path / "outputs"),
        "CAPTURE_ARGS": str(captured),
    }
    env.pop("PY_OVERRIDE", None)
    subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/reproduce/_common.sh; "
            "dispatch tasks/smoke_demo benchmarks/math/circle_packing smoke",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = json.loads(captured.read_text())
    for flag, value in [
        ("--seed", "17"),
        ("--workers", "3"),
        ("--seconds", "21"),
        ("--dollars", "0.7"),
    ]:
        assert args[args.index(flag) + 1] == value
    if method == "relayevolve":
        assert args[args.index("-m", 2) + 1] == "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
        assert args[args.index("--guide-model") + 1] == "openrouter/openai/gpt-5"


@pytest.mark.parametrize(
    "flag,value",
    [("--seconds", "0"), ("--seconds", "nan"), ("--workers", "0"), ("--workers", "-2")],
)
def test_invalid_baseline_controls_fail_before_running(flag, value, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cli", "evaluator.py", flag, value])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_baseline_time_budget_requests_graceful_shutdown():
    class Runner:
        def __init__(self):
            self.stopped = asyncio.Event()
            self.discovery_controller = SimpleNamespace(request_shutdown=self.stopped.set)

        async def run(self, **kwargs):
            await self.stopped.wait()
            return "saved result"

    async def check():
        runner = Runner()
        result = await asyncio.wait_for(
            _run_with_time_budget(runner, iterations=10, checkpoint_path=None, seconds=0.01),
            timeout=1,
        )
        assert result == "saved result"
        assert runner.stopped.is_set()

    asyncio.run(check())


def test_smoke_demo_reference_matches_all_grading_answers():
    spec = importlib.util.spec_from_file_location(
        "smoke_problem", ROOT / "tasks/smoke_demo/problem.py"
    )
    problem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(problem)
    for coins, target, expected in problem._TEST_CASES:
        assert problem._reference(coins, target) == expected
    assert problem.score_fn(problem._reference)["correct_fraction"] == 1.0


def test_lsr_suite_finalizes_each_saved_run(tmp_path):
    # A minimal copied suite keeps this test independent of downloaded LSR data.
    import shutil

    root = tmp_path / "repo"
    scripts = root / "scripts" / "reproduce"
    scripts.mkdir(parents=True)
    for name in ["_common.sh", "lsr_synth.sh"]:
        shutil.copyfile(ROOT / "scripts" / "reproduce" / name, scripts / name)
    (root / "tasks/llm_srbench/chem_react/crk0").mkdir(parents=True)
    captured = tmp_path / "calls.jsonl"
    wrapper = tmp_path / "python"
    wrapper.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['CAPTURE_ARGS'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    wrapper.chmod(0o755)
    env = {
        **os.environ,
        "PY": str(wrapper),
        "METHOD": "specevo",
        "SEED": "9",
        "DOMAIN": "chem_react",
        "OPENROUTER_API_KEY": "fake",
        "OUT": str(tmp_path / "out"),
        "CAPTURE_ARGS": str(captured),
    }
    env.pop("PY_OVERRIDE", None)
    subprocess.run(
        ["bash", str(scripts / "lsr_synth.sh")], env=env, check=True, capture_output=True, text=True
    )
    run, finalize = [json.loads(line) for line in captured.read_text().splitlines()]
    assert run[0] == "scripts/run_specevo.py"
    assert finalize[0] == "scripts/lsr_finalize.py"
    assert "--allow-failed-program" in finalize
    for flag, value in [
        ("--domain", "chem_react"),
        ("--problem", "crk0"),
        ("--method", "specevo"),
        ("--seed", "9"),
    ]:
        assert finalize[finalize.index(flag) + 1] == value
    assert finalize[finalize.index("--run-dir") + 1] == run[run.index("--output-dir") + 1]
    assert finalize[finalize.index("--results") + 1].endswith("/specevo/seed9/results.jsonl")


def test_baseline_cli_applies_worker_seed_and_guide_model(monkeypatch, tmp_path):
    from specevo_baselines import cli

    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("def evaluate(path): return {'combined_score': 1.0}\n")
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.config = kwargs["config"]
            self.output_dir = kwargs["output_dir"]
            self.discovery_controller = None

        async def run(self, **kwargs):
            return None

    monkeypatch.setattr(cli, "Runner", FakeRunner)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli",
            str(evaluator),
            "-s",
            "relayevolve",
            "-m",
            "openrouter/cheap",
            "--guide-model",
            "openrouter/strong",
            "--workers",
            "3",
            "--seed",
            "0",
            "--seconds",
            "10",
            "-o",
            str(tmp_path),
        ],
    )
    assert cli.main() == 0
    config = captured["config"]
    assert config.max_parallel_iterations == 3
    assert config.random_seed == 0
    assert config.search.database.random_seed == 0
    assert config.llm.models[0].name == "cheap"
    assert config.llm.guide_models[0].name == "strong"
    saved = json.loads((tmp_path / "run_settings.json").read_text())
    assert saved["seed"] == 0
    assert saved["seconds"] == 10
    assert "api_key" not in saved


@pytest.mark.parametrize("allow_failed", [False, True])
def test_lsr_finalizer_records_failed_programs_without_aborting_opted_in_sweeps(
    monkeypatch, tmp_path, allow_failed
):
    module = load_script("lsr_finalize")
    results = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize",
            "--domain",
            "test",
            "--problem",
            "test",
            "--method",
            "specevo",
            "--run-dir",
            str(tmp_path),
            "--results",
            str(results),
        ]
        + (["--allow-failed-program"] if allow_failed else []),
    )
    monkeypatch.setattr(
        module,
        "build_record",
        lambda **kwargs: ({"status": "no_program", "error": "missing"}, False),
    )
    assert module.main() == (0 if allow_failed else 1)
    assert json.loads(results.read_text())["status"] == "no_program"
