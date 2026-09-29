"""Verify release wheels work without importing packages from the checkout."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from zipfile import ZipFile


def test_wheel_contains_both_packages_and_runtime_assets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "setup.py", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(root / name, source / name)
    shutil.copytree(root / "licenses", source / "licenses")
    for name in ("specevo", "specevo_baselines"):
        shutil.copytree(
            root / name, source / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    wheel_dir = tmp_path / "wheels"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(wheel_dir), str(source)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    wheel, = wheel_dir.glob("*.whl")
    installed = tmp_path / "installed"
    with ZipFile(wheel) as package:
        members = package.namelist()
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/GEPA.txt"):
            assert any(member.endswith("/" + name) for member in members), name
        package.extractall(installed)

    # -I removes the checkout and PYTHONPATH from module resolution. Dependencies
    # remain available in the test environment; CI also tests a fresh wheel venv.
    probe = textwrap.dedent("""\
        import sys
        from importlib.resources import files
        from pathlib import Path

        installed = Path(sys.argv[1])
        sys.path.insert(0, str(installed))
        import specevo
        import specevo_baselines
        from specevo_baselines.config import Config
        from specevo_baselines.context_builder.default.builder import DefaultContextBuilder
        from specevo_baselines.llm.agentic_generator import TOOL_SCHEMAS

        for module in (specevo, specevo_baselines):
            assert Path(module.__file__).is_relative_to(installed), module.__file__
        assert callable(specevo.run_specevo)
        assert callable(specevo_baselines.run_discovery)
        assert TOOL_SCHEMAS
        DefaultContextBuilder(Config())
        package = files("specevo_baselines")
        for name in (
            "evaluation/evaluate.sh",
            "extras/monitor/dashboard.html",
            "extras/external/defaults/openevolve_default.yaml",
            "search/evox/config/search.yaml",
        ):
            assert package.joinpath(name).read_text().strip(), name
    """)
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    subprocess.run(
        [sys.executable, "-I", "-c", probe, str(installed)],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True, timeout=60,
    )
