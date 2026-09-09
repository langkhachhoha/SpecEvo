"""Tests for ``prompt.exclude_metrics`` — keeping a metric out of the prompt.

The knob exists because the default prompt renders the parent's metric dict up
to nine times per call (header, one per context program, one per previous
attempt, one on the current program). A metric that is a paragraph of prose
rather than a number — LLM-SRBench's ``feedback``, which restates the NMSE/R2
floats sitting next to it — is therefore paid for nine times over.

Two things are pinned here: the metric really does leave every section of the
prompt, and a config that does not set the knob renders exactly what it
rendered before it existed (every already-measured benchmark must be
untouched).
"""

from specevo_baselines.config import Config, ContextBuilderConfig
from specevo_baselines.context_builder.default import DefaultContextBuilder
from specevo_baselines.search.base_database import Program

FEEDBACK = "train: NMSE=1e-05 R2=0.99999 — a paragraph restating the floats"

METRICS = {
    "combined_score": 3.5,
    "train_nmse": 1e-05,
    "id_nmse": 2e-05,
    "feedback": FEEDBACK,
}


def _program(pid: str) -> Program:
    return Program(
        id=pid,
        solution="def equation(x, params):\n    return params[0] * x\n",
        language="python",
        metrics=dict(METRICS),
        metadata={"changes": "scaled the linear term", "parent_metrics": dict(METRICS)},
    )


def _render(config: Config) -> str:
    """Build a prompt exercising every section that renders a metric dict."""
    prompt = DefaultContextBuilder(config).build_prompt(
        current_program={"": _program("parent")},
        context={
            "program_metrics": dict(METRICS),
            "other_context_programs": {"": [_program("ctx1"), _program("ctx2")]},
            "previous_programs": [_program("prev1"), _program("prev2")],
        },
    )
    return prompt["system"] + prompt["user"]


def _config(exclude: list[str] | None = None) -> Config:
    config = Config()
    config.diff_based_generation = False
    if exclude is not None:
        config.context_builder = ContextBuilderConfig(exclude_metrics=exclude)
    return config


class TestExcludeMetrics:
    def test_default_config_keeps_every_metric(self):
        """No knob set: the prompt is what it has always been."""
        assert _render(_config()).count(FEEDBACK) > 0

    def test_excluded_metric_leaves_every_section(self):
        assert FEEDBACK not in _render(_config(["feedback"]))

    def test_other_metrics_survive_the_exclusion(self):
        rendered = _render(_config(["feedback"]))
        assert "train_nmse" in rendered
        assert "id_nmse" in rendered

    def test_exclusion_shrinks_the_prompt(self):
        assert len(_render(_config(["feedback"]))) < len(_render(_config()))

    def test_default_is_empty_so_existing_benchmarks_are_untouched(self):
        assert ContextBuilderConfig().exclude_metrics == []
        assert _render(_config()) == _render(_config([]))
