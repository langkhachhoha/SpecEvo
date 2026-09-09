"""Tests for EvaluatorConfig defaults."""

from specevo_baselines.config import EvaluatorConfig


class TestEvaluatorConfigDefaults:
    def test_default_timeout(self):
        assert EvaluatorConfig().timeout == 360

    def test_default_max_retries(self):
        assert EvaluatorConfig().max_retries == 3
