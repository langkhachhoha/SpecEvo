"""Cheap/strong routing baselines, all on the same OpenEvolve backend.

Each of these shares RelayEvolve's evaluator, prompt template, population
mechanics (island MAP-Elites), parallel loop, generation cap and dollar cap —
the *only* difference is which model a given generation calls, which is what
makes the comparison in the paper's Table 1 an apples-to-apples one.

============== ================================================================
All-cheap      Every generation uses the cheap model.
All-strong     Every generation uses the strong model.
Fixed-switch   Cheap for a fixed prefix of the budget, strong afterwards
               (Bhan et al., 2026).  The prefix is measured in dollars when a
               budget is set, otherwise in generations.
Random         An independent coin flip per generation.
Bandit         Two-armed UCB over {cheap, strong}, rewarded by the realized
               improvement in best-so-far fitness (ShinkaEvolve's selector).
============== ================================================================
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Dict, List, Optional, Union

from specevo_baselines.search.base_database import Program
from specevo_baselines.search.default_discovery_controller import DiscoveryControllerInput
from specevo_baselines.search.relay.scheduler import TwoArmedBandit
from specevo_baselines.search.relay.tiered import CHEAP, STRONG, TieredController
from specevo_baselines.search.utils.discovery_utils import SerializableResult
from specevo_baselines.utils.metrics import get_score

logger = logging.getLogger(__name__)


class RouterController(TieredController):
    """Base class: run the parallel loop, deciding a tier per generation."""

    method_name = "router"

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        opts = self.config.search.database
        self.rng = random.Random(getattr(opts, "random_seed", None) or 0)
        self.phase = "route"

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
    ) -> Optional[Union[Program, SerializableResult]]:
        self.requested_iterations = max_iterations
        best = self.database.get_best_program()
        if best is not None:
            self._track_best(best)
        self.mark_phase("start", start_iteration)

        await self._route(start_iteration, max_iterations, self.retry_times, checkpoint_callback)

        self.mark_phase("end", start_iteration + self.iterations_used)
        self.write_summary(self._extra_summary())
        return self._finalize_discovery()

    async def _route(
        self,
        start_iteration: int,
        max_iterations: int,
        retry_times: int,
        checkpoint_callback: Optional[Callable[[int], None]],
    ) -> None:
        raise NotImplementedError

    def _extra_summary(self) -> Dict:
        return {}


class _SingleTierController(RouterController):
    """All-cheap / All-strong: one tier, one fully pipelined block."""

    tier = CHEAP

    async def _route(self, start_iteration, max_iterations, retry_times, checkpoint_callback):
        self.phase = self.tier
        await self.run_block(
            start_iteration,
            max_iterations,
            self.tier,
            retry_times=retry_times,
            checkpoint_callback=checkpoint_callback,
        )


class AllCheapController(_SingleTierController):
    method_name = "all_cheap"
    tier = CHEAP


class AllStrongController(_SingleTierController):
    method_name = "all_strong"
    tier = STRONG


class FixedSwitchController(RouterController):
    """Cheap exploration prefix, then strong for the remainder.

    The prefix ends at ``switch_fraction`` of the *generation* budget or of the
    *dollar* budget, whichever comes first. Both bounds are needed: the cheap
    model is roughly 14x cheaper per call, so half the dollars can be more
    generations than the run is allowed — a dollars-only rule would never fire
    and this baseline would silently degenerate into All-cheap.
    """

    method_name = "fixed_switch"

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        self.switch_fraction = float(getattr(self.config.search.database, "switch_fraction", 0.5))
        self._switch_reason = "generation_cap"

    async def _route(self, start_iteration, max_iterations, retry_times, checkpoint_callback):
        self.phase = CHEAP
        cursor = start_iteration
        end = start_iteration + max_iterations
        switch_budget = (
            self.total_budget_usd * self.switch_fraction
            if self.total_budget_usd is not None
            else None
        )
        switch_iteration = start_iteration + int(round(max_iterations * self.switch_fraction))

        while cursor < end and not self.stop_requested():
            if cursor >= switch_iteration:
                self._switch_reason = "generation_fraction"
                break
            if switch_budget is not None and self.spent_usd() >= switch_budget:
                self._switch_reason = "budget_fraction"
                break
            count = min(self.max_parallel, end - cursor)
            await self.run_block(
                cursor,
                count,
                CHEAP,
                retry_times=retry_times,
                checkpoint_callback=checkpoint_callback,
            )
            cursor += count

        self.handoff_iteration = cursor
        self.mark_phase("switch", cursor)
        logger.info(
            "Fixed-switch: handing over to the strong model at generation %d "
            "($%.4f spent, trigger: %s)",
            cursor,
            self.spent_usd(),
            self._switch_reason,
        )

        if cursor < end and not self.stop_requested():
            self.phase = STRONG
            await self.run_block(
                cursor,
                end - cursor,
                STRONG,
                retry_times=retry_times,
                checkpoint_callback=checkpoint_callback,
            )

    def _extra_summary(self) -> Dict:
        return {
            "switch_fraction": self.switch_fraction,
            "switch_reason": self._switch_reason,
        }


class RandomRouteController(RouterController):
    """Independent per-generation coin flip between the two models."""

    method_name = "random"

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        self.p_strong = float(getattr(self.config.search.database, "p_strong", 0.5))
        self._choices: Dict[int, str] = {}

    def _tier_of(self, iteration: int) -> str:
        tier = self._choices.get(iteration)
        if tier is None:
            tier = STRONG if self.rng.random() < self.p_strong else CHEAP
            self._choices[iteration] = tier
        return tier

    async def _route(self, start_iteration, max_iterations, retry_times, checkpoint_callback):
        await self.run_block(
            start_iteration,
            max_iterations,
            CHEAP,
            retry_times=retry_times,
            checkpoint_callback=checkpoint_callback,
            tier_of=self._tier_of,
        )

    def _extra_summary(self) -> Dict:
        counts = {CHEAP: 0, STRONG: 0}
        for tier in self._choices.values():
            counts[tier] = counts.get(tier, 0) + 1
        return {"p_strong": self.p_strong, "tier_draws": counts}


class BanditRouteController(RouterController):
    """Two-armed UCB over the models, rewarded by best-so-far improvement.

    Arms are pulled in batches of ``max_parallel_iterations`` so the loop stays
    pipelined: every generation in a batch draws its arm from the statistics as
    of the batch start, and each resulting program is credited with the
    improvement it made over the best score at that same point.
    """

    method_name = "bandit"

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        self.bandit = TwoArmedBandit(
            exploration_c=float(getattr(self.config.search.database, "ucb_c", 1.0))
        )
        self._assigned: Dict[int, str] = {}

    async def _route(self, start_iteration, max_iterations, retry_times, checkpoint_callback):
        cursor = start_iteration
        end = start_iteration + max_iterations

        while cursor < end and not self.stop_requested():
            count = min(self.max_parallel, end - cursor)
            pending: Dict[str, int] = {}
            for iteration in range(cursor, cursor + count):
                arm = self.bandit.select(pending)
                pending[arm] = pending.get(arm, 0) + 1
                self._assigned[iteration] = arm

            best_before = self.best_score
            produced: List[Program] = await self.run_block(
                cursor,
                count,
                CHEAP,
                retry_times=retry_times,
                checkpoint_callback=checkpoint_callback,
                tier_of=self._assigned.get,
            )

            baseline = best_before if best_before > float("-inf") else 0.0
            succeeded = set()
            for program in produced:
                arm = (program.metadata or {}).get("model_tier") or CHEAP
                improvement = get_score(program.metrics or {}) - baseline
                self.bandit.observe(arm, improvement)
                succeeded.add(program.iteration_found)

            # A generation that produced nothing (unparseable output, or an
            # evaluation that failed every retry) is still a pull that bought
            # no improvement — otherwise an arm that always fails would stay
            # at zero pulls and be re-picked forever as "untried".
            for iteration in range(cursor, cursor + count):
                if iteration not in succeeded:
                    self.bandit.observe(self._assigned[iteration], 0.0)

            logger.info("Bandit arms after generation %d: %s", cursor + count, self.bandit.state())
            cursor += count

    def _extra_summary(self) -> Dict:
        counts = {CHEAP: 0, STRONG: 0}
        for tier in self._assigned.values():
            counts[tier] = counts.get(tier, 0) + 1
        return {"bandit": self.bandit.state(), "tier_draws": counts}
