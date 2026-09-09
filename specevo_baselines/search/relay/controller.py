"""RelayEvolve: adaptive population handoff over the OpenEvolve backend.

The run has two phases and one decision that joins them:

**Cheap phase.**  A cheap model explores several trajectories, each its own
island/MAP-Elites population, advanced in fixed-length blocks of ``h``
generations.  A Grow-Deepen bandit picks which trajectory each block extends
(or whether to start a new one).  After every block the online relay bank is
updated and the block's *Relay Gain* becomes the bandit's reward.

**Handoff.**  When Relay Gain stays below ``epsilon_rel`` for ``patience``
consecutive blocks — or the cheap stage budget ``B_c`` runs out — the relay
objective is re-optimised over the *whole* terminal candidate pool with greedy
submodular selection plus local search, yielding a compact seed set ``S*``.

**Strong phase.**  ``S*`` initialises one shared strong-model population, which
spends the remaining budget refining it.  Seeds from different cheap
trajectories can therefore be crossed and recombined inside a single
population, which is the point of relaying rather than routing.

Every phase runs the parallel loop: within a block, up to
``max_parallel_iterations`` generations are in flight at once.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from specevo_baselines.search.base_database import Program
from specevo_baselines.search.default_discovery_controller import DiscoveryControllerInput
from specevo_baselines.search.openevolve_native.database import OpenEvolveNativeDatabase
from specevo_baselines.search.relay.bank import Candidate, RelayBank, curate_seed_population
from specevo_baselines.search.relay.embedding import CandidateEmbedder
from specevo_baselines.search.relay.scheduler import GROW, GrowDeepenScheduler
from specevo_baselines.search.relay.tiered import CHEAP, STRONG, TieredController, program_text_view
from specevo_baselines.search.utils.discovery_utils import SerializableResult
from specevo_baselines.utils.metrics import get_score

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """One cheap-model search thread with its own population."""

    id: int
    database: OpenEvolveNativeDatabase
    blocks: int = 0
    iterations: int = 0
    best_score: float = float("-inf")
    gains: List[float] = field(default_factory=list)


class RelayEvolveController(TieredController):
    method_name = "relayevolve"

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        opts = self.config.search.database

        # --- block scheduling -------------------------------------------------
        self.block_size = max(1, int(getattr(opts, "block_size", 5)))
        self.max_trajectories = max(1, int(getattr(opts, "max_trajectories", 5)))
        self.trajectory_horizon = max(1, int(getattr(opts, "trajectory_horizon", 6)))
        self.init_grow_blocks = max(1, int(getattr(opts, "init_grow_blocks", 2)))
        self.cheap_max_iteration_frac = float(getattr(opts, "cheap_max_iteration_frac", 0.5))

        # --- relay objective --------------------------------------------------
        self.bank_k = max(1, int(getattr(opts, "bank_size", 8)))
        self.bank_r = max(1, int(getattr(opts, "quality_top_r", 3)))
        self.lam = float(getattr(opts, "relay_lambda", 0.5))
        self.eta = float(getattr(opts, "code_view_eta", 0.7))
        self.epsilon_f = float(getattr(opts, "epsilon_f", 1e-3))

        # --- handoff rule -----------------------------------------------------
        self.epsilon_rel = float(getattr(opts, "epsilon_rel", 0.02))
        self.patience = max(1, int(getattr(opts, "patience", 3)))

        # --- ablation switches (Figure 4) ------------------------------------
        self.random_allocation = bool(getattr(opts, "random_allocation", False))
        self.disable_relay_stop = bool(getattr(opts, "disable_relay_stop", False))
        self.curation_random = bool(getattr(opts, "curation_random", False))

        self.embedder = CandidateEmbedder(
            backend=str(getattr(opts, "embedding_backend", "hash")),
            dim=int(getattr(opts, "embedding_dim", 512)),
            model=getattr(opts, "embedding_model", None),
            api_base=getattr(opts, "embedding_api_base", None),
            api_key=getattr(opts, "embedding_api_key", None) or os.environ.get("OPENAI_API_KEY"),
        )
        self.bank = RelayBank(
            embedder=self.embedder,
            k=self.bank_k,
            r=self.bank_r,
            lam=self.lam,
            eta=self.eta,
            epsilon_f=self.epsilon_f,
            max_pool=int(getattr(opts, "max_candidate_pool", 600)),
        )
        self.scheduler = GrowDeepenScheduler(
            exploration_c=float(getattr(opts, "ucb_c", 0.5)),
            window=int(getattr(opts, "ucb_window", 8)),
            max_trajectories=self.max_trajectories,
            trajectory_horizon=self.trajectory_horizon,
            init_grow_blocks=self.init_grow_blocks,
        )

        self.trajectories: Dict[int, Trajectory] = {}
        self.block_log: List[Dict[str, Any]] = []
        self._rel_gains: List[float] = []
        self._handoff_reason: Optional[str] = None
        self._seed_info: List[Dict[str, Any]] = []
        self._root: Optional[Program] = None

    # ==================================================================
    # Main loop
    # ==================================================================

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
    ) -> Optional[Union[Program, SerializableResult]]:
        self.requested_iterations = max_iterations
        self._root = self._resolve_root()
        if self._root is None:
            logger.error("RelayEvolve needs an evaluated initial program; falling back.")
            return await super().run_discovery(
                start_iteration,
                max_iterations,
                checkpoint_callback,
                post_process_result,
                self.retry_times,
            )

        main_database = self.database
        self._track_best(self._root)
        self.bank.add_to_pool([self._to_candidate(self._root, trajectory=-1)])
        self.mark_phase("start", start_iteration)

        cursor = await self._cheap_phase(start_iteration, max_iterations, self.retry_times)
        # Dispatched vs completed: the generation cap counts what was launched,
        # the summary reports what actually produced a program.
        cheap_iterations = cursor - start_iteration
        cheap_completed = self.iterations_used

        self._handoff(main_database, cursor)

        remaining = max_iterations - cheap_iterations
        if remaining > 0 and not self.stop_requested():
            await self._strong_phase(cursor, remaining, self.retry_times, checkpoint_callback)
        elif remaining <= 0:
            logger.info("RelayEvolve: generation budget spent during the cheap phase.")

        self._ensure_best_in_database()
        self.mark_phase("end", start_iteration + self.iterations_used)
        self.write_summary(
            {
                "handoff_reason": self._handoff_reason,
                "cheap_iterations": cheap_iterations,
                "cheap_completed": cheap_completed,
                "strong_iterations": max(0, self.iterations_used - cheap_completed),
                "block_size": self.block_size,
                "blocks": self.block_log,
                "trajectories": {
                    str(t.id): {
                        "blocks": t.blocks,
                        "iterations": t.iterations,
                        "best_score": None if t.best_score == float("-inf") else t.best_score,
                    }
                    for t in self.trajectories.values()
                },
                "scheduler": self.scheduler.state(),
                "seeds": self._seed_info,
                "candidate_pool_size": len(self.bank.pool),
            }
        )
        return self._finalize_discovery()

    # ==================================================================
    # Phase 1 — cheap multi-trajectory exploration
    # ==================================================================

    async def _cheap_phase(
        self, start_iteration: int, max_iterations: int, retry_times: int
    ) -> int:
        self.phase = CHEAP
        cursor = start_iteration
        cheap_cap = self._cheap_iteration_cap(max_iterations)
        stage_budget = self.cheap_stage_budget()

        logger.info(
            "RelayEvolve cheap phase: ≤%d generations in blocks of %d, ≤%d trajectories, "
            "stage budget %s",
            cheap_cap,
            self.block_size,
            self.max_trajectories,
            f"${stage_budget:.4f}" if stage_budget else "unbounded",
        )

        while True:
            done_reason = self._cheap_stop_reason(cursor - start_iteration, cheap_cap, stage_budget)
            if done_reason:
                self._handoff_reason = done_reason
                break

            live = [t.id for t in self.trajectories.values()]
            action = (
                self._random_action(live) if self.random_allocation else self.scheduler.select(live)
            )
            if action is None:
                self._handoff_reason = "exploration_exhausted"
                break

            if action == GROW:
                traj = self._grow_trajectory()
            else:
                traj = self.trajectories[int(action.split(":")[1])]

            budget_left = cheap_cap - (cursor - start_iteration)
            count = max(1, min(self.block_size, budget_left))

            self.database = traj.database
            produced = await self.run_block(cursor, count, CHEAP, retry_times=retry_times)
            cursor += count
            traj.iterations += count
            traj.blocks += 1

            candidates = [self._to_candidate(p, traj.id) for p in produced]
            gain, rel_gain = self.bank.update_block(candidates)
            traj.gains.append(gain)
            for program in produced:
                traj.best_score = max(traj.best_score, get_score(program.metrics or {}))

            warmup = len(self.scheduler.blocks_run) < self.init_grow_blocks
            self.scheduler.observe(action, traj.id, rel_gain, count_reward=not warmup)
            if not warmup:
                self._rel_gains.append(rel_gain)

            self.block_log.append(
                {
                    "block": len(self.block_log),
                    "action": action,
                    "trajectory": traj.id,
                    "iterations": count,
                    "valid_candidates": len(candidates),
                    "relay_gain": round(gain, 6),
                    "relative_relay_gain": round(rel_gain, 6),
                    "bank_objective": round(self.bank.objective(self.bank.bank), 6),
                    "warmup": warmup,
                    "cost_usd": self.spent_usd(),
                }
            )
            logger.info(
                "Relay block %d: %s traj=%d (+%d gens) | gain=%.4f rel=%.4f | "
                "bank F=%.4f | pool=%d | best=%.4f | $%.4f",
                len(self.block_log) - 1,
                "GROW" if action == GROW else "DEEPEN",
                traj.id,
                count,
                gain,
                rel_gain,
                self.bank.objective(self.bank.bank),
                len(self.bank.pool),
                self.best_score,
                self.spent_usd(),
            )

        logger.info(
            "RelayEvolve cheap phase done after %d generations across %d trajectories (%s).",
            cursor - start_iteration,
            len(self.trajectories),
            self._handoff_reason,
        )
        return cursor

    def _cheap_iteration_cap(self, max_iterations: int) -> int:
        """Generations the cheap phase may dispatch.

        Three caps, whichever is smallest: what the trajectory structure can
        hold, the fraction guard on the total generation budget, and one full
        block (so a short run still gets a real block). Never more than the
        run's own generation budget.
        """
        structural = self.max_trajectories * self.trajectory_horizon * self.block_size
        by_fraction = int(math.ceil(max_iterations * self.cheap_max_iteration_frac))
        return min(max_iterations, max(self.block_size, min(structural, by_fraction)))

    def _cheap_stop_reason(
        self, used: int, cap: int, stage_budget: Optional[float]
    ) -> Optional[str]:
        if self.stop_requested():
            return "budget_stop" if self.budget_stop_triggered else "shutdown"
        if used >= cap:
            return "cheap_generation_cap"
        if stage_budget is not None and self.spent_usd() >= stage_budget:
            return "cheap_stage_budget"
        if (
            not self.disable_relay_stop
            and len(self._rel_gains) >= self.patience
            and max(self._rel_gains[-self.patience :]) < self.epsilon_rel
        ):
            return "relay_gain_saturated"
        return None

    def _random_action(self, live: List[int]) -> Optional[str]:
        """Ablation: replace the Grow/Deepen bandit with a uniform choice."""
        actions = self.scheduler.available_actions(live)
        return random.choice(actions) if actions else None

    def _grow_trajectory(self) -> Trajectory:
        traj_id = len(self.trajectories)
        database = self._new_population(f"relay_traj_{traj_id}", [self._root])
        traj = Trajectory(id=traj_id, database=database)
        self.trajectories[traj_id] = traj
        logger.info("Relay GROW: started trajectory %d from the shared initial program", traj_id)
        return traj

    # ==================================================================
    # Handoff
    # ==================================================================

    def _handoff(self, main_database: OpenEvolveNativeDatabase, cursor: int) -> None:
        self.handoff_iteration = cursor
        if self.curation_random:
            pool = list(self.bank.pool)
            seeds = random.sample(pool, min(self.bank_k, len(pool)))
        else:
            seeds = curate_seed_population(self.bank, self.bank_k)
        seed_ids = {s.id for s in seeds}

        # Hold the Program objects on the candidates: a candidate can be
        # evicted from its trajectory's population by the size limit long
        # before handoff, and a curated seed must survive that.
        seed_programs: List[Program] = [
            s.metadata["program"] for s in seeds if isinstance(s.metadata.get("program"), Program)
        ]
        if not seed_programs:
            logger.warning("Relay curation produced no seeds — seeding with the initial program.")
            seed_programs = [self._root]
        elif self.best_program is not None and self.best_program.id not in seed_ids:
            # The strong phase must never start behind the cheap phase.
            seed_programs.append(self.best_program)

        self._seed_info = [
            {
                "id": p.id,
                "score": get_score(p.metrics or {}),
                "trajectory": (p.metadata or {}).get("relay_trajectory"),
                "iteration_found": p.iteration_found,
            }
            for p in seed_programs
        ]

        self.database = self._new_population("relayevolve", seed_programs, base=main_database)
        self.mark_phase("handoff", cursor)
        logger.info(
            "🔀 Relay handoff at generation %d: %d seed(s), best=%.4f, spent $%.4f (%s)",
            cursor,
            len(seed_programs),
            self.best_score,
            self.spent_usd(),
            self._handoff_reason,
        )

    # ==================================================================
    # Phase 2 — shared strong-model refinement
    # ==================================================================

    async def _strong_phase(
        self,
        start_iteration: int,
        count: int,
        retry_times: int,
        checkpoint_callback: Optional[Callable[[int], None]],
    ) -> None:
        self.phase = STRONG
        remaining = self.budget_remaining()
        logger.info(
            "RelayEvolve strong phase: up to %d generations from %d seeds, %s left",
            count,
            len(self.database.programs),
            f"${remaining:.4f}" if remaining is not None else "unbounded budget",
        )
        await self.run_block(
            start_iteration,
            count,
            STRONG,
            retry_times=retry_times,
            checkpoint_callback=checkpoint_callback,
        )

    # ==================================================================
    # Helpers
    # ==================================================================

    def _resolve_root(self) -> Optional[Program]:
        if not self.database.programs:
            return None
        root_id = getattr(self.database, "initial_program_id", None)
        if root_id and root_id in self.database.programs:
            return self.database.programs[root_id]
        return self.database.get_best_program()

    def _to_candidate(self, program: Program, trajectory: int) -> Candidate:
        if program.metadata is not None:
            program.metadata["relay_trajectory"] = trajectory
        return Candidate(
            id=program.id,
            solution=program.solution,
            score=get_score(program.metrics or {}),
            text=program_text_view(program),
            trajectory=trajectory,
            iteration=program.iteration_found,
            # Hold the Program itself: a candidate can be evicted from its
            # trajectory's population by the size limit long before handoff,
            # and a curated seed must survive that.
            metadata={"program": program},
        )

    def _new_population(
        self,
        name: str,
        seeds: List[Program],
        base: Optional[OpenEvolveNativeDatabase] = None,
    ) -> OpenEvolveNativeDatabase:
        """A fresh island/MAP-Elites population initialised from ``seeds``."""
        db_config = copy.copy(self.config.search.database)
        # Sub-populations must not re-seed the global RNG (that would make
        # every trajectory sample identically) and must not fight over the
        # single on-disk program directory.
        db_config.random_seed = None
        db_config.db_path = None

        database = OpenEvolveNativeDatabase(name, db_config)
        database.language = getattr(base, "language", None) or self.config.language or "python"

        best_seed, best_seed_score = None, float("-inf")
        for seed in seeds:
            clone = copy.deepcopy(seed)
            clone.metadata = dict(clone.metadata or {})
            clone.metadata.pop("island", None)
            database.add(clone)
            score = get_score(clone.metrics or {})
            if score > best_seed_score:
                best_seed, best_seed_score = clone, score

        if best_seed is not None:
            database.initial_program_id = best_seed.id
            database.initial_program_score = best_seed_score
        return database

    def _ensure_best_in_database(self) -> None:
        """Guarantee the run returns the best program found in *either* phase."""
        if self.best_program is None:
            return
        if self.best_program.id in self.database.programs:
            return
        current = self.database.get_best_program()
        if current is not None and get_score(current.metrics or {}) >= self.best_score:
            return
        clone = copy.deepcopy(self.best_program)
        clone.metadata = dict(clone.metadata or {})
        clone.metadata.pop("island", None)
        self.database.add(clone)
        logger.info(
            "Re-inserted the cheap-phase best program (score=%.4f) into the final population.",
            self.best_score,
        )
