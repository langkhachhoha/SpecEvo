"""Grow-Deepen bandit scheduler and the two-armed cheap/strong router bandit.

``GrowDeepenScheduler`` implements Eq. (9): a recent-window UCB over one
shared ``Grow`` meta-arm and one ``Deepen(i)`` arm per live trajectory.  The
window makes it non-stationary-aware (Besbes et al., 2014) — a trajectory
that has stopped paying off loses its lead even after a strong start.

``TwoArmedBandit`` is the *baseline* router: arms are the cheap and strong
models, and the reward is the realized improvement in best-so-far fitness.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GROW = "grow"


@dataclass
class _ArmStats:
    rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    pulls: int = 0

    def update(self, reward: float) -> None:
        self.rewards.append(float(reward))
        self.pulls += 1

    @property
    def recent_mean(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0


class GrowDeepenScheduler:
    """Chooses ``Grow`` or ``Deepen(i)`` for the next cheap-model block."""

    def __init__(
        self,
        exploration_c: float = 0.5,
        window: int = 8,
        max_trajectories: int = 6,
        trajectory_horizon: int = 8,
        init_grow_blocks: int = 2,
    ):
        self.exploration_c = float(exploration_c)
        self.window = max(1, int(window))
        self.max_trajectories = max(1, int(max_trajectories))
        self.trajectory_horizon = max(1, int(trajectory_horizon))
        self.init_grow_blocks = max(1, int(init_grow_blocks))

        self.arms: Dict[str, _ArmStats] = {GROW: _ArmStats(deque(maxlen=self.window))}
        self.blocks_run: Dict[int, int] = {}
        self.t = 0

    # ------------------------------------------------------------------

    def _arm(self, key: str) -> _ArmStats:
        arm = self.arms.get(key)
        if arm is None:
            arm = _ArmStats(deque(maxlen=self.window))
            self.arms[key] = arm
        return arm

    @staticmethod
    def deepen_key(traj_id: int) -> str:
        return f"deepen:{traj_id}"

    def available_actions(self, live_trajectories: List[int]) -> List[str]:
        actions: List[str] = []
        if len(live_trajectories) < self.max_trajectories:
            actions.append(GROW)
        for traj_id in live_trajectories:
            if self.blocks_run.get(traj_id, 0) < self.trajectory_horizon:
                actions.append(self.deepen_key(traj_id))
        return actions

    def score(self, action: str) -> float:
        """``U_a(t)`` (Eq. 9); unpulled arms score ``+inf`` so each is tried once."""
        arm = self.arms.get(action)
        if arm is None or arm.pulls == 0:
            return math.inf
        bonus = self.exploration_c * math.sqrt(math.log(max(2, self.t + 1)) / max(1, arm.pulls))
        return arm.recent_mean + bonus

    def select(self, live_trajectories: List[int]) -> Optional[str]:
        """Pick the next block action, or ``None`` when nothing is available."""
        actions = self.available_actions(live_trajectories)
        if not actions:
            return None

        # Warm-up: a few independent Grow blocks seed the trajectory set and
        # fill the relay bank before the bandit statistics mean anything.
        if len(self.blocks_run) < self.init_grow_blocks and GROW in actions:
            return GROW

        return max(actions, key=lambda a: (self.score(a), a == GROW))

    def observe(self, action: str, traj_id: int, reward: float, count_reward: bool) -> None:
        """Record the block outcome.

        ``count_reward=False`` for the warm-up Grow blocks: gains observed
        while an empty bank is being filled say more about arrival order than
        about the trajectory, so they are kept out of the scheduler history.
        """
        self.t += 1
        self.blocks_run[traj_id] = self.blocks_run.get(traj_id, 0) + 1
        if not count_reward:
            return
        if action == GROW:
            # A Grow reward updates the shared Grow statistics *and* seeds the
            # new trajectory's own utility estimate.
            self._arm(GROW).update(reward)
            self._arm(self.deepen_key(traj_id)).update(reward)
        else:
            self._arm(self.deepen_key(traj_id)).update(reward)

    def state(self) -> Dict[str, Dict[str, float]]:
        return {
            key: {"pulls": arm.pulls, "recent_mean": round(arm.recent_mean, 6)}
            for key, arm in self.arms.items()
        }


class TwoArmedBandit:
    """UCB1 over ``{cheap, strong}`` for the *Bandit* routing baseline.

    Reward is the realized best-so-far improvement produced by the call,
    rescaled by the largest improvement seen so far so that UCB's exploration
    bonus stays commensurate across benchmarks with very different fitness
    scales.
    """

    def __init__(self, exploration_c: float = 1.0, arms: Tuple[str, ...] = ("cheap", "strong")):
        self.exploration_c = float(exploration_c)
        self.arm_names = arms
        self.sums: Dict[str, float] = {a: 0.0 for a in arms}
        self.pulls: Dict[str, int] = {a: 0 for a in arms}
        self.t = 0
        self._max_improvement = 0.0

    def select(self, pending: Optional[Dict[str, int]] = None) -> str:
        """Pick an arm, counting arms already assigned in this batch.

        A batch of generations is dispatched together, so ``select`` is called
        several times before a single reward comes back. Charging each
        assignment as a provisional pull keeps a batch from collapsing onto one
        arm: the first call still takes an untried arm, the next sees that arm
        as tried and its confidence bound as narrowed.
        """
        pending = pending or {}
        effective = {a: self.pulls[a] + pending.get(a, 0) for a in self.arm_names}
        for arm in self.arm_names:
            if effective[arm] == 0:
                return arm
        self.t = max(self.t, 1)

        def score(arm: str) -> float:
            mean = self.sums[arm] / self.pulls[arm] if self.pulls[arm] else 0.0
            return mean + self.exploration_c * math.sqrt(
                math.log(max(2, self.t + 1)) / effective[arm]
            )

        return max(self.arm_names, key=score)

    def observe(self, arm: str, improvement: float) -> float:
        improvement = max(0.0, float(improvement))
        self._max_improvement = max(self._max_improvement, improvement)
        reward = 0.0 if self._max_improvement <= 0 else improvement / self._max_improvement
        self.sums[arm] = self.sums.get(arm, 0.0) + reward
        self.pulls[arm] = self.pulls.get(arm, 0) + 1
        self.t += 1
        return reward

    def state(self) -> Dict[str, Dict[str, float]]:
        return {
            arm: {
                "pulls": self.pulls[arm],
                "mean_reward": (
                    round(self.sums[arm] / self.pulls[arm], 6) if self.pulls[arm] else 0.0
                ),
            }
            for arm in self.arm_names
        }
