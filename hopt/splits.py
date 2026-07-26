"""Deterministic train/test splits over Harbor datasets.

Task names come from the Harbor registry, so splits are reproducible without
downloading task content. We train on at most `train_frac` of the tasks and
hold out the remainder for test.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hopt.config import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "harbor" / "registry.json"


@dataclass(frozen=True)
class Split:
    dataset: str          # "name@version"
    train: tuple[str, ...]
    test: tuple[str, ...]
    # Fixed candidate-selection set, carved out of the <=20% training budget.
    # Best-so-far selection needs a *constant* comparison set: train batches are
    # resampled every iteration, so comparing their solve rates compares
    # different task sets and selects on noise.
    val: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.dataset.split("@")[0]

    @property
    def version(self) -> str:
        return self.dataset.split("@")[1]

    def summary(self) -> str:
        return (
            f"{self.dataset}: {len(self.train)} train / {len(self.val)} val / "
            f"{len(self.test)} test "
            f"({len(self.train) + len(self.val) + len(self.test)} total)"
        )


@lru_cache(maxsize=1)
def _registry() -> list[dict]:
    return json.loads(REGISTRY_PATH.read_text())


def dataset_task_names(dataset: str, registry_path: Path | None = None) -> list[str]:
    """Return the sorted task names for a 'name@version' dataset spec."""
    name, _, version = dataset.partition("@")
    entries = (
        json.loads(Path(registry_path).read_text()) if registry_path else _registry()
    )
    for entry in entries:
        if entry.get("name") == name and (not version or entry.get("version") == version):
            return sorted(t["name"] for t in entry.get("tasks", []))
    available = sorted({e.get("name", "") for e in entries})
    raise KeyError(f"dataset {dataset!r} not in registry. Available: {available[:20]}...")


def make_split(
    dataset: str, train_frac: float, seed: int, val_frac_of_train: float = 0.0
) -> Split:
    """Shuffle task names with a fixed seed and cut train/val/test slices.

    The split is over *task names*, so the same seed yields the same split
    across every cell of the ablation grid -- the three axes are compared on
    identical data.

    ``val_frac_of_train`` carves the selection set out of the training budget,
    not out of test: train + val together stay within ``train_frac`` of the
    dataset, so the held-out test set is unchanged whether or not selection is
    enabled.
    """
    names = dataset_task_names(dataset)
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)

    n_budget = max(1, int(len(shuffled) * train_frac))
    n_val = int(n_budget * val_frac_of_train)
    if val_frac_of_train > 0:
        n_val = max(1, min(n_val, n_budget - 1))
    n_train = n_budget - n_val

    return Split(
        dataset=dataset,
        train=tuple(sorted(shuffled[:n_train])),
        val=tuple(sorted(shuffled[n_train:n_budget])),
        test=tuple(sorted(shuffled[n_budget:])),
    )


def batch_for_iteration(
    train: list[str], batch_size: int | None, seed: int, iteration: int
) -> list[str]:
    """The batch for a given iteration, derived deterministically.

    Resume-safety: the previous version consumed a single mutable RNG across the
    run, so a restarted cell could not reproduce which tasks a given iteration
    had used. Deriving the batch from (seed, iteration) makes every iteration
    reproducible in isolation, so a cell can restart mid-run and continue with
    the same task sequence it would have used originally.
    """
    if batch_size is None:
        return sorted(train)
    rng = random.Random(f"{seed}:{iteration}")
    return sorted(rng.sample(train, min(batch_size, len(train))))


def batches(items: list[str], batch_size: int | None, rng: random.Random) -> list[list[str]]:
    """Partition train tasks into optimizer-update batches.

    batch_size=None means one full-batch update per pass.
    """
    pool = items[:]
    rng.shuffle(pool)
    if batch_size is None:
        return [pool]
    return [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
