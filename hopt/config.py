"""Experiment configuration for harness optimization.

Three ablation axes, mirroring the three factors in the paper:

1. ``harness``          -> starting artifact
2. ``batch_size``       -> experience batching
3. ``credit_horizon``   -> credit horizon (fraction of the agent trajectory shown
                           to the optimizer)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
JOBS_DIR = REPO_ROOT / "jobs"
RESULTS_DIR = REPO_ROOT / "results"


# --------------------------------------------------------------------------
# Axis 1: starting artifact (harness)
# --------------------------------------------------------------------------
# Each harness names a Harbor agent plus the surface the optimizer may rewrite.
#
# mini-swe-agent subclasses BaseInstalledAgent, which accepts a
# `prompt_template_path` kwarg (a Jinja2 template that must reference
# {{ instruction }}). That is a natively supported, per-run optimizable surface.
#
# Terminus-2 subclasses BaseAgent and has no such kwarg; its scaffold prompt is
# a package file. hopt.agents.OptimizableTerminus2 adds the same kwarg by
# overriding _get_prompt_template_path(), so both harnesses expose one uniform
# optimizable artifact.

HARNESSES: dict[str, dict] = {
    # --- code artifacts: the optimizer rewrites a directory of agent source ---
    # Two seeds differing only in decomposition, mirroring the one-function vs
    # many-function contrast in the paper's MLAgentBench study, now at the
    # harness level.
    "code-mono": {
        "agent_name": None,
        "import_path": "hopt.code_agent:CodeArtifactAgent",
        "seed_artifact": "seeds/code_agent",
        "surface": "code_directory",
        "is_directory": True,
        "entrypoint": "run_agent",
    },
    "code-modular": {
        "agent_name": None,
        "import_path": "hopt.code_agent:CodeArtifactAgent",
        "seed_artifact": "seeds/code_agent_modular",
        "surface": "code_directory",
        "is_directory": True,
        "entrypoint": "run_agent",
    },
    # --- prompt-only harnesses (scaffold text, not code) ---
    "terminus-2": {
        "agent_name": None,
        "import_path": "hopt.agents:OptimizableTerminus2",
        "seed_artifact": "seeds/terminus2_code",
        "surface": "code_directory",
        "is_directory": True,
        "entrypoint": "build_scaffold",
    },
    # mini-swe-agent driven host-side (hopt.mini_agent). Unlike the packaged
    # adapter above, this exposes the agent's own system_template and
    # instance_template as the optimizable artifact rather than a wrapper around
    # the instruction, so the starting-artifact axis actually moves the scaffold.
    "mini-swe-host": {
        "agent_name": None,
        "import_path": "hopt.mini_agent:MiniSweAgentHost",
        "seed_artifact": "seeds/mini_swe_host_code",
        "surface": "code_directory",
        "is_directory": True,
        "entrypoint": "build_templates",
    },
}

CreditHorizon = Literal["first_10", "first_50", "full"]
HORIZON_FRACTIONS: dict[str, float] = {
    "first_10": 0.10,
    "first_50": 0.50,
    "full": 1.00,
}


@dataclass
class ExperimentConfig:
    # --- identity -------------------------------------------------------
    run_name: str

    # --- axis 1: starting artifact --------------------------------------
    harness: str = "code-mono"

    # --- axis 2: experience batching ------------------------------------
    # Number of task rollouts aggregated into one optimizer update.
    # `None` means "full batch": update once per pass over the train split.
    batch_size: int | None = 5

    # --- axis 3: credit horizon -----------------------------------------
    credit_horizon: CreditHorizon = "full"

    # --- data -----------------------------------------------------------
    train_dataset: str = "openthoughts-tblite@2.0"
    # Transfer/eval datasets, scored only with the final artifact.
    transfer_datasets: tuple[str, ...] = ("terminal-bench@2.0", "arc_agi_2@1.0")
    train_frac: float = 0.20          # cap: at most 20% of tasks used for training
    split_seed: int = 0
    # Fraction of the training budget reserved for candidate selection. Carved
    # out of train, never out of test, so the held-out set is identical whether
    # or not selection is on. 0.0 disables selection (accept every candidate).
    val_frac_of_train: float = 0.30
    # Keep the best-scoring artifact rather than the most recent one. Without
    # this a rewrite that makes things worse is still carried forward and the
    # optimizer keeps building on it.
    keep_best: bool = True
    # Two-stage evaluation. The sweep only needs to *rank* cells against each
    # other, so it scores a capped subsample; the cells that get reported are
    # then re-scored on the full held-out sets, because a headline number drawn
    # from a 50-task subsample when 80 were available invites exactly the
    # "limited evaluation" criticism the paper is already fighting.
    #   sweep : eval_subsample tasks per dataset
    #   final : all tasks (run with --final, or eval_subsample=None)
    eval_subsample: int | None = 50
    # Abort a cell when an entire batch is harness-broken (no LLM calls, or all
    # observations empty) rather than optimizing on evidence that is not real.
    strict_health: bool = True
    # Consecutive unhealthy batches tolerated by rolling back to the last healthy
    # artifact before the cell is declared broken.
    max_rollbacks: int = 2

    # --- execution ------------------------------------------------------
    env_type: str = "modal"           # harbor environment backend
    model_name: str = "anthropic/claude-haiku-4-5"    # model the *agent* runs
    # Model that rewrites the artifact. Sonnet 5 measured at 3.9 min/call vs
    # Opus 5's 6.7 min on an identical 691-line artifact + 140k-char evidence,
    # with an equally valid rewrite. The optimizer call is ~30-40% of each
    # iteration's serial latency, so this is the single biggest wall-clock lever.
    optimizer_model: str = "claude-sonnet-5"
    n_concurrent: int = 20
    n_attempts: int = 1
    # Code harnesses may need to install python3/ca-certificates into a minimal
    # task image; Harbor's default agent-setup budget is 360s, which is not
    # enough. Only applied to directory-of-code harnesses.
    setup_timeout_sec: int = 900

    # --- loop -----------------------------------------------------------
    n_iterations: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.harness not in HARNESSES:
            raise ValueError(
                f"unknown harness {self.harness!r}; choose from {sorted(HARNESSES)}"
            )
        if self.credit_horizon not in HORIZON_FRACTIONS:
            raise ValueError(
                f"unknown credit_horizon {self.credit_horizon!r}; "
                f"choose from {sorted(HORIZON_FRACTIONS)}"
            )
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1 or None (full batch)")
        # The 20% cap was the original methodological choice: fit on a small
        # minority of tasks, evaluate on the rest, which is the stronger
        # generalization claim. It is now a soft default rather than a hard
        # limit -- raising it trades held-out test size for fitting data. The
        # transfer benchmarks stay 100% held out either way, so the headline
        # generalization evidence does not depend on this number.
        if not 0 < self.train_frac <= 0.50:
            raise ValueError(
                f"train_frac must be in (0, 0.50], got {self.train_frac}. "
                "Above 0.20 the held-out test set shrinks; state the split "
                "explicitly in any reported result."
            )

    # --- derived --------------------------------------------------------
    @property
    def horizon_fraction(self) -> float:
        return HORIZON_FRACTIONS[self.credit_horizon]

    @property
    def harness_spec(self) -> dict:
        return HARNESSES[self.harness]

    @property
    def entrypoint_spec(self):
        """EntrypointSpec for this harness's artifact contract."""
        from hopt.artifact import (
            BUILD_SCAFFOLD_SPEC,
            BUILD_TEMPLATES_SPEC,
            RUN_AGENT_SPEC,
        )

        return {
            "run_agent": RUN_AGENT_SPEC,
            "build_scaffold": BUILD_SCAFFOLD_SPEC,
            "build_templates": BUILD_TEMPLATES_SPEC,
        }[self.harness_spec.get("entrypoint", "run_agent")]

    @property
    def run_dir(self) -> Path:
        return RESULTS_DIR / self.run_name

    @property
    def artifact_dir(self) -> Path:
        return ARTIFACTS_DIR / self.run_name

    @property
    def seed_artifact_path(self) -> Path:
        return ARTIFACTS_DIR / self.harness_spec["seed_artifact"]

    def cell_id(self) -> str:
        """Stable identifier for this (harness, batch, horizon) cell."""
        b = "full" if self.batch_size is None else str(self.batch_size)
        return f"{self.harness}__b{b}__{self.credit_horizon}__s{self.seed}"

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        p = self.run_dir / "config.json"
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p
