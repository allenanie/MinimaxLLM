"""A proposed Harbor task, as an optimizer-written artifact.

Game 2's adversary proposes $(x, v_x)$ -- a task and its verifier. In Harbor terms
that is a task *directory*, and it maps onto the optimizer's existing
``<file path=...>`` output format without any new machinery.

Feasible without publishing to a registry because ``DatasetConfig`` accepts a
local path (``harbor/src/harbor/models/job/config.py``): a "dataset" is just a
directory whose children are valid task dirs, so generated tasks can be run by
pointing a job at the directory they were written into.
"""

from __future__ import annotations

from pathlib import Path

from hopt.artifact import ArtifactError, CodeArtifact, EntrypointSpec, TASK_BUNDLE_SPEC


class TaskArtifact(CodeArtifact):
    """A single Harbor task directory.

    Adds Harbor's own validity check on top of the generic file-shape checks.
    Doing it here rather than at job-submission time matters: an invalid task dir
    is silently *skipped* by ``_get_local_task_configs`` (it filters on
    ``is_valid_dir``), so without this check a malformed proposal would look like
    an empty task list rather than a rejected candidate.
    """

    def validate(self, spec: EntrypointSpec | None = None) -> None:
        spec = spec or TASK_BUNDLE_SPEC
        super().validate(spec)

        # Imported lazily: harbor pulls in a large dependency tree, and the
        # generic artifact checks above should stay usable without it.
        from harbor.models.task.task import Task as TaskModel

        if not TaskModel.is_valid_dir(self.root):
            raise ArtifactError(
                f"{self.root.name} is not a valid Harbor task directory. It needs a "
                "parseable task.toml, an environment/ directory with a Dockerfile, "
                "an instruction.md, and a tests/ entry the config points at. Harbor "
                "silently skips invalid task dirs, so this must fail here."
            )

    def reward_path_declared(self) -> bool:
        """Whether tests/test.sh mentions the reward file it is required to write.

        A cheap, static smoke check. A verifier that never writes a reward yields
        0 for every harness, which is indistinguishable from an impossible task
        until the solvability gate runs -- and the gate costs a container build.
        """
        tests = self.files().get("tests/test.sh", "")
        return "reward.txt" in tests or "rewards.json" in tests


#: Harbor caches downloaded tasks here, keyed by content hash then task name.
TASK_CACHE = Path.home() / ".cache" / "harbor" / "tasks"


def resolve_cached_task(task_name: str, cache: Path | None = None) -> Path | None:
    """Find a real benchmark task's directory on disk, or None if not cached.

    Reads Harbor's download cache rather than re-resolving through the registry:
    ``DatasetConfig.get_task_configs`` returns bare names for registry datasets
    and only materializes paths during a job, so it cannot answer this question
    without side effects.

    A miss is not an error. Grounding is a nice-to-have for the proposer, and a
    cold cache should degrade the prompt, not stop the run.
    """
    root = cache or TASK_CACHE
    if not root.is_dir():
        return None
    for config_path in root.glob(f"*/{task_name}/task.toml"):
        return config_path.parent
    return None


def render_task_as_example(task_dir: Path, max_chars_per_file: int = 2000) -> str:
    """Render a real task as a worked example of the bundle the proposer must emit.

    Shows the same four files the proposer is asked for -- instruction, verifier,
    reference solution, image -- because the most useful grounding is a correct
    instance of the exact artifact under construction, not a description of it.

    Truncated per file: a real task's Dockerfile can carry a large data blob, and
    the point is the shape, not the payload.
    """
    shown = ("instruction.md", "tests/test.sh", "solution/solve.sh", "environment/Dockerfile")
    blocks = [f"### Real task: {task_dir.name}"]
    for rel in shown:
        path = task_dir / rel
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + f"\n... [{len(text) - max_chars_per_file} chars elided]"
        blocks.append(f'<example_file path="{rel}">\n{text}\n</example_file>')
    other = sorted(
        str(p.relative_to(task_dir))
        for p in task_dir.rglob("*")
        if p.is_file() and str(p.relative_to(task_dir)) not in shown and "__pycache__" not in str(p)
    )
    if other:
        blocks.append(f"(this task also ships: {', '.join(other[:12])})")
    return "\n\n".join(blocks)


def task_pool_dir(root: Path) -> Path:
    """The directory Game 2 points a local Harbor dataset at.

    Every accepted task is a child of this one, which is what makes the growing
    task set a dataset rather than a list of paths to stitch together.
    """
    path = root / "task_pool"
    path.mkdir(parents=True, exist_ok=True)
    return path
