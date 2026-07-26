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


def task_pool_dir(root: Path) -> Path:
    """The directory Game 2 points a local Harbor dataset at.

    Every accepted task is a child of this one, which is what makes the growing
    task set a dataset rather than a list of paths to stitch together.
    """
    path = root / "task_pool"
    path.mkdir(parents=True, exist_ok=True)
    return path
