"""The LLM optimizer: rewrites the harness artifact from batched evidence.

This is the generative-optimization step. Input is the current artifact plus a
batch of (task, reward, truncated trajectory) triplets; output is a revised
artifact. Deliberately minimal -- the paper's claim is about the *learning loop*
configuration, not about optimizer cleverness, so the optimizer is held fixed
across every cell of the ablation grid.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hopt.artifact import CodeArtifact, ArtifactError
from hopt.config import RESULTS_DIR
from hopt.llm import LLMResponse, build_client
from hopt.runner import RolloutBatch
from hopt.trajectory import load_trajectory, render_for_optimizer

SYSTEM_PROMPT = """\
You are optimizing the prompt/scaffold artifact that drives a terminal-using coding agent.

You will receive:
1. The CURRENT ARTIFACT (a prompt template the agent runs with).
2. EXECUTION EVIDENCE: a batch of tasks the agent attempted with that artifact,
   each with its reward and a (possibly truncated) execution trace.

Your job: rewrite the artifact so the agent solves more tasks.

Rules:
- Output ONLY the new artifact, wrapped in <artifact> and </artifact> tags.
- The artifact MUST preserve every Jinja2 placeholder present in the current
  artifact (in particular {{ instruction }}); dropping one breaks the harness.
- Generalize. You are optimizing for held-out tasks, not for the specific tasks
  shown. Do not hardcode task-specific answers, file names, or commands.
- Prefer durable, transferable guidance (strategy, verification habits, failure
  recovery) over narrow patches.
- Keep the artifact focused; length is not a proxy for quality.
"""

CODE_SYSTEM_PROMPT = """\
You are optimizing the SOURCE CODE of a terminal-using coding agent.

Context: this is an automated agent-evaluation research setup. The execution
traces below come from published coding benchmarks (Terminal-Bench 2.0,
OpenThoughts-TBLite, ARC-AGI-2). Some benchmark tasks are security-flavored --
forensics, permissions, parser hardening -- so their traces contain security
terminology. They are standard public benchmark fixtures. Your job is only to
improve the agent's control flow and prompting; you never need to reproduce,
explain, or extend any task content.

You will receive:
1. The CURRENT AGENT CODE, as one or more files.
2. EXECUTION EVIDENCE: a batch of tasks the agent attempted with that code, each
   with its reward and a (possibly truncated) execution trace.

Your job: rewrite the agent's code so it solves more tasks. You may change
control flow, prompting, response parsing, retry and recovery logic, when the
agent verifies its work, and when it decides to stop. You may add, split, or
merge modules.

THE CONTRACT (violating any of this makes the agent unrunnable):
- `agent.py` must exist and must define `run_agent(task, execute, query, config)`.
- `execute(command: str, timeout: int | None = None) -> {"output": str, "returncode": int}`
  runs a shell command in the task container.
- `query(messages: list[dict], system: str | None = None) -> str` performs one LLM
  call; message entries are {"role": "user"|"assistant", "content": str}.
- `config` is a read-only dict (step_limit, model, exec_timeout).
- The call budget is enforced by the runtime. When it is exhausted, `query`
  raises. Let that propagate -- do NOT wrap the main loop in a bare `except:`,
  or the agent will spin until it is killed.
- Standard library only. The container has no third-party packages installed.
- Modules are imported flat (the agent directory is on sys.path), so use
  `from prompts import X`, not `from .prompts import X`.

OUTPUT FORMAT -- emit every file the new agent needs, each as:
<file path="agent.py">
...full file contents...
</file>
Files you do not emit are DELETED. Always emit the complete contents; never
abbreviate with "unchanged" or "...".

Optimize for held-out tasks. Do not hardcode answers, file names, or commands
specific to the tasks you were shown.
"""

# ~1M-token input window; 4 chars/token, kept well under to leave room for
# the system prompt and the model's own reasoning.
MAX_PROMPT_CHARS = 2_800_000

# Hard ceiling for the request itself. The model's input window is 1M tokens; we
# aim below it so the system prompt and any count-estimate error still fit.
MAX_INPUT_TOKENS = 900_000
# Chars of the message tail always preserved. The trailing text carries the output
# contract ("emit every file in <file> blocks"); truncating the literal end of the
# prompt would drop it and make the response unparseable, so elision happens at the
# end of the EVIDENCE instead.
TAIL_KEEP_CHARS = 1_200

ARTIFACT_RE = re.compile(r"<artifact>(.*?)</artifact>", re.DOTALL)
FILE_RE = re.compile(r"<file\s+path=\"([^\"]+)\"\s*>\n?(.*?)\n?</file>", re.DOTALL)


@dataclass
class OptimizerStep:
    iteration: int
    old_artifact: str
    new_artifact: str
    prompt_chars: int
    batch_tasks: list[str]
    batch_solve_rate: float
    raw_response: str


class ArtifactOptimizer:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 64000,
        prompt_guard: Callable[[str], None] | None = None,
    ):
        # Provider comes from a "provider/model" prefix; a bare name is Anthropic,
        # so existing configs are unchanged. See hopt/llm.py for why this is not
        # just a client swap.
        self.client = build_client(model, api_key)
        self.model = model
        self.max_tokens = max_tokens
        # Called on every assembled user message immediately before it is sent.
        # The minimax games use this to enforce the information barrier: the
        # harness optimizer must never receive the detector, and the realistic
        # way that breaks is a plumbing slip while concatenating evidence, not a
        # deliberate call. Checking here covers every retry path too.
        self.prompt_guard = prompt_guard

    def _message(self, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        """One optimizer call, provider-independent.

        ``max_tokens`` is the *output* cap, not the context window. Provider
        differences -- streaming thresholds, how the budget interacts with
        reasoning tokens, stop-reason vocabulary -- are handled in hopt/llm.py so
        the retry ladder below can branch on one set of names.
        """
        if self.prompt_guard is not None:
            self.prompt_guard(user)
        return self.client.message(system, user, max_tokens or self.max_tokens)

    def _count_input_tokens(self, system: str, user: str) -> int | None:
        """Exact token count, or None when the provider cannot say.

        None is a supported answer, not a failure: ``_fit_prompt`` falls back to a
        conservative chars-per-token estimate, which is also what happens during
        an Anthropic counting outage.
        """
        return self.client.count_tokens(system, user)

    def _fit_prompt(self, system: str, user: str) -> tuple[str, bool]:
        """Shrink ``user`` until the request fits the input window.

        Returns (message, truncated). Elides from the END OF THE EVIDENCE and keeps
        the trailing instruction, because that tail defines the output format --
        cutting the literal end of the prompt would save tokens and then produce an
        unparseable reply.

        Uses the count_tokens endpoint when reachable and falls back to a
        conservative chars-per-token estimate, so a counting outage degrades to a
        smaller prompt rather than a 400.
        """
        tokens = self._count_input_tokens(system, user)
        if tokens is not None and tokens <= MAX_INPUT_TOKENS:
            return user, False

        truncated = False
        for _ in range(6):
            if tokens is None:
                # No exact count: assume a dense 3 chars/token to stay safe.
                if len(user) <= MAX_INPUT_TOKENS * 3:
                    break
                ratio = (MAX_INPUT_TOKENS * 3) / len(user)
            else:
                if tokens <= MAX_INPUT_TOKENS:
                    break
                ratio = MAX_INPUT_TOKENS / tokens

            tail = user[-TAIL_KEEP_CHARS:]
            body = user[:-TAIL_KEEP_CHARS]
            keep = max(1000, int(len(body) * ratio * 0.92))
            if keep >= len(body):
                break
            elided = len(body) - keep
            user = (
                body[:keep]
                + f"\n\n[... {elided:,} characters of execution evidence elided to fit "
                  f"the model input limit; the tasks below this point were not shown ...]\n\n"
                + tail
            )
            truncated = True
            tokens = self._count_input_tokens(system, user)

        return user, truncated

    def build_evidence(
        self,
        batch: RolloutBatch,
        horizon_fraction: float,
        max_chars_per_step: int = 1200,
    ) -> str:
        """Render the batch as learning evidence at the configured credit horizon."""
        blocks: list[str] = []
        for outcome in batch.outcomes:
            traj = load_trajectory(
                outcome.trial_dir, outcome.task_name, horizon_fraction
            )
            status = "SOLVED" if outcome.solved else "FAILED"
            block = (
                f"{'=' * 70}\n"
                f"TASK: {outcome.task_name}  |  REWARD: {outcome.reward:.3f}  |  {status}\n"
            )
            if outcome.exception:
                block += f"ERROR: {outcome.exception}\n"
            block += render_for_optimizer(
                traj, max_chars_per_step=max_chars_per_step
            )
            blocks.append(block)
        return "\n".join(blocks)

    def build_summary_evidence(self, batch: RolloutBatch, horizon_fraction: float) -> str:
        """Structural summary with no verbatim task output.

        Last-resort tier when the optimizer declines the raw traces. A single
        security-flavored benchmark task (e.g. an XXE-parser fixture) can trigger
        a refusal that zeroes out the whole batch's update, so this preserves the
        signal harness optimization actually needs -- did it solve, in how many
        steps, where did it stall, what failed -- while dropping the task content
        that is not ours to relay.
        """
        lines: list[str] = []
        for outcome in batch.outcomes:
            traj = load_trajectory(
                outcome.trial_dir, outcome.task_name, horizon_fraction
            )
            status = "SOLVED" if outcome.solved else "FAILED"
            n_cmds = sum(
                1 for s in traj.steps if s.get("tool_calls") or s.get("observation")
            )
            lines.append(
                f"- task={outcome.task_name} reward={outcome.reward:.2f} {status} "
                f"steps_shown={traj.n_kept_steps}/{traj.n_total_steps} "
                f"commands_issued={n_cmds}"
                + (f" error={outcome.exception[:100]}" if outcome.exception else "")
            )
        return (
            "Structural summary only (raw traces withheld for this batch):\n"
            + "\n".join(lines)
        )

    def step(
        self,
        iteration: int,
        current_artifact: str,
        batch: RolloutBatch,
        horizon_fraction: float,
    ) -> OptimizerStep:
        evidence = self.build_evidence(batch, horizon_fraction)
        user_msg = (
            f"## CURRENT ARTIFACT\n<artifact>\n{current_artifact}\n</artifact>\n\n"
            f"## EXECUTION EVIDENCE "
            f"({len(batch.outcomes)} tasks, solve rate {batch.solve_rate:.2f})\n\n"
            f"{evidence}\n\n"
            "Rewrite the artifact to improve held-out performance. "
            "Output only the new artifact inside <artifact> tags."
        )

        user_msg, _was_truncated = self._fit_prompt(SYSTEM_PROMPT, user_msg)
        response = self._message(SYSTEM_PROMPT, user_msg)
        raw = response.text

        match = ARTIFACT_RE.search(raw)
        new_artifact = match.group(1).strip() if match else current_artifact

        # Guard: never ship an artifact that dropped a required placeholder.
        for placeholder in set(re.findall(r"\{\{\s*\w+\s*\}\}", current_artifact)):
            key = re.sub(r"\s+", "", placeholder)
            if key not in re.sub(r"\s+", "", new_artifact):
                new_artifact = current_artifact
                break

        return OptimizerStep(
            iteration=iteration,
            old_artifact=current_artifact,
            new_artifact=new_artifact,
            prompt_chars=len(user_msg),
            batch_tasks=[o.task_name for o in batch.outcomes],
            batch_solve_rate=batch.solve_rate,
            raw_response=raw,
        )


def write_artifact(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@dataclass
class CodeOptimizerStep:
    iteration: int
    artifact: CodeArtifact
    changed: bool
    prompt_chars: int
    batch_tasks: list[str]
    batch_solve_rate: float
    rejected: list[str]        # validation errors from discarded candidates
    evidence_tier: str = "full"   # full | condensed | summary-only | excluded[...]
    excluded_tasks: list[str] = field(default_factory=list)
    inner_retries_used: int = 0
    raw_response: str = ""


class CodeArtifactOptimizer(ArtifactOptimizer):
    """Rewrites a directory of agent source instead of a single prompt file.

    Candidates are validated statically before they are accepted. A syntactically
    broken agent would burn a whole rollout batch at real cost per task, so on a
    validation failure we retry with the error fed back, and fall back to the
    previous artifact if the retries are exhausted.
    """

    def __init__(
        self,
        *args: Any,
        max_retries: int = 2,
        inner_retries: int = 4,
        spec: Any = None,
        system_prompt: str = CODE_SYSTEM_PROMPT,
        artifact_cls: Any = CodeArtifact,
        # What the artifact IS, and what the player is being asked to do with it.
        # Both are player-specific: telling a detector-writing adversary to
        # "rewrite the agent's code" is simply the wrong instruction, and the
        # minimax games reuse this class for the harness, the detector, and the
        # task proposer.
        artifact_label: str = "AGENT CODE",
        task_instruction: str = (
            "Rewrite the agent's code to improve held-out performance. "
            "Emit every file the agent needs, each in its own <file> block."
        ),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt
        self.artifact_cls = artifact_cls
        self.artifact_label = artifact_label
        self.task_instruction = task_instruction
        self.max_retries = max_retries
        # Contract violations (wrong function name, wrong arity, unresolved
        # import, syntax error) are cheap to detect and cheap to fix, so they get
        # their own retry budget that does NOT consume a learning-loop iteration.
        # A rollout batch has already been paid for by the time we get here;
        # burning it because the model wrote `build_scaffold(task)` instead of
        # `build_scaffold()` would waste real money on a formatting slip.
        self.inner_retries = inner_retries
        from hopt.artifact import RUN_AGENT_SPEC

        self.spec = spec or RUN_AGENT_SPEC
        # Tasks whose traces the optimizer declined to process. Identified once
        # by probe, then excluded from every later batch in the run. This is
        # exclusion, not circumvention: content the model declines is simply not
        # resent, and the exclusion is recorded so affected iterations are
        # visible in analysis.
        # Persisted across runs: discovery costs several large API calls, and a
        # per-cell denylist would rediscover the same task in all 24 cells.
        self._denylist_path = RESULTS_DIR / "refusal_denylist.json"
        self.refusal_denylist: set[str] = self._load_denylist()

    def _load_denylist(self) -> set[str]:
        try:
            return set(json.loads(self._denylist_path.read_text()))
        except (OSError, json.JSONDecodeError):
            return set()

    def _save_denylist(self) -> None:
        try:
            self._denylist_path.parent.mkdir(parents=True, exist_ok=True)
            self._denylist_path.write_text(
                json.dumps(sorted(self.refusal_denylist), indent=2)
            )
        except OSError:
            pass

    def _probe_refusals(self, batch: RolloutBatch, horizon_fraction: float) -> set[str]:
        """Find which individual task traces the optimizer refuses.

        One small call per task, run only after a batch has already refused at
        the summary tier, so the cost is bounded and rare.
        """
        refusing: set[str] = set()
        for outcome in batch.outcomes:
            single = RolloutBatch(job_dir=batch.job_dir, outcomes=[outcome])
            evidence = self.build_summary_evidence(single, horizon_fraction)
            try:
                response = self._message(
                    self.system_prompt,
                    f"Acknowledge receipt of this summary.\n\n{evidence}",
                    max_tokens=64,
                )
            except Exception:
                continue
            if response.refused:
                refusing.add(outcome.task_name)
        return refusing

    def _compose(
        self,
        current: Any,
        batch: RolloutBatch,
        evidence: str,
        label: str | None = None,
        extra_context: str = "",
    ) -> str:
        """Assemble one optimizer message.

        Single place the prompt is built, because the retry ladder (full ->
        condensed -> summary-only -> excluded) rebuilds it four times, and having
        four copies is how the evidence tier and the message drift apart.
        """
        tag = f"{label}; " if label else ""
        return (
            f"## CURRENT {self.artifact_label}\n\n{current.render_for_optimizer()}\n\n"
            f"## EXECUTION EVIDENCE ({tag}{len(batch.outcomes)} tasks, "
            f"solve rate {batch.solve_rate:.2f})\n\n{evidence}\n\n"
            f"{extra_context}"
            f"{self.task_instruction}"
        )

    def step_code(
        self,
        iteration: int,
        current: CodeArtifact,
        batch: RolloutBatch,
        horizon_fraction: float,
        dest: Path,
        extra_context: str = "",
    ) -> CodeOptimizerStep:
        # Drop already-known refused traces before spending anything.
        excluded: list[str] = []
        if self.refusal_denylist:
            kept = [o for o in batch.outcomes if o.task_name not in self.refusal_denylist]
            excluded = [o.task_name for o in batch.outcomes if o.task_name not in {k.task_name for k in kept}]
            if kept and len(kept) < len(batch.outcomes):
                batch = RolloutBatch(job_dir=batch.job_dir, outcomes=kept)

        # Proactive context cap. The bfull x full corner produced a 1.2M-token
        # prompt (over the 1M input limit) and failed the whole cell: the artifact
        # grows across iterations AND every task contributes a full trace, so the
        # two compound. Condense *before* sending rather than discovering the
        # ceiling with a 400.
        evidence = self.build_evidence(batch, horizon_fraction)
        code_chars = len(current.render_for_optimizer())
        budget_chars = MAX_PROMPT_CHARS - code_chars
        if len(evidence) > budget_chars:
            evidence = self.build_evidence(batch, horizon_fraction, max_chars_per_step=200)
            evidence_tier_initial = "condensed(precap)"
            if len(evidence) > budget_chars:
                evidence = self.build_summary_evidence(batch, horizon_fraction)
                evidence_tier_initial = "summary-only(precap)"
        else:
            evidence_tier_initial = "full"
        base_msg = self._compose(current, batch, evidence, extra_context=extra_context)

        system_prompt = self.system_prompt + self.spec.contract_text()
        rejected: list[str] = []
        raw = ""
        n_inner = 0
        n_refusals = 0
        evidence_tier = evidence_tier_initial
        user_msg = base_msg
        # +3 attempts: full, condensed, summary-only, then one more so the
        # post-exclusion evidence is actually sent rather than just built.
        for attempt in range(self.max_retries + 3 + self.inner_retries):
            user_msg, was_truncated = self._fit_prompt(system_prompt, user_msg)
            if was_truncated:
                evidence_tier = f"{evidence_tier}+truncated"
                rejected.append("prompt exceeded input window; evidence tail elided")
            try:
                response = self._message(system_prompt, user_msg)
            except Exception as exc:
                # A context-length 400 is recoverable by condensing, exactly like a
                # refusal. Anything else is a real error and should surface.
                if "prompt is too long" not in str(exc):
                    raise
                rejected.append(f"context too long; condensing ({str(exc)[:80]})")
                evidence = self.build_summary_evidence(batch, horizon_fraction)
                evidence_tier = "summary-only(overflow)"
                base_msg = self._compose(
                    current, batch, evidence, "summary-only", extra_context
                )
                user_msg = base_msg
                continue
            raw = response.text
            files = {m.group(1).strip(): m.group(2) for m in FILE_RE.finditer(raw)}

            # Distinguish *why* nothing usable came back. Treating a refusal or a
            # length cutoff as "no <file> blocks" hides the cause and makes the
            # retry pointless -- the pilot re-sent an identical refused prompt
            # three times and reported it as a formatting problem.
            stop = response.stop_reason
            if stop in {"refusal", "max_tokens"} or not files:
                if stop == "refusal":
                    # Escalating fallback: re-sending identical content refuses
                    # again, so each retry strips more raw task output. Tier 3
                    # carries no verbatim task content at all.
                    n_refusals += 1
                    if n_refusals == 1:
                        evidence = self.build_evidence(
                            batch, horizon_fraction, max_chars_per_step=200
                        )
                        label = "condensed"
                    elif n_refusals == 2:
                        evidence = self.build_summary_evidence(batch, horizon_fraction)
                        label = "summary-only"
                    else:
                        # Even the structural summary was declined: identify the
                        # specific traces and drop them for the rest of the run.
                        offenders = self._probe_refusals(batch, horizon_fraction)
                        self.refusal_denylist |= offenders
                        self._save_denylist()
                        excluded.extend(sorted(offenders))
                        kept = [
                            o for o in batch.outcomes if o.task_name not in offenders
                        ]
                        if not kept:
                            rejected.append(
                                "every task trace in this batch was refused; "
                                "no update possible"
                            )
                            break
                        batch = RolloutBatch(job_dir=batch.job_dir, outcomes=kept)
                        evidence = self.build_evidence(
                            batch, horizon_fraction, max_chars_per_step=200
                        )
                        label = f"excluded[{','.join(sorted(offenders))}]"
                    rejected.append(
                        f"optimizer REFUSED the learning context "
                        f"(stop_reason=refusal); retrying with {label} evidence"
                    )
                    evidence_tier = label
                    base_msg = self._compose(
                        current, batch, evidence, label, extra_context
                    )
                    user_msg = base_msg
                    continue
                if stop == "max_tokens":
                    rejected.append(
                        f"response hit max_tokens ({self.max_tokens}) before emitting "
                        "a complete <file> block"
                    )
                else:
                    rejected.append(f"no <file> blocks found (stop_reason={stop})")
            else:
                try:
                    artifact = self.artifact_cls.from_files(files, dest, self.spec)
                    return CodeOptimizerStep(
                        iteration=iteration,
                        artifact=artifact,
                        changed=files != current.files(),
                        prompt_chars=len(user_msg),
                        batch_tasks=[o.task_name for o in batch.outcomes],
                        batch_solve_rate=batch.solve_rate,
                        rejected=rejected,
                        evidence_tier=evidence_tier,
                        excluded_tasks=excluded,
                        inner_retries_used=n_inner,
                        raw_response=raw,
                    )
                except ArtifactError as exc:
                    # Contract violation: retry immediately with the exact error,
                    # without consuming one of the outer attempts.
                    rejected.append(f"contract violation: {exc}")
                    if n_inner < self.inner_retries:
                        n_inner += 1
                        user_msg = (
                            base_msg
                            + f"\n\n## YOUR PREVIOUS OUTPUT WAS REJECTED (attempt "
                              f"{n_inner}/{self.inner_retries})\n{exc}\n\n"
                              f"Required entrypoint: {self.spec.signature()}\n"
                              "Re-emit EVERY file, complete, in <file> blocks."
                        )
                        continue

            if attempt < self.max_retries:
                user_msg = (
                    base_msg
                    + f"\n\n## YOUR PREVIOUS ATTEMPT WAS REJECTED\n{rejected[-1]}\n"
                    "Fix it and re-emit every file."
                )

        # All attempts failed validation: keep the last good artifact rather than
        # spending a batch on code we know is broken.
        kept = current.copy_to(dest)
        return CodeOptimizerStep(
            iteration=iteration,
            artifact=kept,
            changed=False,
            prompt_chars=len(user_msg),
            batch_tasks=[o.task_name for o in batch.outcomes],
            batch_solve_rate=batch.solve_rate,
            rejected=rejected,
            evidence_tier=evidence_tier,
            excluded_tasks=excluded,
            inner_retries_used=n_inner,
            raw_response=raw,
        )
