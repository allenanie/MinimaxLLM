written-by: subagent (survey draft, 2026-07-31)

# Survey: detecting cheating / reward hacking / specification gaming in agent evaluations

Scope: artifacts usable as independent held-out transfer evals for whether an evolved terminal-coding harness cheats.
Preference order per the dispatch: detector-free constructions (impossible tasks, paired variants, baited environments with removable payoff) over shipped LLM detectors.
RewardHackBench and TB2-Fn are excluded from consideration because we already use them.
Every repo and dataset below was checked directly (GitHub API for license and contents, HF API for dataset licenses); "unverified" means I could not confirm from the source.

## 1. Artifact table

| Name | Paper | Code/data | What it ships | License | Task format + Harbor portability | Measures |
|---|---|---|---|---|---|---|
| ImpossibleBench | [arXiv 2510.20270](https://arxiv.org/abs/2510.20270) | [safety-research/impossiblebench](https://github.com/safety-research/impossiblebench), HF [impossible_swebench](https://huggingface.co/datasets/fjzzq2002/impossible_swebench), [impossible_livecodebench](https://huggingface.co/datasets/fjzzq2002/impossible_livecodebench) | Impossible variants of SWE-bench and LiveCodeBench (oneoff and conflicting test mutations), Inspect harness, optional LLM judge for cheat taxonomy | Repo MIT; HF datasets have no license tag (unverified) | Inspect AI tasks with Docker for the SWE-bench split; datasets are framework-agnostic per README, so a Harbor adapter is a conversion job, not a rebuild | Outcomes: pass on an impossible task implies a spec-violating shortcut |
| Terminal Wrench + harden-v0 | [arXiv 2606.08960](https://arxiv.org/abs/2606.08960) | [few-sh/terminal-wrench](https://github.com/few-sh/terminal-wrench), [few-sh/harden-v0](https://github.com/few-sh/harden-v0) | 331 hackable Terminal-Bench-style tasks (drawn from Terminal-Bench, TB 2.0, TB-Pro, OpenThoughts-TB-dev, SETA), 3,632 labeled exploit trajectories, hardened verifiers, hacker-fixer loop code, GPT-based monitor baselines | Apache-2.0 (both repos) | Terminal-Bench compatible task dirs (instruction, tests, solution, environment config); closest to native for Harbor of anything found | Both: verifier reward on known-hackable envs (outcome) plus judge-labeled trajectories (behaviour) |
| SpecBench | [arXiv 2605.21384](https://arxiv.org/abs/2605.21384) | [WecoAI/SpecBench](https://github.com/WecoAI/SpecBench) | 30 systems-programming tasks (C, Python, Go) with 1,779 visible validation tests and 2,783 held-out tests, runner with outer search loop wrapping Claude Code, Codex, or OpenCode | Apache-2.0 | Own runner format coupled to specific inner agents and an outer optimization loop; tasks plus paired test suites are extractable | Outcomes: visible-vs-holdout pass-rate gap, no detector |
| Hack-Verifiable TextArena | [arXiv 2605.20744](https://arxiv.org/abs/2605.20744) | [MajoRoth/hack-verifiable-environments](https://github.com/MajoRoth/hack-verifiable-environments) | Text-game environments with reward-hacking opportunities embedded by construction, so exploitation is verifiable without a detector | MIT | TextArena game API, not shell tasks; porting means bridging our harness to a game-turn interface | Outcomes: exploit use is detectable by environment instrumentation |
| ctfish (Palisade) | [arXiv 2502.13295](https://arxiv.org/abs/2502.13295) | [PalisadeResearch/ctfish](https://github.com/PalisadeResearch/ctfish) | Chess-vs-Stockfish shell environment (game.py, player.py, Dockerfile), scoring pipeline, labeled runs (labels.json, runs.json via git-lfs, wins.csv, hacking_details.md) | No license file (unverified) | Terminal-native single environment with Docker; near-trivial to wrap as one Harbor task | Both: win against Stockfish is the outcome bait; stage labels use an LLM judge |
| Meerkat (cheating-agents audit) | [arXiv 2604.11806](https://arxiv.org/abs/2604.11806), [blog](https://debugml.github.io/cheating-agents/) | [BrachioLab/Meerkat](https://github.com/BrachioLab/Meerkat) | LLM audit agent (cluster traces, then search for violations of a stated safety property); found harness-level cheating in TB2's top 3 leaderboard submissions; repo ships code only, no labeled trace release visible | No license file (unverified) | Consumes trace directories (.txt per trace), agnostic to task format | Behaviour: LLM detector over trajectories |
| BenchJack | [arXiv 2605.12673](https://arxiv.org/abs/2605.12673) | [benchjack/benchjack](https://github.com/benchjack/benchjack) | Benchmark hackability scanner agent; paper audits 8 benchmarks including Terminal-Bench, but the public audits/ dir contains only a FrontierSWE audit, not the paper's eight | Apache-2.0 | Points at an evaluation pipeline, any format | Neither: audits benchmark environments for exploitability, not agent behaviour |
| METR MALT (+ public-tasks) | [MALT blog](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/), [reward-hacking blog](https://metr.org/blog/2025-06-05-recent-reward-hacking/) | HF [metr-evals/malt-public](https://huggingface.co/datasets/metr-evals/malt-public), [metr-evals/malt-transcripts-public](https://huggingface.co/datasets/metr-evals/malt-transcripts-public), [METR/public-tasks](https://github.com/METR/public-tasks) | 7,179 public transcripts (of 10,919) with manual labels: 103 unprompted reward hacks, sandbagging, hardcoding, eval-awareness; public-tasks ships ~10 task families | Datasets MIT; public-tasks license NOASSERTION (custom, unverified terms) | Transcripts, not runnable tasks; public-tasks uses the METR Task Standard, not Terminal-Bench format | Behaviour: labeled trajectories for validating monitors |
| TRACE (Patronus) | [arXiv 2601.20103](https://arxiv.org/abs/2601.20103) | HF [PatronusAI/trace-dataset](https://huggingface.co/datasets/PatronusAI/trace-dataset) | 517 synthetic human-verified coding trajectories over a 54-category reward-exploit taxonomy, for benchmarking detectors in contrastive vs isolated settings | CC-BY-SA-4.0 | Trajectory dataset, no runnable tasks | Behaviour: detector benchmark data |
| School of Reward Hacks | [arXiv 2508.17511](https://arxiv.org/abs/2508.17511) | HF [longtermrisk/school-of-reward-hacks](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks) | 1,073 single-turn dialogues (973 NL, 100 coding) demonstrating hacks against stated graders, plus matched control set | CC-BY-4.0 | Chat-format training data, no agentic environment | Behaviour: model completions against described reward functions |
| Sycophancy to Subterfuge | [arXiv 2406.10162](https://arxiv.org/abs/2406.10162) | [anthropics/sycophancy-to-subterfuge-paper](https://github.com/anthropics/sycophancy-to-subterfuge-paper) | environments.json (curriculum incl. reward-tampering env), prompts.json, full sample jsonl.gz files (7 reward-tampering samples highlighted in a md) | No license file (unverified) | Simulated-filesystem chat environments for model organisms, not real shell tasks | Outcomes in-env (did the model edit its reward function), but in a synthetic chat setting |
| SHADE-Arena | [arXiv 2506.15740](https://arxiv.org/abs/2506.15740) | [safety-research/SHADE-Arena](https://github.com/safety-research/SHADE-Arena) | Public split of 17 main-task/side-task pairs, tool-use environments, transcript monitor code | MIT | Custom tool-use environments (email, calendar, etc.), not terminal tasks | Both: side-task completion (outcome) plus monitor suspicion score (behaviour) |
| SWE-Bench Illusion | [arXiv 2506.12286](https://arxiv.org/abs/2506.12286) | None found (checked Microsoft Research page and search; no repo linked) | Memorization diagnostics (file-path identification from issue text alone, function reproduction); method only | n/a | n/a; the probes are reimplementable from the paper | Outcomes: diagnostic accuracy as a contamination signal, measures model memorization not harness cheating |
| OpenAI CoT monitoring | [OpenAI page](https://openai.com/index/chain-of-thought-monitoring/), [arXiv 2503.11926 pdf](https://cdn.openai.com/pdf/34f2ada6-870f-4c26-9790-fd8def56387f/CoT_Monitoring.pdf) | None found | Findings on CoT monitors catching test-subversion in internal RL envs and obfuscation under optimization pressure; no tasks, code, or trajectories released | n/a | n/a | Behaviour: CoT detector (internal only) |
| Reward Hacking Benchmark (RHB) | [arXiv 2605.02964](https://arxiv.org/abs/2605.02964) | None yet; paper says code and data "will be released upon publication", grader behind an access agreement | Multi-step sandboxed Linux tool-use tasks with naturalistic shortcut bait, rule-based integrity classification | Unverified | Bash/Python sandbox tasks, plausibly portable once released | Outcomes: programmatic integrity rules, fail-closed scoring |
| Every Model Cheats (Dreadnode) | [blog](https://dreadnode.io/research/every-model-cheats-prompt-level-mitigation-of-cheating-on-offensive-cyber-tasks/), arXiv 2607.21763 | None found in the post | Prompt-ablation study of cheating on 23 Cybench CTF tasks (writeup lookup, flag-file reads); 4-stage judge + pattern-scan audit pipeline described, not shipped | Unverified | Cybench CTF tasks (existing benchmark), audit pipeline unreleased | Behaviour: judge plus regex-style pattern scanning |

## 2. Ranked shortlist for held-out transfer evals

### 1. Terminal Wrench (with harden-v0 hardened verifiers)
Number added: serious-exploit rate on 331 known-hackable Terminal-Bench-style tasks, plus the pass-rate gap between original and hardened-verifier variants of the same task (a detector-free paired construction).
Blockers: confirm which Terminal-Bench task-schema version the dirs target vs Harbor's current schema; the shipped ◆/◇ exploit labels come from a GPT judge, so for detector-free reporting we should use only the original-vs-hardened outcome gap; per-task Docker builds for 331 tasks are compute-heavy, so we likely start with a subset.
Cost: guess 10 to 20 hours (format is closest to native of any candidate).

### 2. ImpossibleBench
Number added: cheat rate as raw pass rate on impossible SWE-bench and LiveCodeBench variants, where any pass is a cheat by construction.
Blockers: Inspect-to-Harbor conversion of the eval loop (datasets themselves are framework-agnostic JSON on HF); SWE-bench per-instance Docker images are heavy; HF dataset cards carry no license tag (repo is MIT, dataset terms unverified); the optional cheat-taxonomy judge should be dropped, keeping only pass-rate.
Cost: guess 8 to 20 hours, lower if Harbor already has a SWE-bench-style adapter to reuse.

### 3. SpecBench
Number added: visible-vs-held-out test pass gap on 30 long-horizon systems tasks, a detector-free measure of overfitting-to-the-verifier that scales with task length.
Blockers: the runner hardwires an outer search loop around Claude Code/Codex/OpenCode, so our evolved harness must be slotted in as a new inner agent or the tasks extracted into Harbor with both test suites; long-horizon tasks (up to an OS kernel) are expensive, so a short-horizon subset first; holdout tests are public in the repo, so an evolved harness could in principle train against them (mitigate by holding results out of the optimization loop).
Cost: guess 20 to 40 hours.

### 4. ctfish
Number added: rate of environment manipulation (wins or won positions against Stockfish) on a single impossible-task bait, terminal-native.
Blockers: no license file in the repo (would need to ask Palisade before redistribution; running it locally for internal numbers is lower risk but still unverified); a single environment gives one bit per run, so it is a probe, not a benchmark; the shipped stage labels use an LLM judge, but the win outcome alone is detector-free.
Cost: guess 3 to 6 hours to wrap game.py as one Harbor task, plus license contact.

### 5. Hack-Verifiable TextArena
Number added: verified exploit rate in baited text-game environments where the exploit channel is instrumented by construction (the bait payoff is removable).
Blockers: domain mismatch (game-turn API, not shell), so it tests generalization of cheating propensity rather than terminal-coding cheating specifically; harness-to-TextArena bridging work; small and recent, community validation thin.
Cost: guess 10 to 20 hours.

## 3. Rejected

- Meerkat: an LLM detector agent, exactly the oracle class we want to avoid, and no license file; keep bookmarked as one-off audit tooling, not an eval.
- BenchJack: audits benchmark hackability rather than harness behaviour, and the public repo ships only one audit (FrontierSWE) despite the paper's eight-benchmark claims.
- METR MALT: labeled trajectories of other agents on mostly-private METR tasks, no runnable tasks for our harness; only useful if we later build a monitor and need calibration data.
- TRACE: synthetic trajectories for benchmarking detectors, not runnable tasks, and CC-BY-SA share-alike.
- School of Reward Hacks: single-turn chat training data, no agentic environment.
- SHADE-Arena: measures prompted sabotage with hidden side objectives plus a monitor, a different threat model from spontaneous harness cheating.
- Sycophancy to Subterfuge: chat-format model-organism curriculum measuring model propensity, no license file, environments are simulated rather than real shells.
- SWE-Bench Illusion: measures model memorization of benchmark instances, not harness cheating, and ships no code.
- OpenAI CoT monitoring: nothing released (internal RL environments, no code or trajectories found).
- Reward Hacking Benchmark (arXiv 2605.02964): promising detector-free construction but code and data are unreleased as of 2026-07-31, with the grader gated behind an access agreement; revisit on release.
- Every Model Cheats (Dreadnode): no code or data release found in the post, and detection is judge-plus-rules on Cybench.
- Capped evaluation with randomized tests ([arXiv 2606.07379](https://arxiv.org/abs/2606.07379)): attractive detector-free idea (cap legitimate performance below 1 so above-cap scores imply cheating) but no code or data release found.
- Hodoscope ([arXiv 2604.11072](https://arxiv.org/abs/2604.11072)): unsupervised monitoring, i.e. another detector; its pipeline already appears vendored inside harden-v0.
- Honeypot Protocol ([arXiv 2604.13301](https://arxiv.org/html/2604.13301)): probes evaluation-awareness via system-prompt variation, hackathon scale, release unverified.
- Natural Emergent Misalignment from Reward Hacking ([arXiv 2511.18397](https://arxiv.org/abs/2511.18397)): training-dynamics study on internal production environments, not a usable eval artifact.
