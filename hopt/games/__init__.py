"""Minimax games over harnesses.

Two games from ``certified_harness_opt/main.tex``, sharing one alternating loop:

* ``robust`` -- the Robust Harness Game. The adversary proposes cheating
  detectors constrained to fit an adjudicated dataset; the harness must satisfy
  them without being told what they check.
* ``selfimprove`` -- the Self-Improving Game. The adversary proposes tasks and
  verifiers; the harness must solve them.

The adversary is the only thing that differs, which is why one loop covers both.
Both games are asymmetric-information: the adversary moves second with the policy
in hand, and the harness never sees the adversary's artifact. See
``hopt/games/views.py`` for how that is enforced rather than assumed.
"""
