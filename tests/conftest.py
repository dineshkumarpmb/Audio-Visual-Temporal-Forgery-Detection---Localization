"""Shared pytest configuration.

Hypothesis profiles (P15-5): `default` explores new random cases on every local run; `ci`
(set by `HYPOTHESIS_PROFILE=ci` in .github/workflows/ci.yml) is derandomized, so a CI failure
is the same failure on a laptop. Either way a failing case is printed with its reproduction
blob.
"""

from __future__ import annotations

import os

from hypothesis import settings

settings.register_profile("default", print_blob=True)
settings.register_profile("ci", derandomize=True, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
