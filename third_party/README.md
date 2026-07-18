# Third-party research repositories

All upstream paper code is isolated under `third_party/upstream/<name>` and
checked out at the exact commit recorded in `upstream-lock.json`.

Upstream code is reference-only:

- core `rl_attack` code must not import it;
- do not edit an upstream checkout;
- do not install its dependencies into the core environment;
- create one Conda/venv environment per repository;
- port a method into `src/rl_attack` only with source, commit, and license
  provenance recorded.

Run `scripts/sync_upstream.ps1` after reviewing the lock file.

The lock records the upstream commit, submodule commits, license status, role,
and environment profile. `sync_upstream.ps1 -VerifyOnly` fails on a missing or
different checkout. A repository with an unknown root license is strictly
read-only: an algorithm may be reimplemented from the paper, but its code must
not be copied into this project without a separate permission review.
