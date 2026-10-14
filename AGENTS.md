# Repository maintenance

- Read `README.md`, `docs/COMPONENTS.md` and the relevant component documentation
  before changing training or evaluation behavior.
- Keep the supplied `methods/sft/` Python implementation unchanged unless the
  maintainer explicitly authorizes a modification. Environment and integration
  changes belong outside those source files.
- Evaluation is a separately packaged subproject. Follow `evaluation/AGENTS.md`
  and preserve its frozen prompts, datasets, profiles, parsers, invalid handling,
  support gates and instance-first estimators. New behavior requires a separately
  reviewed policy, not an implicit migration change.
- Do not put results, weights, credentials, private paths or experiment-specific
  registrations in this source distribution.
- Do not modify shared environments or run training/inference on login nodes.
  GPU validation uses non-preemptible, explicitly typed H200/B200 allocations.
  Job submission and publication belong to the user.
- Preserve inherited copyright/license notices and component license boundaries.
