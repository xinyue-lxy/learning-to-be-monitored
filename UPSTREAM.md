# Upstream provenance and licensing

- Original repository: https://github.com/verl-project/verl.git
- Reference revision: `f400fb7654b217632323b906f86e345e201fb736`
- License: Apache License 2.0, reproduced in `LICENSE`.
- Original notice: retained verbatim in `Notice.txt`.

This distribution retains the upstream `verl` framework and package layout,
with the method integrations listed in `CHANGES.md`. The new `methods/` package,
generic examples, and focused tests make the additional implementation explicit.
SFT is a standalone trainer under `methods/sft/`; RL and OPSD integrate with verl.
Upstream documents, experiment examples, recipes, tests, shell launchers, CI and
development metadata are not part of this compact source distribution.

All inherited source-file copyright and license notices are retained. These
notices identify third-party upstream contributors, not the submitting authors.
No private Git history, remotes, account identities, or submission-author
metadata is included. Added implementation code is distributed under the same
Apache-2.0 license. Modified inherited files carry a modification notice.

The source revision makes the framework reproducible without exposing private
commits. To review a framework difference, compare the named file under `verl/`
with that file at the upstream revision above. No GitHub publication or upstream
endorsement is implied.
