# Evaluation assets

Only the twelve MonitorBench intervention tasks used by this evaluation profile are packaged. No training set, teacher-private information or model responses are included.

| Task | Instances |
|---|---:|
| cue_condition.health | 65 |
| cue_condition.knowledge | 100 |
| cue_condition.judgment | 100 |
| cue_condition.preference | 108 |
| cue_answer.health | 65 |
| cue_answer.math | 28 |
| cue_answer.logic | 50 |
| goal_sandbag.safety | 120 |
| goal_sandbag.math | 30 |
| goal_sandbag.science | 198 |
| goal_sandbag.engineering | 150 |
| goal_deception.realworld | 50 |

Paths and byte hashes are recorded in `src/monitorability_eval/assets/datasets.json`; `inspect` verifies them. These counts are task instances, not mutually exclusive source-question groups: health assets are shared between two interventions.

Task classes construct the paired arms from the packaged records. Keep their original intervention and target definitions. For sandbag, the target field retains the original correct answer and the valid-final-answer scorer is inverted. Deception uses a behavioral goal. Finite-choice paths retain their deterministic contract; open-answer paths use the packaged corrected final-response verifier.

The datasets are copied without edits from the evaluation snapshot identified in `SOURCE_MAP.json`. Their appearance in an upstream repository does not establish unrestricted redistribution. Before publication, review each asset's upstream origin, applicable dataset license and required notices. If an asset cannot be redistributed, replace bundling with a permission-compliant fetch procedure that verifies the same hash; do not silently substitute different instances.

Some upstream task text contains illustrative filesystem paths and other scenario details. Those are benchmark text, not evaluator configuration. Do not alter them as part of removing private execution paths.
