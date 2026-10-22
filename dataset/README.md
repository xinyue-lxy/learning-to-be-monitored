# Paired Intervention Dataset

3,810 control–intervention pairs across 12 categories: evidence cues, answer suggestions, sandbagging, and deception.

## Files

- `data.json`: task records, interventions, answers, and reference completions.
- `clean_loader.py`: loader for SFT, OPSD, and GRPO task views.
- `sources.json`: source attribution and license labels.
- `README.md`: usage and dataset notes.


## Usage

Python 3.9+, no dependencies. Run from this folder:

```python
from clean_loader import select

pairs = select("data.json", method="opsd")
control, intervention = next(pairs), next(pairs)
print(control["task"])
print(intervention["task"])
```

Send only `task` to the model; other fields may contain answers. For reference completions, use `method="sft", arms=(1,)`. Filter with `categories=["cue_answer.math"]`.

`arms=(0, 1)` selects control and intervention views; OPSD requires both. Each view includes `task`, `answer_type`, `locator`, and `split`. SFT adds `completion`; GRPO adds `private_reward_reference` for your scorer.

## Data

`data.json` maps categories to record lists. Each record contains public task text, an `intervention`, a `ground_truth_answer`, and draft `reference_responses`. References remain correct or honest despite contrary interventions. Condition-control completions are null. Keep paired arms together when splitting.

This dataset combines material from multiple sources with different license terms. Source mappings and recorded license labels are provided in `sources.json`; these labels have not all been independently verified. No single license is asserted for all dataset content. Permissions for some sources and licensing of added annotations remain to be resolved before public redistribution.
