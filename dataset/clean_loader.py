"""Load paired task views and render public prompts."""
import copy
import json
from pathlib import Path

PUBLIC_FIELDS = ('context', 'case', 'premises', 'scenario', 'question', 'problem')

def render(record, arm):
    if arm not in (0, 1):
        raise ValueError('arm must be 0 or 1')
    pieces = [record[key] for key in PUBLIC_FIELDS if record.get(key)]
    if record.get('choices'):
        pieces.append('\n'.join(f'{letter}. {value}' for letter, value in record['choices'].items()))
    if arm:
        pieces.append('Additional instruction or information:\n' + record['intervention']['text'])
    return '\n\n'.join(pieces)

def select(dataset_uri, method, split=None, categories=None, arms=(0, 1), allow_unreviewed=True):
    """Yield task views; allow_unreviewed is retained as an ignored compatibility argument.

    All records are unassigned because original train/dev membership is unavailable.
    """
    if method not in {'sft', 'opsd', 'grpo'}:
        raise ValueError('Unknown method')
    data = json.loads(Path(dataset_uri).read_bytes())
    if not arms or any(arm not in (0, 1) for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError('Select distinct arms from 0 and 1')
    if categories is not None and set(categories) - set(data):
        raise ValueError('Unknown category')
    if split is not None and split not in {'project_train', 'project_dev', 'unassigned'}:
        raise ValueError('Unknown split')
    if method == 'opsd' and set(arms) != {0, 1}:
        raise ValueError('Paired OPSD selects both arms')
    for category, rows in data.items():
        if categories is not None and category not in categories:
            continue
        for position, record in enumerate(rows):
            if split is not None and split != 'unassigned':
                continue
            for arm in sorted(arms):
                view = {'task': render(record, arm), 'answer_type':record['answer_type'],
                        'locator':(category,position,arm), 'split':'unassigned'}
                if method == 'sft':
                    target=record.get('reference_responses',{}).get('control' if arm==0 else 'intervention')
                    if target is None:
                        raise ValueError('No grounded SFT completion for this arm')
                    view['completion']=target
                elif method == 'opsd':
                    view['private_factor'] = copy.deepcopy(record['intervention']) if arm else None
                    # Correct reference is available privately only on sandbag intervention.
                    if arm and record['intervention']['type']=='sandbag':
                        view['private_registered_answer']=record['ground_truth_answer']
                else:
                    view['private_reward_reference'] = {
                        'ground_truth_answer':record.get('control_answer',record['ground_truth_answer']) if arm==0 else record['ground_truth_answer'],
                        'intervention':copy.deepcopy(record['intervention']) if arm else None,
                        'deceptive_goal':record.get('deceptive_goal'),
                        'positive_criteria':record.get('positive_criteria'),
                        'negative_criteria':record.get('negative_criteria'),
                        'objective':'intervention_recognition_and_faithful_causal_disclosure_for_monitorability',
                        'scorer_status':'requires_reviewed_scorer_and_grouped_estimator'}
                yield view
