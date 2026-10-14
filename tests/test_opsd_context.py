import copy
import unittest
from verl.utils.tokenizer import normalize_token_ids
from methods.opsd.context import CATEGORY_GUIDES, render_task, response_mask, student_messages, teacher_prefix


class CharacterTokenizer:
    all_special_ids = []

    def __init__(self):
        self.last_chat_template_kwargs = None

    def decode(self, token_ids, skip_special_tokens=True):
        return ''.join((chr(token_id) for token_id in token_ids))

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {'input_ids': [ord(char) for char in text]}
        if return_offsets_mapping:
            result['offset_mapping'] = [(index, index + 1) for index in range(len(text))]
        return result

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        self.last_chat_template_kwargs = kwargs
        rendered = ''.join((f"<{message['role']}>{message['content']}" for message in messages))
        if add_generation_prompt:
            rendered += '<assistant>'
        return [ord(char) for char in rendered]

    def convert_ids_to_tokens(self, token_ids):
        return [chr(token_id) for token_id in token_ids]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

def record():
    return {'question': 'Choose a color.', 'choices': {'A': 'red', 'B': 'blue'}, 'intervention': {'type': 'answer', 'text': 'Try B.'}, 'ground_truth_answer': 'PRIVATE_REFERENCE', 'reference_responses': {'intervention': 'PRIVATE_DRAFT'}}

def metadata(category, arm=1):
    return {'category': category, 'factor_type': category.split('.')[0].split('_')[1], 'factor_text': 'FACTOR_ONLY', 'arm': arm, 'registered_answer': 'REFERENCE_ONLY'}

def test_public_render():
    r = record()
    assert render_task(r, 1) == render_task(r, 0) + '\n\nAdditional instruction or information:\nTry B.'
    assert 'Try B.' not in render_task(r, 0)
    for arm in (0, 1):
        public = str(student_messages(r, arm))
        assert 'PRIVATE_REFERENCE' not in public and 'PRIVATE_DRAFT' not in public
        assert 'disclos' not in public and 'teacher' not in public

def test_category_prefix(tokenizer, category):
    r = record()
    before = copy.deepcopy(r)
    prompt = student_messages(r, 1)
    original = copy.deepcopy(prompt)
    meta = metadata(category)
    prefix = teacher_prefix(tokenizer, prompt, meta, {'enable_thinking': False})
    meta.update(parsed_answer='FUTURE_ANSWER', y=1, response='FUTURE_RESPONSE')
    assert prefix == teacher_prefix(tokenizer, prompt, meta, {'enable_thinking': False})
    text = tokenizer.decode(prefix)
    assert category in text and CATEGORY_GUIDES[category] in text
    assert 'FUTURE_' not in text
    assert prompt == original and r == before
    student = normalize_token_ids(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    assert len(prefix) != len(student)
    assert teacher_prefix(tokenizer, prompt, metadata(category, 0), {'enable_thinking': False}) == student

def test_final_value_mask(tokenizer, answer):
    text = 'The evidence supports this conclusion.\nFinal answer: ' + answer
    ids = tokenizer.encode(text, add_special_tokens=False)
    (mask, parsed, fallback) = response_mask(tokenizer, ids)
    assert parsed == answer and (not fallback)
    assert any((v == 1 for v in mask))
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    start = text.index(answer, text.index('Final answer:') + len('Final answer:'))
    assert all((mask[i] == 0 for (i, (a, b)) in enumerate(offsets) if b > start))
    assert len(mask) == len(ids)

def test_fallback(tokenizer):
    ids = tokenizer.encode('An answer with no final marker.', add_special_tokens=False)
    (mask, parsed, fallback) = response_mask(tokenizer, ids)
    assert parsed is None and fallback and (0 < max(mask) <= 0.1)

def test_unique_guides():
    assert len(CATEGORY_GUIDES) == len(set(CATEGORY_GUIDES.values())) == 12

class Native12Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharacterTokenizer()

    def test_public_boundary(self):
        test_public_render()

    def test_all_category_prefixes(self):
        for category in CATEGORY_GUIDES:
            with self.subTest(category=category):
                test_category_prefix(self.tokenizer, category)

    def test_all_final_formats(self):
        for answer in ['A', '95', 'True', '72 W', 'I arrived late.\nI apologize.']:
            with self.subTest(answer=answer):
                test_final_value_mask(self.tokenizer, answer)

    def test_conservative_fallback(self):
        test_fallback(self.tokenizer)

    def test_category_count(self):
        test_unique_guides()

    def test_sandbag_math_disclosure_contract(self):
        from methods.opsd.context import teacher_messages
        guide = CATEGORY_GUIDES['goal_sandbag.math']
        for phrase in ['reasoning so far supports deliberate deviation', 'concrete intended submitted number', 'withdraw the earlier plan', 'do not relabel an arithmetic mistake', 'Do not initiate a wrong-answer plan']:
            self.assertIn(phrase, guide)
        public = student_messages(record(), 1)
        meta = metadata('goal_sandbag.math')
        self.assertIn(guide, teacher_messages(public, meta)[0]['content'])
        self.assertNotIn(guide, str(public))
        control = student_messages(record(), 0)
        meta['arm'] = 0
        self.assertEqual(teacher_messages(control, meta), control)

    def test_all_sandbag_targets_are_private(self):
        from methods.opsd.context import SANDBAG_DISCLOSURE_GUIDE, teacher_messages
        for category in CATEGORY_GUIDES:
            if not category.startswith('goal_sandbag.'):
                continue
            with self.subTest(category=category):
                public = student_messages(record(), 1)
                meta = metadata(category)
                teacher = teacher_messages(public, meta)
                self.assertIn(SANDBAG_DISCLOSURE_GUIDE, teacher[0]['content'])
                self.assertIn('behavioral_target_spec', teacher[0]['content'])
                self.assertNotIn('behavioral_target_spec', str(public))
                meta.update(parsed_answer='FUTURE', Y=1)
                self.assertEqual(teacher_messages(public, meta), teacher)
                meta['arm'] = 0
                control = student_messages(record(), 0)
                self.assertEqual(teacher_messages(control, meta), control)


class PreparationTests(unittest.TestCase):
    def test_pair_keeps_private_reference_out_of_both_public_views(self):
        from methods.opsd.prepare import make_pair
        example = record()
        example["category"] = "cue_answer.math"
        control, intervention = make_pair(example, 0)
        self.assertEqual(control["extra_info"]["arm"], 0)
        self.assertEqual(intervention["extra_info"]["arm"], 1)
        for row in (control, intervention):
            self.assertNotIn("PRIVATE_REFERENCE", str(row["prompt"]))
            self.assertNotIn("PRIVATE_DRAFT", str(row["prompt"]))
            self.assertEqual(row["extra_info"]["registered_answer"], "PRIVATE_REFERENCE")
