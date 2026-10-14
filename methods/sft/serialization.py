"""Native tokenization with loss on the assistant payload only."""
import copy
import unicodedata
from .common import require, validate_row


class QwenSerializer:
    def __init__(self, snapshot, max_length=2048):
        self.max_length = max_length
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = 'right'
        self.pad_token_id = self.tokenizer.pad_token_id
        require(self.pad_token_id is not None, 'Missing pad token')

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def render(self, messages, generate):
        kwargs = dict(add_generation_prompt=generate, enable_thinking=False, tools=None)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        require(self.decode(ids) == unicodedata.normalize('NFC', text), 'Tokenizer content loss')
        return text, ids

    def serialize(self, row):
        validate_row(row)
        messages, payload = row['ordinary_student_messages'], row['target_assistant_content']
        prefix, prefix_ids = self.render(messages, True)
        text, ids = self.render(messages + [dict(role='assistant', content=payload)], False)
        suffix = '<|im_end|>\n'
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        require(text == prefix+payload+suffix and ids[:len(prefix_ids)] == prefix_ids
                and ids[-len(suffix_ids):] == suffix_ids, 'Assistant boundary mismatch')
        payload_ids = ids[len(prefix_ids):-len(suffix_ids)]
        require(payload_ids and self.decode(payload_ids) == payload, 'Payload normalized or lost')
        require(payload_ids == self.tokenizer.encode(payload, add_special_tokens=False), 'Boundary token merge')
        require(len(ids) <= self.max_length, 'Target exceeds max_sequence_length; truncation is forbidden')
        labels = [-100]*len(prefix_ids) + payload_ids + [-100]*len(suffix_ids)
        return dict(request_id=row['request_id'], category=row['category'],
                    target_path=dict(input_ids=ids, labels=labels, truncated=False))

    def save_pretrained(self, path):
        self.tokenizer.save_pretrained(path)


def mask_payload(prompt, payload, full, eos_id=2, cap=2048):
    require(prompt and payload and full == prompt+payload+[eos_id], 'Native assistant boundary mismatch')
    require(eos_id == 2 and len(full) <= cap, 'No truncation permitted / wrong EOS')
    return [-100]*len(prompt)+payload+[-100]


class MistralSerializer:
    def __init__(self, snapshot, max_length=2048):
        self.max_length = max_length
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_common.protocol.instruct.validator import ValidationMode
        from transformers.tokenization_mistral_common import MistralCommonBackend
        self.common = MistralTokenizer.from_file(str(snapshot/'tekken.json'))
        self.training = MistralTokenizer.from_file(str(snapshot/'tekken.json'), mode=ValidationMode.finetuning)
        self.hf = MistralCommonBackend.from_pretrained(str(snapshot), local_files_only=True)
        self.hf_training = MistralCommonBackend(tokenizer_path=str(snapshot/'tekken.json'), mode=ValidationMode.finetuning)
        self.pad_token_id = 2

    def ids(self, messages, completed):
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        common = self.training if completed else self.common
        hf = self.hf_training if completed else self.hf
        ids = common.encode_chat_completion(ChatCompletionRequest.from_openai(messages=copy.deepcopy(messages))).tokens
        actual = hf.apply_chat_template(copy.deepcopy(messages), tokenize=True, return_dict=False,
            add_generation_prompt=not completed, truncation=False)
        require(ids == actual, 'HF/native Mistral token disagreement')
        return ids

    def serialize(self, row):
        validate_row(row)
        messages, payload = row['ordinary_student_messages'], row['target_assistant_content']
        prompt = self.ids(messages, False)
        full = self.ids(messages+[dict(role='assistant', content=payload)], True)
        base = self.common.instruct_tokenizer.tokenizer
        payload_ids = base.encode(payload, bos=False, eos=False)
        require(self.common.decode(payload_ids) == payload, 'Assistant round trip changed')
        labels = mask_payload(prompt, payload_ids, full, base.eos_id, cap=self.max_length)
        return dict(request_id=row['request_id'], category=row['category'],
                    target_path=dict(input_ids=full, labels=labels, truncated=False))

    def save_pretrained(self, path):
        self.hf.save_pretrained(path)


def serialize_all(config, snapshot, train, validation):
    serializer_class = MistralSerializer if config['model_family'] == 'mistral3' else QwenSerializer
    serializer = serializer_class(snapshot, max_length=config['max_sequence_length'])
    return serializer, {split: [serializer.serialize(row) for row in rows]
                        for split, rows in [('train', train), ('validation', validation)]}
